import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType

from models.embeddings.token_embedding_provider import build_token_embedding_provider
from models.encoders import NRMSTitleEncoder, NRMSUserEncoder, BaseTextEncoder, BaseUserEncoder

logger = logging.getLogger(__name__)


class NRMS(SequentialRecommender):

    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.title_field: str = config["title_field"]
        self.vocab_size: int = dataset.num(self.title_field)
        self.padding_idx: int = int(config.get("padding_idx", 0))

        # Hyperparams
        self.word_embedding_dim = int(config.get("word_embedding_dim", 300))
        self.nb_head = int(config.get("num_heads", 16))
        self.size_per_head = int(config.get("head_dim", 16))
        self.att_dim = int(config.get("att_dim", 200))
        self.dropout = float(config.get("emb_dropout", 0.2))
        self.neg_k = int(config.get("neg_sample_num", 4))
        self.similarity = config.get("similarity", "dot")

        # Fields produced by impression dataloader
        self.cand_field = config.get("cand_field", "cand_item_id")
        self.pos_index_field = config.get("pos_index_field", "pos_index")

        # Sentence-level embedding mode
        self.use_sentence_embeddings = (
            config.get("sentence_embedding_source", None) is not None
        )

        if self.use_sentence_embeddings:
            self._init_sentence_mode(config, dataset)
        else:
            self._init_token_mode(config, dataset)

        news_dim = self.nb_head * self.size_per_head

        # User encoder (same for both embedding modes)
        self.user_encoder: BaseUserEncoder = NRMSUserEncoder(
            news_dim=news_dim,
            num_heads=self.nb_head,
            head_dim=self.size_per_head,
            att_dim=self.att_dim,
        )

        self.hidden_size = self.user_encoder.out_dim

    def _init_token_mode(self, config, dataset):
        """Default: token-level word embeddings + NRMS title encoder."""
        provider = build_token_embedding_provider(
            config=config,
            dataset=dataset,
            field=self.title_field,
            dim=self.word_embedding_dim,
        )

        pretrained_emb_matrix = provider.get_embedding_matrix(
            vocab_size=self.vocab_size,
            padding_idx=self.padding_idx,
            dtype=torch.float32,
        )

        self.news_encoder: BaseTextEncoder = NRMSTitleEncoder(
            vocab_size=self.vocab_size,
            word_embed_dim=self.word_embedding_dim,
            num_heads=self.nb_head,
            head_dim=self.size_per_head,
            att_dim=self.att_dim,
            dropout=self.dropout,
            padding_idx=self.padding_idx,
            pretrained_embeddings=pretrained_emb_matrix,
            freeze_embeddings=bool(config.get("freeze_embeddings", False)),
        )

        item_feat = dataset.get_item_feature()
        if self.title_field not in item_feat:
            raise KeyError(
                f"Missing item feature '{self.title_field}'. "
                f"Your item_feat must contain title tokens as a fixed-length TOKEN_SEQ."
            )

        title_tokens = item_feat[self.title_field]
        if not torch.is_tensor(title_tokens):
            title_tokens = torch.tensor(title_tokens, dtype=torch.long)
        title_tokens = title_tokens.long()

        if title_tokens.dim() != 2:
            raise ValueError(
                f"Expected '{self.title_field}' to be 2-D (n_items, seq_len), "
                f"got {title_tokens.dim()}-D"
            )
        self.title_len = title_tokens.size(1)
        self.register_buffer("item_title_tokens", title_tokens)

        pad_title = self.item_title_tokens[self.padding_idx]
        if (pad_title != self.padding_idx).any():
            print("Warning: item_title_tokens[padding_idx] is not all padding tokens.")

    def _init_sentence_mode(self, config, dataset):
        """Sentence embedding mode: pre-compute dense item vectors via a
        sentence-level encoder"""
        from models.embeddings.sentence_embedding_provider import build_sentence_embedding_provider

        sent_dim = int(config.get("sentence_embedding_dim", 384))
        provider = build_sentence_embedding_provider(config, dim=sent_dim)

        # Extract raw text for each item from RecBole's tokenised fields
        abstract_field = config.get("abstract_field", "abstract")
        use_abstract = bool(config.get("use_abstract", False))

        n_items = dataset.item_num
        item_texts = [
            self._get_item_text(dataset, i, abstract_field if use_abstract else None)
            for i in range(n_items)
        ]

        logger.info("Encoding %d items with sentence embeddings …", n_items)
        sent_embeddings = provider.encode(item_texts, show_progress=True)
        self.register_buffer("item_sentence_embeddings", sent_embeddings)

        # Project sentence dim to news_dim if they differ
        news_dim = self.nb_head * self.size_per_head
        if sent_dim != news_dim:
            self.sent_projection = nn.Linear(sent_dim, news_dim)
        else:
            self.sent_projection = nn.Identity()

    def _get_item_text(
        self, dataset, item_idx: int, abstract_field: str | None = None,
    ) -> str:
        """Reconstruct item text from RecBole's tokenised feature fields."""
        item_feat = dataset.get_item_feature()
        tokens: list[str] = []
        fields = [self.title_field] + ([abstract_field] if abstract_field else [])
        for field in fields:
            if field not in item_feat:
                continue
            ids = item_feat[field][item_idx]
            if torch.is_tensor(ids):
                ids = ids.cpu().numpy()
            tokens.extend(
                dataset.id2token(field, [int(t)])[0] for t in ids if int(t) != 0
            )
        return " ".join(tokens)


    def _title_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids != self.padding_idx

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        item_ids: (B,) or (B, K) or (B, L)
        returns:  (B, D) or (B, K, D) or (B, L, D)
        """
        if self.use_sentence_embeddings:
            vecs = self.sent_projection(
                self.item_sentence_embeddings[item_ids.reshape(-1)]
            )
            return vecs.view(*item_ids.shape, -1)

        # Default: token-level encoding
        flat = item_ids.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)

        titles = self.item_title_tokens[uniq]
        mask = titles != self.padding_idx
        
        # Encode unique items only once
        uniq_vec = self.news_encoder.encode(titles, mask)  # (U, D)

        vec = uniq_vec[inv]  # (B*, D)
        return vec.view(*item_ids.shape, -1)

    def encode_user(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        item_seq: (B, T)
        """
        seq_mask = item_seq != self.padding_idx
        
        seq_vecs = self.encode_items(item_seq)  # (B, T, D)
        u = self.user_encoder.encode(seq_vecs, seq_mask)  # (B, D)
        return u

    # Helpers
    def score_user_candidates(self, user_vec: torch.Tensor, cand_vecs: torch.Tensor) -> torch.Tensor:
        """
        user_vec: (B, D)
        cand_vecs: (B, C, D)
        returns: (B, C) logits (dot product)
        """
        return torch.einsum("bd,bcd->bc", user_vec, cand_vecs)

    def forward(self, item_seq: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
        """
        Single item scoring (for predict).
        """
        u = self.encode_user(item_seq)  # (B, D)
        r = self.encode_items(item_id)  # (B, D)
        return torch.sum(u * r, dim=-1)

    def calculate_loss(self, interaction) -> torch.Tensor:
        """
        Paper training: pseudo (K+1)-way classification with shuffled candidates.
        """
        
        item_seq = interaction[self.ITEM_SEQ]  # (B, T)
        cand_item_ids = interaction[self.cand_field]  # (B, 1+K)
            
        pos_index = interaction[self.pos_index_field].long()  # (B,)
        u = self.encode_user(item_seq)
        cand_vecs = self.encode_items(cand_item_ids)
        logits = self.score_user_candidates(u, cand_vecs)
        
        return F.cross_entropy(logits, pos_index)

    def predict(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_id = interaction[self.ITEM_ID]
        u = self.encode_user(item_seq)  # (B, D)
        r = self.encode_items(item_id)  # (B, D)
        
        if self.similarity == "cosine":
            u = F.normalize(u, dim=-1)
            r = F.normalize(r, dim=-1)
        
        return torch.sum(u * r, dim=-1)

    def full_sort_predict(self, interaction) -> torch.Tensor:
        raise NotImplementedError("Not used, so we don't implement full-sort prediction.")
