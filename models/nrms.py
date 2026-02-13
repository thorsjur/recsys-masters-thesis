import torch
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType

from models.embeddings.token_embedding_provider import build_token_embedding_provider
from models.encoders import NRMSTitleEncoder, NRMSUserEncoder, BaseTextEncoder, BaseUserEncoder


class NRMS(SequentialRecommender):

    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.title_field: str = config["title_field"]
        self.title_len: int = int(config["title_len"])
        self.vocab_size: int = dataset.num(self.title_field)
        self.padding_idx: int = int(config.get("padding_idx", 0))

        # Hyperparams
        self.word_embedding_dim = int(config.get("word_embedding_dim", 300))
        self.nb_head = int(config.get("num_heads", 16))
        self.size_per_head = int(config.get("head_dim", 16))
        self.att_dim = int(config.get("att_dim", 200))
        self.dropout = float(config.get("emb_dropout", 0.2))
        self.neg_k = int(config.get("neg_sample_num", 4))

        # Fields produced by impression dataloader
        self.cand_field = config.get("cand_field", "cand_item_id")
        self.pos_index_field = config.get("pos_index_field", "pos_index")

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

        # Encoders
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

        self.user_encoder: BaseUserEncoder = NRMSUserEncoder(
            news_dim=self.news_encoder.out_dim,
            num_heads=self.nb_head,
            head_dim=self.size_per_head,
            att_dim=self.att_dim,
        )

        self.hidden_size = self.user_encoder.out_dim

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

        if title_tokens.dim() != 2 or title_tokens.size(1) != self.title_len:
            raise ValueError(
                f"Expected '{self.title_field}' of shape (n_items, title_len={self.title_len}), got {tuple(title_tokens.shape)}"
            )
        self.register_buffer("item_title_tokens", title_tokens)
        
        pad_title = self.item_title_tokens[self.padding_idx]
        if (pad_title != self.padding_idx).any():
            print("Warning: item_title_tokens[padding_idx] is not all padding tokens.")
            

    # Helpers
    def _title_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids != self.padding_idx

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        item_ids: (B,) or (B, K) or (B, L)
        returns:  (B, D) or (B, K, D) or (B, L, D)
        """
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
        return self.forward(item_seq, item_id)

    def full_sort_predict(self, interaction) -> torch.Tensor:
        raise NotImplementedError("Not used, so we don't implement full-sort prediction.")
