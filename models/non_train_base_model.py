

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class NewsEmbeddingRecommender(GeneralRecommender, ABC):

    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.device = config["device"]

        self.n_items = dataset.item_num
        self.n_users = dataset.user_num

        self.title_field = config["title_field"] if "title_field" in config else "title"
        self.abstract_field = config["abstract_field"] if "abstract_field" in config else "abstract"
        self.use_abstract = config["use_abstract"] if "use_abstract" in config else True

        self.hist_field = config["hist_field"]
        self.use_hist = config.get("use_hist", False)
        self.has_hist_field = self.hist_field is not None and self.hist_field in dataset.inter_feat and self.use_hist

        # Fields produced by impression dataloader for mlp sim training
        self.cand_field = config.get("cand_field", "cand_item_id")
        self.pos_index_field = config.get("pos_index_field", "pos_index")

        self.aggregation = config["aggregation"] if "aggregation" in config else "mean"
        self.similarity = config["similarity"] if "similarity" in config else "cosine"
        if self.similarity not in ("cosine", "dot", "mlp"):
            raise ValueError(f"Unknown similarity='{self.similarity}'. Must be one of ('cosine', 'dot', 'mlp').")

        # Dummy param so RecBole is happy
        self.dummy_param = nn.Parameter(torch.zeros(1))

        self._build_item_embeddings(dataset)  # (n_items, dim)
        self._build_user_embeddings(dataset)  # (n_users, dim)

        assert hasattr(self, "item_embeddings"), "_build_item_embeddings() must set self.item_embeddings"
        assert hasattr(self, "user_embeddings"), "_build_user_embeddings() must set self.user_embeddings"

        if self.item_embeddings.device != torch.device(self.device):
            self.item_embeddings = self.item_embeddings.to(self.device)
        if self.user_embeddings.device != torch.device(self.device):
            self.user_embeddings = self.user_embeddings.to(self.device)

        if self.item_embeddings.dtype != torch.float32:
            self.item_embeddings = self.item_embeddings.float()
        if self.user_embeddings.dtype != torch.float32:
            self.user_embeddings = self.user_embeddings.float()

        # This MLP is based on, and uses heuristics from "Natural Language Inference by Tree-Based Convolution
        # and Heuristic Matching" (https://aclanthology.org/P16-2022.pdf)
        embed_dim = self.item_embeddings.shape[1]
        self.scoring_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        ).to(self.device)

        self.logger.info(
            f"{self.__class__.__name__} initialized: "
            f"item_embeddings={self.item_embeddings.shape}, "
            f"user_embeddings={self.user_embeddings.shape}, "
            f"aggregation={self.aggregation}, similarity={self.similarity}"
        )

    def _build_user_hist_items(self, dataset) -> dict:
        inter = dataset.inter_feat
        mask = inter["label"] > 0
        inter = inter[mask]

        user_ids = inter[self.USER_ID].cpu().numpy()
        item_ids = inter[self.ITEM_ID].cpu().numpy()

        return {u: [i for i in item_ids[user_ids == u]] for u in user_ids}

    def _get_user_embedding(self, interaction):
        if self.has_hist_field:
            # If history field is present in interaction features, we use it while evaluating
            # as it holds all history up to the current interaction anyways. And it matches
            # the setting of other models that use history from before the training data.
            hist = interaction[self.hist_field]
            rep = self.item_embeddings[hist]
            u = self._aggregate_history(rep)
        else:
            u = self.user_embeddings[interaction[self.USER_ID]]
        return u

    def _aggregate_history(self, hist_emb: torch.Tensor) -> torch.Tensor:

        # Determine aggregation dimension based on tensor shape
        # (n_items, dim) -> aggregate along dim=0
        # (n_users, n_items, dim) -> aggregate along dim=1 (items dimension)
        agg_dim = -2 if hist_emb.dim() == 3 else 0

        if self.aggregation == "mean":
            return hist_emb.mean(dim=agg_dim)
        if self.aggregation == "sum":
            return hist_emb.sum(dim=agg_dim)
        if self.aggregation == "max":
            return hist_emb.max(dim=agg_dim)[0]
        raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def _build_interaction_features(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.cat([u, v, u * v, torch.abs(u - v)], dim=-1)

    def _score_mlp(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x = self._build_interaction_features(u, v)
        return self.scoring_mlp(x).squeeze(-1)

    def forward(self, interaction):
        if self.similarity != "mlp":
            raise NotImplementedError("Forward method is only implemented for MLP similarity.")

        u = self._get_user_embedding(interaction)
        v = self.item_embeddings[interaction[self.ITEM_ID]]

        return self._score_mlp(u, v)

    def calculate_loss(self, interaction) -> torch.Tensor:
        if self.similarity != "mlp":
            return self.dummy_param.new_zeros(())

        cand_item_ids = interaction[self.cand_field]
        pos_index = interaction[self.pos_index_field].long()

        u = self._get_user_embedding(interaction)
        cand_vecs = self.item_embeddings[cand_item_ids]

        u = u.detach()
        cand_vecs = cand_vecs.detach()

        u = u.unsqueeze(1).expand(-1, cand_vecs.size(1), -1)

        logits = self._score_mlp(u, cand_vecs)
        return F.cross_entropy(logits, pos_index)

    @torch.no_grad()
    def predict(self, interaction):
        item = interaction[self.ITEM_ID]  # (B,)

        u = self._get_user_embedding(interaction)
        v = self.item_embeddings[item]

        if self.similarity == "mlp":
            return self._score_mlp(u, v)

        if self.similarity == "cosine":
            return F.cosine_similarity(u, v, dim=1)

        return torch.sum(u * v, dim=1)

    @torch.no_grad()
    def full_sort_predict(self, interaction):
        raise NotImplementedError("Full sort prediction is not supported for NewsEmbeddingRecommender.")
        # user = interaction[self.USER_ID]

        # u = self.user_embeddings[user]
        # return u @ self.item_embeddings.t()

    @abstractmethod
    def _build_item_embeddings(self, dataset):
        raise NotImplementedError

    def _get_text_fields(self) -> List[str]:
        """Return the list of item text fields to use (title, and optionally abstract)."""
        fields = [self.title_field]
        if self.use_abstract and self.abstract_field:
            fields.append(self.abstract_field)
        return fields

    def _build_embedding_matrices(
        self,
        dataset,
        fields: List[str],
        dim: int,
    ) -> Dict[str, torch.Tensor]:
        """Build a `(vocab_size, dim)` embedding matrix per text field via the
        configured token_embedding_provider.
        """
        from models.embeddings.token_embedding_provider import build_token_embedding_provider

        item_feat = dataset.get_item_feature()
        matrices: Dict[str, torch.Tensor] = {}

        for field in fields:
            if field not in item_feat:
                self.logger.warning(f"Field '{field}' not in item features, skipping.")
                continue

            provider = build_token_embedding_provider(
                config=self.config,
                dataset=dataset,
                field=field,
                dim=dim,
            )
            vocab_size = dataset.num(field)
            matrix = provider.get_embedding_matrix(
                vocab_size=vocab_size,
                padding_idx=0,
                dtype=torch.float32,
            )
            if matrix is None:
                # Fallback for providers that return None (e.g. random)
                matrix = torch.zeros(vocab_size, dim)
            matrices[field] = matrix

        return matrices

    def _average_token_embeddings(
        self,
        dataset,
        matrices: Dict[str, torch.Tensor],
        dim: int,
    ) -> torch.Tensor:
        """Compute per-item mean embeddings by averaging token vectors across fields.

        For each item, all non-padding token vectors from every field in
        matrices are pooled together.  The result is a single
        `(n_items, dim)` tensor.
        """
        item_feat = dataset.get_item_feature()
        total_sum = torch.zeros(self.n_items, dim)
        total_count = torch.zeros(self.n_items, 1)

        for field, matrix in matrices.items():
            token_ids = item_feat[field]
            if not torch.is_tensor(token_ids):
                token_ids = torch.tensor(token_ids, dtype=torch.long)
            token_ids = token_ids.long()

            embs = matrix[token_ids]  # (n_items, seq_len, dim)
            mask = (token_ids != 0).unsqueeze(-1).float()  # (n_items, seq_len, 1)
            total_sum += (embs * mask).sum(dim=1)  # (n_items, dim)
            total_count += mask.sum(dim=1)  # (n_items, 1)

        return total_sum / total_count.clamp(min=1)

    def _build_token_provider_item_embeddings(
        self,
        dataset,
        dim: int,
        fields: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if fields is None:
            fields = self._get_text_fields()

        matrices = self._build_embedding_matrices(dataset, fields, dim)

        if not matrices:
            self.logger.warning("No text fields found; item embeddings will be zeros.")
            return torch.zeros(self.n_items, dim, device=self.device)

        item_emb = self._average_token_embeddings(dataset, matrices, dim)
        self.logger.info(f"Built item embeddings: {item_emb.shape}")
        return item_emb.to(self.device)

    def _build_user_embeddings(self, dataset):
        dim = self.item_embeddings.shape[1]

        hist_field = self.config["hist_field"]
        if hist_field is not None and hist_field in dataset.inter_feat:
            self.logger.info(f"Using user history from interaction field '{hist_field}' lazily")
            self.user_embeddings = torch.zeros((self.n_users, dim), dtype=torch.float32, device=self.device)
            return

        self.user_hist_items = self._build_user_hist_items(dataset)

        user_emb = torch.zeros((self.n_users, dim), dtype=torch.float32, device=self.device)

        for u in range(self.n_users):
            hist = self.user_hist_items.get(u, [])
            if not hist:
                continue

            idx = torch.as_tensor(hist, dtype=torch.long, device=self.device)
            hist_emb = self.item_embeddings[idx]
            user_emb[u] = self._aggregate_history(hist_emb)

        self.user_embeddings = user_emb
        self.logger.info(f"TFIDF built user embeddings: shape={self.user_embeddings.shape}")

    def _get_item_text_tokens(self, dataset, item_idx: int) -> List[str]:
        assert dataset.item_feat is not None, "Dataset must have item features."

        item_feat = dataset.item_feat
        tokens: List[str] = []

        tokens.extend(self._get_tokens_from_field(dataset, item_feat, self.title_field, item_idx))

        if self.use_abstract:
            tokens.extend(self._get_tokens_from_field(dataset, item_feat, self.abstract_field, item_idx))

        return tokens

    def _get_item_text(self, dataset, item_idx: int) -> str:
        return " ".join(self._get_item_text_tokens(dataset, item_idx))

    def _get_tokens_from_field(self, dataset, item_feat, field: str, item_idx: int) -> List[str]:
        if field not in item_feat:
            return []

        ids = item_feat[field][item_idx]
        if torch.is_tensor(ids):
            ids = ids.cpu().numpy()

        # RecBole token sequences are padded with 0
        return [dataset.id2token(field, [int(t)])[0] for t in ids if int(t) != 0]

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Encode a batch of items into their embedding space."""
        return self.item_embeddings[item_ids]
