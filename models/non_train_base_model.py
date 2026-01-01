from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class NewsEmbeddingRecommender(GeneralRecommender, ABC):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.device = config["device"]

        self.n_items = dataset.item_num
        self.n_users = dataset.user_num

        self.title_field = config["title_field"] if "title_field" in config else "title"
        self.abstract_field = config["abstract_field"] if "abstract_field" in config else "abstract"
        self.use_abstract = config["use_abstract"] if "use_abstract" in config else True

        self.aggregation = config["aggregation"] if "aggregation" in config else "mean"
        self.similarity = config["similarity"] if "similarity" in config else "cosine"

        # Dummy param so RecBole is happy
        self.dummy_param = nn.Parameter(torch.zeros(1))

        self._build_item_embeddings(dataset)
        self._build_user_embeddings(dataset)

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

        if self.similarity == "cosine":
            self.item_embeddings = F.normalize(self.item_embeddings, p=2, dim=1)
            self.user_embeddings = F.normalize(self.user_embeddings, p=2, dim=1)

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

    def _aggregate_history(self, hist_emb: torch.Tensor) -> torch.Tensor:
        
        # Attention-based aggregation is handled separately during prediction,
        # and thus the user-embeddings are not really used in that case.
        # For simplicity, we just use mean aggregation here.
        if self.aggregation == "mean" or self.aggregation == "attention":
            return hist_emb.mean(dim=0)
        if self.aggregation == "sum":
            return hist_emb.sum(dim=0)
        if self.aggregation == "max":
            return hist_emb.max(dim=0)[0]
        raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def forward(self, interaction):
        pass

    def calculate_loss(self, interaction):
        return self.dummy_param.new_zeros(())

    @torch.no_grad()
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        if self.aggregation == "attention":
            inters = zip(user.tolist(), item.tolist())
            scores = [self._attention_score_single(u_id, i_id) for u_id, i_id in inters]

            return torch.stack(scores)
        
        u = self.user_embeddings[user]
        v = self.item_embeddings[item]

        return torch.sum(u * v, dim=1)
        

    @torch.no_grad()
    def full_sort_predict(self, interaction):
        if self.aggregation == "attention":
            raise NotImplementedError("Full sort prediction is not supported with attention-based aggregation.")

        user = interaction[self.USER_ID]

        u = self.user_embeddings[user]
        return u @ self.item_embeddings.t()

    @abstractmethod
    def _build_item_embeddings(self, dataset):
        raise NotImplementedError

    def _build_user_embeddings(self, dataset):
        self.user_hist_items = self._build_user_hist_items(dataset)
        
        dim = int(self.item_embeddings.shape[1])
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

    def _attention_score_single(self, user_id: int, item_id: int) -> torch.Tensor:
        hist = self.user_hist_items.get(user_id, [])
        if not hist:
            return torch.tensor(0.0, device=self.device)

        hist_idx = torch.as_tensor(hist, dtype=torch.long, device=self.device)

        hist_emb = self.item_embeddings[hist_idx]
        item_emb = self.item_embeddings[item_id]

        similarity = (hist_emb * item_emb.unsqueeze(0)).sum(dim=-1)
        weights = torch.softmax(similarity, dim=0)
        u_attn = (weights.unsqueeze(-1) * hist_emb).sum(dim=0)

        score = (u_attn * item_emb).sum()
        return score
