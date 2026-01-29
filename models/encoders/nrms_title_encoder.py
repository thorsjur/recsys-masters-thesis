from typing import Optional

import torch
import torch.nn as nn

from models.core import AdditiveAttentionPooling, MultiHeadSelfAttention
from models.encoders.base_text_encoder import BaseTextEncoder


class NRMSTitleEncoder(BaseTextEncoder):
    """
    NRMS Title Encoder
    """

    def __init__(
        self,
        vocab_size: int,
        word_embed_dim: int,
        num_heads: int,
        head_dim: int,
        att_dim: int,
        dropout: float,
        padding_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()

        self.padding_idx = padding_idx

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, word_embed_dim, padding_idx=padding_idx)
        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != (vocab_size, word_embed_dim):
                raise ValueError(f"Bad pretrained shape {pretrained_embeddings.shape}")
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = not freeze_embeddings

        # Dropout on embeddings (paper uses 0.2)
        self.dropout = nn.Dropout(dropout)

        # Word-level self-attention
        self.selfatt = MultiHeadSelfAttention(
            num_heads=num_heads,
            input_dim=word_embed_dim,
            head_dim=head_dim,
        )
        self.selfatt_drop = nn.Dropout(dropout)

        self.out_dim = num_heads * head_dim

        # Additive attention pooling
        self.pool = AdditiveAttentionPooling(in_dim=self.out_dim, att_dim=att_dim)

    def _build_token_mask(self, token_ids: torch.Tensor, token_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Ensures bool mask with True = valid token positions.
        If token_mask is None, infer from padding_idx.
        """
        if token_mask is None:
            return token_ids != self.padding_idx
        return token_mask.bool()

    def encode(self, token_ids: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        token_ids: (B, L)
        token_mask: (B, L) bool (True for real tokens)
        returns: (B, out_dim)
        """
        mask = self._build_token_mask(token_ids, token_mask)
        x = self.embedding(token_ids)
        if torch.isnan(x).any(): raise RuntimeError("NaN after embedding")

        x = self.dropout(x)
        if torch.isnan(x).any(): raise RuntimeError("NaN after dropout")

        h = self.selfatt(x, mask=mask)
        if torch.isnan(h).any(): raise RuntimeError("NaN after selfatt")

        h = self.selfatt_drop(h)
        if torch.isnan(h).any(): raise RuntimeError("NaN after selfatt_drop")

        rep = self.pool(h, mask=mask)
        if torch.isnan(rep).any(): raise RuntimeError("NaN after pooling")
        return rep
