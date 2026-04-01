from typing import Optional
import torch
import torch.nn as nn

from models.core import AdditiveAttentionPooling, MultiHeadSelfAttention
from models.encoders.base_user_encoder import BaseUserEncoder


class NRMSUserEncoder(BaseUserEncoder):
    """
    NRMS User Encoder
    """

    def __init__(
        self,
        news_dim: int,
        num_heads: int,
        head_dim: int,
        att_dim: int,
        dropout: float = 0.0,  # Paper only mentions dropout on a word encoding level
    ):
        super().__init__()

        self.selfatt = MultiHeadSelfAttention(
            num_heads=num_heads,
            input_dim=news_dim,
            head_dim=head_dim,
        )
        self.selfatt_drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.out_dim = num_heads * head_dim
        self.pool = AdditiveAttentionPooling(in_dim=self.out_dim, att_dim=att_dim)

    def encode(self, seq_vectors: torch.Tensor, seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        seq_vectors: (B, T, news_dim)
        seq_mask: (B, T) bool (True for valid history positions)
        returns: (B, out_dim)
        """
        mask = None if seq_mask is None else seq_mask.bool()
        
        # (B, T, out_dim)
        h = self.selfatt(seq_vectors, mask=mask)
        h = self.selfatt_drop(h)
        
        # (B, out_dim)
        u = self.pool(h, mask=mask)
        return u
