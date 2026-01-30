import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _masked_fill(scores: torch.Tensor, mask: Optional[torch.Tensor], fill: float) -> torch.Tensor:
    """mask: True for valid, False for invalid"""
    if mask is None:
        return scores
    return scores.masked_fill(~mask, fill)


class BilinearMultiHeadSelfAttention(nn.Module):
    """
    Multi Head Self Attention as described by the NRMS paper.
    Initialized following the paper's code (notebook): https://github.com/wuch15/EMNLP2019-NRMS/blob/master/Baseline-NRMS.ipynb
    
    Note that this slightly deviates from the standard multi-head attention mechanism with Q, K and V matrices
    (used in the NRMS code).

    Output is concatenation over heads: (B, L, H * head_dim)
    """

    def __init__(self, num_heads: int, input_dim: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.output_dim = num_heads * head_dim

        self.Q = nn.Parameter(torch.empty(num_heads, input_dim, input_dim)) # (H, D, D)
        self.V = nn.Parameter(torch.empty(num_heads, head_dim, input_dim))

        # Initialized following the NRMS notebook impl.
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.V)

    def forward(self, E: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        E: (B, L, D)  where D=input_dim, L=sequence length
        mask: (B, L) bool, True for valid positions, False else

        returns: (B, L, H*head_dim)
        """
        
        B, L, D = E.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {D}")

        # Compute scores for each head and batch
        scores = torch.einsum("bid,hde,bje->bhij", E, self.Q, E) # (B, H, L, L)

        # Mask keys
        if mask is not None:
            key_mask = mask[:, None, None, :].expand_as(scores)  # (B, H, L, L)
            scores = _masked_fill(scores, key_mask, -1e9 if scores.dtype == torch.float32 else -1e4)
            
        # Softmax to get attention weights alpha
        alpha = F.softmax(scores, dim=-1)  # (B, H, L, L)

        # Weighted sum of original E
        context = torch.einsum("bhij,bjd->bhid", alpha, E)  # (B, H, L, D)

        # Apply V^k
        out = torch.einsum("hkd,bhid->bhik", self.V, context)  # (B, H, L, head_dim)

        # Concatenate heads -> (B, L, H*head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, self.output_dim)
        return out
