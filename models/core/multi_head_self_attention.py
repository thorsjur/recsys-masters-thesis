import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_fill(scores: torch.Tensor, mask: Optional[torch.Tensor], fill: float) -> torch.Tensor:
    """mask: True for valid, False for invalid"""
    if mask is None:
        return scores
    return scores.masked_fill(~mask, fill)


class MultiHeadSelfAttention(nn.Module):
    """
    While not the exact same implementation as described in the NRMS paper, it follows the
    NRMS source code more closely than the BilinearMultiHeadSelfAttention.

    I am pretty sure this is what the authors used in practice, as the current implementation
    with the QKV-style matrices give a parameter count much closer to what is reported in the paper.
    531K params against the reported 530K, while with the bilinear version it is 2.7M params.
    """

    def __init__(self, num_heads: int, input_dim: int, head_dim: int):
        super().__init__()
        if num_heads <= 0 or input_dim <= 0 or head_dim <= 0:
            raise ValueError("num_heads, input_dim, head_dim must be positive")

        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.output_dim = num_heads * head_dim

        self.WQ = nn.Linear(input_dim, self.output_dim, bias=True)
        self.WK = nn.Linear(input_dim, self.output_dim, bias=True)
        self.WV = nn.Linear(input_dim, self.output_dim, bias=True)

        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.zeros_(self.WQ.bias)
        nn.init.zeros_(self.WK.bias)
        nn.init.zeros_(self.WV.bias)

    def forward(self, E: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        E: (B, L, D)  where D=input_dim, L=sequence length
        mask: (B, L) bool, True for valid positions, False else

        returns: (B, L, H*head_dim)
        """
        B, L, D = E.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {D}")

        # (B, L, H*Hd)
        Q = self.WQ(E)
        K = self.WK(E)
        V = self.WV(E)

        # (B, H, L, Hd)
        Q = Q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        K = K.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        V = V.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        # (B, H, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Mask keys
        if mask is not None:
            key_mask = mask[:, None, None, :].expand_as(scores)  # (B, H, L, L)
            fill = -1e9 if scores.dtype == torch.float32 else -1e4
            scores = _masked_fill(scores, key_mask, fill)

        alpha = F.softmax(scores, dim=-1)  # (B, H, L, L)

        # (B, H, L, Hd)
        out = torch.matmul(alpha, V)

        # (B, L, H*Hd)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, self.output_dim)
        return out
