from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttentionPooling(nn.Module):

    def __init__(self, in_dim: int, att_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, att_dim)
        self.fc = nn.Linear(att_dim, 1, bias=False)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, E: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        E: (B, L, D)
        mask: (B, L) bool

        returns: (B, D)
        """
        h = torch.tanh(self.proj(E))  # (B, L, att_dim)
        scores = self.fc(h).squeeze(-1)  # (B, L)

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9 if scores.dtype == torch.float32 else -1e4)

        alpha = F.softmax(scores, dim=-1)  # (B, L)
        rep = torch.einsum("bl,bld->bd", alpha, E)
        return rep
