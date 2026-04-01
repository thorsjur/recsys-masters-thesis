from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class BaseUserEncoder(nn.Module, ABC):
    @abstractmethod
    def encode(self, seq_vectors: torch.Tensor, seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns (B, D)"""
        raise NotImplementedError
