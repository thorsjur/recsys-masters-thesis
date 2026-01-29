from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseTextEncoder(nn.Module, ABC):
    @abstractmethod
    def encode(self, token_ids: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns (B, D)"""
        raise NotImplementedError