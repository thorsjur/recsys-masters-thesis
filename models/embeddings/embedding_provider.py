from dataclasses import dataclass
from typing import Dict, Optional

import torch

from recbole.data.dataset import Dataset
from recbole.config import Config


_PROVIDER_REGISTRY: Dict[str, type["BaseEmbeddingProvider"]] = {}


def register_provider(name: str):
    def deco(cls):
        _PROVIDER_REGISTRY[name] = cls
        return cls
    return deco


def build_embedding_provider(config: Config, dataset: Dataset, field: str, dim: int) -> "BaseEmbeddingProvider":
    src = config["embedding_source"]
    assert src is not None, "embedding_source must be specified in config"
    
    cls = _PROVIDER_REGISTRY.get(src)
    if cls is None:
        raise KeyError(f"Unknown embedding_source='{src}'. Available: {list(_PROVIDER_REGISTRY)}")
    return cls(config=config, dataset=dataset, field=field, dim=dim)


@dataclass
class BaseEmbeddingProvider:
    config: Config
    dataset: Dataset
    field: str
    dim: int

    def get_embedding_matrix(
        self,
        vocab_size: int,
        padding_idx: int,
        dtype: torch.dtype = torch.float32
    ) -> Optional[torch.Tensor]:
        """
        Return (vocab_size, dim) or None for random init.
        """
        raise NotImplementedError

@register_provider("random")
class RandomProvider(BaseEmbeddingProvider):
    def get_embedding_matrix(self, vocab_size: int, padding_idx: int, dtype=torch.float32):
        return None
