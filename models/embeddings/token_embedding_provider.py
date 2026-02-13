from dataclasses import dataclass
from typing import Dict, Optional

import torch

from recbole.data.dataset import Dataset
from recbole.config import Config


_TOKEN_PROVIDER_REGISTRY: Dict[str, type["BaseTokenEmbeddingProvider"]] = {}


def register_token_provider(name: str):
    """Register a token-level embedding provider by name."""
    def deco(cls):
        _TOKEN_PROVIDER_REGISTRY[name] = cls
        return cls
    return deco


def build_token_embedding_provider(config: Config, dataset: Dataset, field: str, dim: int) -> "BaseTokenEmbeddingProvider":
    src = config["embedding_source"]
    assert src is not None, "embedding_source must be specified in config"
    
    cls = _TOKEN_PROVIDER_REGISTRY.get(src)
    if cls is None:
        raise KeyError(f"Unknown embedding_source='{src}'. Available: {list(_TOKEN_PROVIDER_REGISTRY)}")
    return cls(config=config, dataset=dataset, field=field, dim=dim)


@dataclass
class BaseTokenEmbeddingProvider:
    """
    Base class for token-level embedding providers.

    These providers produce an embedding matrix of shape (vocab_size, dim)
    that maps discrete token IDs to dense vectors (e.g., GloVe, random init).
    """
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


@register_token_provider("random")
class RandomTokenProvider(BaseTokenEmbeddingProvider):
    def get_embedding_matrix(self, vocab_size: int, padding_idx: int, dtype=torch.float32):
        return None


# ── Backward-compatible aliases ──────────────────────────────────────────────
BaseEmbeddingProvider = BaseTokenEmbeddingProvider
register_provider = register_token_provider
build_embedding_provider = build_token_embedding_provider
RandomProvider = RandomTokenProvider
