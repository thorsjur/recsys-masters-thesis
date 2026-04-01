from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import torch

from recbole.config import Config


_SENTENCE_PROVIDER_REGISTRY: Dict[str, type["BaseSentenceEmbeddingProvider"]] = {}


def register_sentence_provider(name: str):
    """Register a sentence-level embedding provider by name."""
    def deco(cls):
        _SENTENCE_PROVIDER_REGISTRY[name] = cls
        return cls
    return deco


def build_sentence_embedding_provider(
    config: Config,
    dim: int,
    **kwargs,
) -> "BaseSentenceEmbeddingProvider":
    """
    Build a sentence embedding provider from config.

    Expected config keys
    --------------------
    sentence_embedding_source : str
        Name used in ``@register_sentence_provider``.
    """
    src = config["sentence_embedding_source"]
    assert src is not None, "sentence_embedding_source must be specified in config"

    cls = _SENTENCE_PROVIDER_REGISTRY.get(src)
    if cls is None:
        raise KeyError(
            f"Unknown sentence_embedding_source='{src}'. "
            f"Available: {list(_SENTENCE_PROVIDER_REGISTRY)}"
        )
    return cls(config=config, dim=dim, **kwargs)


@dataclass
class BaseSentenceEmbeddingProvider:
    """
    Base class for sentence-level embedding providers.

    Unlike token providers (which map token IDs -> vectors),
    sentence providers map arbitrary *strings* to dense vectors.
    """
    config: Config
    dim: int

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Encode a sequence of sentences into a (N, dim) tensor.

        Parameters
        ----------
        sentences : sequence of str
            Raw text strings to encode.
        batch_size : int
            Batch size for encoding (provider may ignore if not applicable).
        show_progress : bool
            Whether to show a progress bar during encoding.
        dtype : torch.dtype
            Desired dtype of the returned tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(len(sentences), self.dim)``.
        """
        raise NotImplementedError

    def encode_single(self, sentence: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convenience wrapper for a single sentence. Returns shape ``(dim,)``."""
        return self.encode([sentence], dtype=dtype).squeeze(0)
