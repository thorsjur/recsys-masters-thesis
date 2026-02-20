import os
from logging import getLogger
from typing import Any, Optional

import numpy as np
import torch

import logging
logging.getLogger("smart_open").setLevel(logging.WARNING)

from models.embeddings.token_embedding_provider import (
    BaseTokenEmbeddingProvider,
    register_token_provider,
)

# Module-level cache so that multiple provider instances (e.g. one per
# text field) share a single copy of the heavyweight keyed-vectors.
_KV_CACHE: dict[str, Any] = {}


def _load_keyed_vectors(bin_path: str, cache_path: Optional[str]) -> Any:
    """Load (and optionally cache) FastText keyed vectors."""

    
    key = cache_path or bin_path
    if key in _KV_CACHE:
        return _KV_CACHE[key]

    from gensim.models.fasttext import FastTextKeyedVectors, load_facebook_vectors

    logger = getLogger()

    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached FastText vectors from {cache_path}")
        kv = FastTextKeyedVectors.load(cache_path, mmap="r")
    else:
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"FastText binary not found: {bin_path}")
        logger.info(f"Loading FastText model from {bin_path}")
        kv = load_facebook_vectors(bin_path)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            kv.save(cache_path)
            logger.info(f"FastText vectors cached at {cache_path}")

    _KV_CACHE[key] = kv
    return kv


@register_token_provider("fasttext")
class FastTextProvider(BaseTokenEmbeddingProvider):
    """Build a token embedding matrix from a pre-trained FastText model.

    FastText can generate vectors for out-of-vocabulary words via subword
    (character n-gram) information, so coverage is typically much higher
    than with GloVe-style static lookup tables.
    """

    def get_embedding_matrix(
        self,
        vocab_size: int,
        padding_idx: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        logger = getLogger()

        bin_path = os.path.expanduser(
            self.config.get("embedding_path", "~/fasttext/cc.en.300.bin")
        )
        cache_path_cfg = self.config.get("fasttext_cache", f"~/fasttext/cache/{bin_path.split('/')[-1]}.kv")
        cache_path = os.path.expanduser(cache_path_cfg) if cache_path_cfg else None

        lower = bool(self.config.get("embedding_lower", False))
        init_std = float(self.config.get("embedding_init_std", 0.02))

        kv = _load_keyed_vectors(bin_path, cache_path)

        id2token = self.dataset.field2id_token[self.field][:vocab_size]

        logger.info(
            f"Building FastText embedding matrix (field={self.field}, "
            f"dim={self.dim}, vocab_size={vocab_size}, lower={lower})"
        )

        # Initialise: random for truly-missing tokens, zeros for padding
        emb = torch.empty((vocab_size, self.dim), dtype=dtype)
        torch.nn.init.normal_(emb, mean=0.0, std=init_std)

        if 0 <= padding_idx < vocab_size:
            emb[padding_idx].zero_()

        found = 0
        oov_samples: list[str] = []
        sample_n = int(self.config.get("embedding_oov_log_n", 20))

        for idx, token in enumerate(id2token):
            if idx == padding_idx:
                continue

            word = token.lower() if lower else token
            try:
                vec = kv.get_vector(word)
            except KeyError:
                if len(oov_samples) < sample_n:
                    oov_samples.append(word)
                continue

            if len(vec) != self.dim:
                continue

            emb[idx] = torch.from_numpy(vec.astype(np.float32)).to(dtype=dtype)
            found += 1

        target = vocab_size - (1 if 0 <= padding_idx < vocab_size else 0)
        coverage = (found / max(1, target)) * 100.0
        logger.info(
            f"FastText coverage: {found}/{target} tokens ({coverage:.2f}%). "
            f"OOV: {target - found}."
        )

        if oov_samples:
            logger.info(f"Example OOV tokens (first {len(oov_samples)}): {oov_samples}")

        return emb
