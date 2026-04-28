from logging import getLogger
import os
import torch

from models.embeddings.token_embedding_provider import BaseTokenEmbeddingProvider, register_token_provider
from pathlib import Path
import numpy as np


@register_token_provider("glove")
class GloveProvider(BaseTokenEmbeddingProvider):
    """
    Loads a GloVe text file and builds an embedding matrix
    """

    @staticmethod
    def _resolve_path(p: str | os.PathLike) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (Path.cwd() / path)

    def get_embedding_matrix(
        self,
        vocab_size: int,
        padding_idx: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        logger = getLogger()

        path_cfg = self.config.get("embedding_path", None)
        assert path_cfg is not None, "embedding_path must be specified in config"

        path = self._resolve_path(path_cfg)
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")

        lower = bool(self.config.get("embedding_lower", False))
        init_std = float(self.config.get("embedding_init_std", 0.02))
        encoding = str(self.config.get("embedding_encoding", "utf-8"))

        id2token = self.dataset.field2id_token[self.field][:vocab_size]

        if lower:
            token2id = {tok.lower(): i for i, tok in enumerate(id2token)}
        else:
            token2id = {tok: i for i, tok in enumerate(id2token)}

        remaining = set(token2id.keys())
        if 0 <= padding_idx < vocab_size:
            pad_tok = id2token[padding_idx]
            remaining.discard(pad_tok)

        logger.info(
            f"Loading embeddings from {path} (dim={self.dim}, vocab_size={vocab_size}, "
            f"lower={lower}, dtype={dtype}, init_std={init_std})"
        )

        # we initialize the embedding matrix with random normal values, and set padding to 0
        emb = torch.empty((vocab_size, self.dim), dtype=dtype)
        torch.nn.init.normal_(emb, mean=0.0, std=init_std)

        if 0 <= padding_idx < vocab_size:
            emb[padding_idx].zero_()

        found = 0
        # Early stop if we find all tokens (except padding)
        target_found = vocab_size - (1 if 0 <= padding_idx < vocab_size else 0)

        # Parse line by line to save some memory
        with open(path, "r", encoding=encoding, errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                # Skip word2vec header if present: "<count> <dim>"
                if lineno == 1:
                    parts0 = line.split()
                    if len(parts0) == 2 and parts0[0].isdigit() and parts0[1].isdigit():
                        continue

                try:
                    word, rest = line.split(" ", 1)
                except ValueError:
                    continue

                if lower:
                    word = word.lower()

                idx = token2id.get(word)
                if idx is None:
                    continue

                vec_np = np.fromstring(rest, sep=" ", dtype=np.float32)
                if vec_np.shape[0] != self.dim:
                    logger.warning(
                        f"Line {lineno}: expected {self.dim} dimensions but got {vec_np.shape[0]} for word '{word}'. Skipping."
                    )
                    continue

                if idx == padding_idx:
                    remaining.discard(word)
                    continue

                emb[idx] = torch.from_numpy(vec_np).to(dtype=dtype)
                found += 1
                remaining.discard(word)

                if found >= target_found:
                    break

        coverage = (found / target_found) * 100.0
        logger.info(
            f"Embedding coverage: found {found}/{target_found} tokens "
            f"({coverage:.2f}%). OOV vocab tokens: {len(remaining)}."
        )

        if len(remaining) > 0:
            sample_n = int(self.config.get("embedding_oov_log_n", 20))
            sample = sorted(remaining)[: max(0, sample_n)]
            logger.info(f"Example OOV tokens (first {len(sample)}): {sample}")

        return emb
