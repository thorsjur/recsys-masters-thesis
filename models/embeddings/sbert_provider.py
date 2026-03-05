import hashlib
import os
import shelve
import threading
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from models.embeddings.sentence_embedding_provider import (
    BaseSentenceEmbeddingProvider,
    register_sentence_provider,
)

logger = getLogger(__name__)


def _load_model(
    model_name: str,
    device: Optional[str] = None,
    fp16: bool = False,
    model_kwargs: Optional[Dict] = None,
    tokenizer_kwargs: Optional[Dict] = None,
):
    """Return a ``SentenceTransformer`` instance."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformer model '%s' …", model_name)
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs=model_kwargs or {},
        tokenizer_kwargs=tokenizer_kwargs or {},
    )
    if fp16 and model.device.type == "cuda":
        model.half()
    return model


def _cache_key(sentence: str, model_name: str, normalize: bool) -> str:
    """Deterministic hash key for one (sentence, model, normalize) combo."""
    raw = f"{model_name}|{normalize}|{sentence}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class _EmbeddingDiskCache:
    """
    A thin wrapper around ``shelve`` that stores sentence embeddings as numpy
    arrays on disk.  Thread-safe for reads; a lock guards writes.
    """

    def __init__(self, cache_dir: str | os.PathLike, model_name: str, normalize: bool):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.replace("/", "__")
        self._db_path = str(self._dir / f"{safe_name}_norm={normalize}")
        self._lock = threading.Lock()
        self._model_name = model_name
        self._normalize = normalize

    def lookup(self, sentences: Sequence[str]) -> Tuple[List[int], List[str], Dict[int, np.ndarray]]:
        """
        Partition *sentences* into hits and misses.

        Returns
        -------
        miss_indices : list[int]
            Positions in *sentences* that are NOT in the cache.
        miss_sentences : list[str]
            The corresponding text strings.
        hits : dict[int, np.ndarray]
            ``{position: cached_vector}`` for every cache hit.
        """
        miss_idx: List[int] = []
        miss_sent: List[str] = []
        hits: Dict[int, np.ndarray] = {}

        try:
            with shelve.open(self._db_path, flag="r") as db:
                for i, s in enumerate(sentences):
                    key = _cache_key(s, self._model_name, self._normalize)
                    val = db.get(key)
                    if val is not None:
                        hits[i] = val
                    else:
                        miss_idx.append(i)
                        miss_sent.append(s)
        except Exception:
            # DB doesn't exist yet
            miss_idx = list(range(len(sentences)))
            miss_sent = list(sentences)

        return miss_idx, miss_sent, hits

    def store(self, sentences: Sequence[str], vectors: np.ndarray) -> None:
        """Persist embeddings for *sentences* (same order as *vectors* rows)."""
        with self._lock:
            with shelve.open(self._db_path, flag="c") as db:
                for s, vec in zip(sentences, vectors):
                    key = _cache_key(s, self._model_name, self._normalize)
                    db[key] = vec



@register_sentence_provider("sentence_transformer")
@register_sentence_provider("sbert")
@dataclass
class SentenceTransformerProvider(BaseSentenceEmbeddingProvider):
    """Provider backed by the ``sentence-transformers`` library.
    Works with any model available via ``SentenceTransformer(model_name)``.
    """

    _model: Optional["SentenceTransformer"] = field(default=None, init=False, repr=False)  # type: ignore[type-arg]
    _cache: Optional[_EmbeddingDiskCache] = field(default=None, init=False, repr=False)
    _is_setup: bool = field(default=False, init=False, repr=False)

    @property
    def model_name(self) -> str:
        return str(self.config.get("sentence_embedding_model", "all-MiniLM-L6-v2"))

    @property
    def _device(self) -> Optional[str]:
        return self.config.get("sentence_embedding_device", None)

    @property
    def _fp16(self) -> bool:
        return bool(self.config.get("sentence_embedding_fp16", False))

    @property
    def _normalize(self) -> bool:
        return bool(self.config.get("sentence_embedding_normalize", True))

    @property
    def _default_batch_size(self) -> int:
        return int(self.config.get("sentence_embedding_batch_size", 128))

    @property
    def _task(self) -> Optional[str]:
        return self.config.get("sentence_embedding_task", None)

    def _apply_instruction(self, sentences: Sequence[str]) -> List[str]:
        if not self._task:
            return list(sentences)
        return [f"Instruct: {self._task}\nQuery: {s}" for s in sentences]

    def _setup(self) -> None:
        if self._is_setup:
            return
        self._model = _load_model(
            self.model_name,
            self._device,
            self._fp16,
            model_kwargs=dict(self.config.get("sentence_embedding_model_kwargs", {}) or {}),
            tokenizer_kwargs=dict(self.config.get("sentence_embedding_tokenizer_kwargs", {}) or {}),
        )

        cache_dir = self.config.get("sentence_embedding_cache_dir", "./cache/sentence_embeddings")
        if cache_dir:
            self._cache = _EmbeddingDiskCache(cache_dir, self.model_name, self._normalize)
            logger.info("Sentence embedding disk cache at %s", cache_dir)

        self._is_setup = True

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 0,
        show_progress: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Encode sentences to (N, dim) tensor.

        Cached embeddings are returned directly; only cache-misses are
        forwarded to the model.
        """
        self._setup()
        sentences = self._apply_instruction(sentences)
        n = len(sentences)
        if n == 0:
            return torch.empty(0, self.dim, dtype=dtype)

        bs = batch_size if batch_size > 0 else self._default_batch_size

        # cache look-up
        if self._cache is not None:
            miss_idx, miss_sent, hits = self._cache.lookup(sentences)
        else:
            miss_idx = list(range(n))
            miss_sent = list(sentences)
            hits = {}

        logger.info(
            "Sentence encode: %d total, %d cached, %d to compute",
            n,
            len(hits),
            len(miss_sent),
        )

        # model inference for misses
        if miss_sent:
            vecs_np = self._encode_batch(miss_sent, bs, show_progress)

            # persist new embeddings
            if self._cache is not None:
                self._cache.store(miss_sent, vecs_np)

            # merge into hits dict
            for local_i, global_i in enumerate(miss_idx):
                hits[global_i] = vecs_np[local_i]

        # assemble output tensor in original order
        out = torch.empty(n, self.dim, dtype=dtype)
        for i in range(n):
            out[i] = torch.from_numpy(hits[i]).to(dtype)

        return out

    def _encode_batch(
        self,
        sentences: List[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        assert self._model is not None, "_encode_batch called before model setup"
        
        """Run the sentence-transformers model and return numpy array."""
        with torch.no_grad():
            vecs = self._model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize,
            )
        return np.asarray(vecs, dtype=np.float32)
