"""Local sentence-embedding wrapper.

A hosted embedding API would add per-call latency, cost, and an external
dependency to something that runs entirely on synthetic, non-sensitive
documents -- there is no reason to leave the machine. `all-MiniLM-L6-v2` is
small enough (~90MB) to download once and run on CPU in milliseconds per
chunk, which keeps the whole retrieval pipeline offline and free after setup.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode a batch of strings into L2-normalised embedding vectors."""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Encode a single string; a thin convenience wrapper over embed_texts."""
    return embed_texts([text])[0]
