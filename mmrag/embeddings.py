"""all-MiniLM-L6-v2 via fastembed (ONNX, no torch). 384-dim, L2-normalized, so the
vectors match the ones the notebook created with sentence-transformers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import load_settings


@lru_cache(maxsize=1)
def get_model():
    from fastembed import TextEmbedding

    settings = load_settings()
    # threads=1 keeps memory low on tiny instances (Render free = 0.1 CPU / 512 MB).
    return TextEmbedding(model_name=settings.embedding_model, threads=1)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = np.array(list(get_model().embed(texts)), dtype=np.float32)
    return _normalize(vectors).tolist()


def embed_query(text: str) -> list[float]:
    return embed_documents([text])[0]


if __name__ == "__main__":  # used by the build step to pre-download the model
    print("Embedding dimension:", len(embed_query("dimension check")))
