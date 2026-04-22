"""
Embeddings module — OpenAI text-embedding-3-small.

Provides:
  - embed_documents(texts) → np.ndarray   shape (N, 1536)
  - embed_query(query)     → np.ndarray   shape (1536,)
"""

import logging
import time
import numpy as np
from openai import OpenAI

from src.components.config import settings

logger = logging.getLogger(__name__)

# Lazy-initialised OpenAI client
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Return (and lazily create) the OpenAI client."""
    global _client
    if _client is None:
        settings.validate()
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Core embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_embedding_api(texts: list[str], retries: int = 3) -> list[list[float]]:
    """
    Call OpenAI Embeddings API with retry logic.

    Batches up to 2048 texts per request (API limit).
    Normalises each vector to unit length for cosine similarity.
    """
    client = _get_client()
    all_embeddings: list[list[float]] = []

    # Process in batches of 2048 (OpenAI limit)
    batch_size = 2048
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        for attempt in range(retries):
            try:
                response = client.embeddings.create(
                    model=settings.embedding_model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as exc:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Embedding API error (attempt %d/%d): %s. Retrying in %ds…",
                        attempt + 1, retries, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"OpenAI Embedding API failed after {retries} attempts: {exc}"
                    ) from exc

    return all_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def embed_documents(texts: list[str]) -> np.ndarray:
    """
    Embed a list of text strings.

    Returns:
        np.ndarray of shape (len(texts), embedding_dimension)
        Each row is an L2-normalised embedding vector.
    """
    if not texts:
        return np.empty((0, settings.embedding_dimension), dtype=np.float32)

    logger.info("Embedding %d document chunks via OpenAI…", len(texts))
    raw = _call_embedding_api(texts)
    matrix = np.array(raw, dtype=np.float32)

    # L2-normalise rows for fast cosine similarity (dot product = cosine)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    matrix /= norms

    logger.info("Embeddings ready: shape %s", matrix.shape)
    return matrix


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        np.ndarray of shape (embedding_dimension,) — L2-normalised.
    """
    logger.info("Embedding query: %r", query[:80])
    raw = _call_embedding_api([query])
    vec = np.array(raw[0], dtype=np.float32)

    # Normalise
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec
