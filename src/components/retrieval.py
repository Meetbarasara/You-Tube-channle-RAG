"""
Retrieval module — in-memory vector store with cosine similarity search.

No external database required. For a single video (~20-60 chunks),
NumPy matrix operations are fast and efficient.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from langchain_core.documents import Document

from src.components.config import settings
from src.components.embeddings import embed_query

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its content, metadata, and similarity score."""
    content:          str
    metadata:         dict
    similarity_score: float
    chunk_index:      int

    def to_dict(self) -> dict:
        return {
            "content":          self.content,
            "metadata":         self.metadata,
            "similarity_score": round(self.similarity_score, 4),
            "chunk_index":      self.chunk_index,
        }


# ─────────────────────────────────────────────────────────────────────────────
# In-memory vector store
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    In-memory vector store for single-video RAG.

    Stores document chunks and their embeddings as a NumPy matrix.
    Similarity search is a single matrix-vector multiply — O(N) and fast.
    """

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None   # shape (N, D)
        self._documents:  list[Document]    = []
        self._video_info: dict              = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents:  list[Document],
        embeddings: np.ndarray,
        video_info: dict,
    ) -> None:
        """
        Load documents and their pre-computed embeddings into the store.

        Args:
            documents:  LangChain Document objects (content + metadata).
            embeddings: NumPy array of shape (len(documents), embedding_dim).
            video_info: Metadata dict for the current video.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"documents ({len(documents)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )

        self._documents  = documents
        self._embeddings = embeddings.astype(np.float32)
        self._video_info = video_info
        logger.info("VectorStore loaded: %d chunks", len(documents))

    def clear(self) -> None:
        """Reset the store (call before ingesting a new video)."""
        self._embeddings = None
        self._documents  = []
        self._video_info = {}
        logger.info("VectorStore cleared")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._embeddings is not None and len(self._documents) > 0

    @property
    def chunk_count(self) -> int:
        return len(self._documents)

    @property
    def video_info(self) -> dict:
        return self._video_info

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Find the most relevant chunks for a query using cosine similarity.

        Because embeddings are L2-normalised, cosine similarity = dot product.
        Matrix multiply gives us all scores in one vectorised operation.

        Args:
            query: The user's question.
            top_k: Number of results to return (defaults to settings.default_top_k).

        Returns:
            List of RetrievedChunk, sorted by descending similarity.
        """
        if not self.is_loaded:
            raise RuntimeError("VectorStore is empty. Ingest a video first.")

        top_k = top_k or settings.default_top_k
        threshold = settings.min_similarity_threshold

        # Embed query and compute similarities (dot product on normalised vecs = cosine)
        query_vec = embed_query(query)  # shape (D,)
        scores    = self._embeddings @ query_vec  # shape (N,)

        # Get top_k indices sorted by descending score
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < threshold:
                continue  # skip below-threshold results
            doc = self._documents[idx]
            results.append(
                RetrievedChunk(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    similarity_score=score,
                    chunk_index=int(idx),
                )
            )

        logger.info(
            "Retrieved %d chunks for query %r (top score: %.3f)",
            len(results), query[:60], scores[top_indices[0]] if len(top_indices) > 0 else 0,
        )
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Context formatting helper
# ─────────────────────────────────────────────────────────────────────────────

def format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM.
    Each chunk is labelled [Source N] so the LLM can cite it.
    """
    if not chunks:
        return "No relevant context found in the video."

    lines = ["Context from the video transcript:\n"]
    for i, chunk in enumerate(chunks, 1):
        score_pct = f"{chunk.similarity_score:.0%}"
        lines.append(f"[Source {i}] (relevance: {score_pct})")
        lines.append(chunk.content)
        lines.append("")  # blank line between chunks

    return "\n".join(lines)
