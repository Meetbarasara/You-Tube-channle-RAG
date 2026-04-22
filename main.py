"""
YouTubeRAGPipeline — orchestrator for the full single-video RAG pipeline.

Usage:
    pipeline = YouTubeRAGPipeline()
    pipeline.ingest("https://youtube.com/watch?v=...")
    result   = pipeline.query("What is the main topic?")
    print(result["answer"])
"""

import logging

from src.components.ingestion  import ingest_single_video
from src.components.embeddings import embed_documents
from src.components.retrieval  import VectorStore, format_context
from src.components.generation import generate_answer
from src.components.config     import settings

logger = logging.getLogger(__name__)


class YouTubeRAGPipeline:
    """
    End-to-end RAG pipeline for a single English-language YouTube video.

    Lifecycle:
      1. Call ingest(url) once to load a video.
      2. Call query(question) any number of times.
      3. Call ingest(url) again to switch to a different video.
    """

    def __init__(self) -> None:
        self._store = VectorStore()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, video_url: str) -> dict:
        """
        Full ingestion pipeline: URL → transcript → chunks → embeddings → store.

        Args:
            video_url: Any YouTube URL format or bare video ID.

        Returns:
            {
                "status":       "success",
                "video_info":   dict,   # title, channel, url, thumbnail, duration
                "chunk_count":  int,
            }

        Raises:
            ValueError: if no English transcript is available.
            RuntimeError: if OpenAI API call fails.
        """
        logger.info("=== INGESTION START ===")

        # Clear previous video
        self._store.clear()

        # Step 1 — Extract transcript + build Documents
        documents, video_info = ingest_single_video(video_url)

        # Step 2 — Embed all chunks
        texts      = [doc.page_content for doc in documents]
        embeddings = embed_documents(texts)

        # Step 3 — Load into vector store
        self._store.add_documents(documents, embeddings, video_info)

        logger.info("=== INGESTION COMPLETE: %d chunks ===", len(documents))
        return {
            "status":      "success",
            "video_info":  video_info,
            "chunk_count": len(documents),
        }

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int | None = None) -> dict:
        """
        Full RAG query: question → embed → retrieve → generate.

        Args:
            question: Natural language question about the video.
            top_k:    How many chunks to retrieve (defaults to settings.default_top_k).

        Returns:
            {
                "question":       str,
                "answer":         str,
                "sources":        list[dict],   # retrieved chunks with scores
                "model":          str,
                "tokens_used":    int,
                "video_info":     dict,
            }

        Raises:
            RuntimeError: if no video has been ingested yet.
        """
        if not self.is_ready:
            raise RuntimeError("No video loaded. Call ingest(url) first.")

        top_k = top_k or settings.default_top_k
        logger.info("=== QUERY: %r ===", question[:80])

        # Step 1 — Retrieve relevant chunks
        chunks = self._store.search(question, top_k=top_k)

        # Step 2 — Format context
        context = format_context(chunks)

        # Step 3 — Generate answer
        gen_result = generate_answer(
            query=question,
            context=context,
            video_info=self._store.video_info,
        )

        return {
            "question":    question,
            "answer":      gen_result["answer"],
            "sources":     [c.to_dict() for c in chunks],
            "model":       gen_result["model"],
            "tokens_used": gen_result["tokens_used"],
            "video_info":  self._store.video_info,
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if a video has been ingested and the store is non-empty."""
        return self._store.is_loaded

    @property
    def video_info(self) -> dict:
        return self._store.video_info

    @property
    def chunk_count(self) -> int:
        return self._store.chunk_count


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test when run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    TEST_VIDEO = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    TEST_Q     = "What is this video about?"

    pipeline = YouTubeRAGPipeline()

    print("\n" + "=" * 60)
    print("[PIPELINE TEST] Ingesting video…")
    print("=" * 60)
    result = pipeline.ingest(TEST_VIDEO)
    print(f"  Title:   {result['video_info']['title']}")
    print(f"  Chunks:  {result['chunk_count']}")

    print("\n" + "=" * 60)
    print(f"[PIPELINE TEST] Query: {TEST_Q!r}")
    print("=" * 60)
    answer_result = pipeline.query(TEST_Q)
    print(f"\nAnswer:\n{answer_result['answer']}")
    print(f"\nSources retrieved: {len(answer_result['sources'])}")
    print(f"Tokens used: {answer_result['tokens_used']}")
