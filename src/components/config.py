"""
Central configuration for the YouTube RAG pipeline.
All constants and settings live here. Import from this module everywhere.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ─── OpenAI ────────────────────────────────────────────────
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    embedding_dimension: int = 1536

    # ─── Chunking ───────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ─── Retrieval ──────────────────────────────────────────────
    default_top_k: int = 4
    min_similarity_threshold: float = 0.05

    # ─── Generation ─────────────────────────────────────────────
    max_context_chunks: int = 5
    llm_temperature: float = 0.2
    llm_max_tokens: int = 800

    # ─── API ────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    def validate(self) -> None:
        """Raise if critical settings are missing."""
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file: OPENAI_API_KEY=sk-..."
            )


# Singleton instance — import this everywhere
settings = Settings()
