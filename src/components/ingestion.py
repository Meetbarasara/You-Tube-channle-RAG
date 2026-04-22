"""
Ingestion module — single English-language YouTube video.

Pipeline:
  URL → video_id → transcript → clean → chunk → LangChain Documents
"""

import os
import re
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.components.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# URL / ID Parsing
# ─────────────────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    """
    Extract an 11-character YouTube video ID from any common URL format.

    Supports:
      - youtube.com/watch?v=ID
      - youtu.be/ID
      - youtube.com/embed/ID
      - youtube.com/v/ID
      - raw 11-character ID
    """
    patterns = [
        r'(?:youtube\.com/watch\?.*v=)([a-zA-Z0-9_-]{11})',
        r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com/v/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # Bare video ID
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url.strip()):
        return url.strip()

    raise ValueError(f"Could not extract a video ID from: {url!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────

def get_video_info(video_id: str) -> dict:
    """
    Fetch video metadata via yt-dlp (title, channel, duration, thumbnail).
    Returns an empty dict on failure — callers should handle gracefully.
    """
    ydl_opts = {"quiet": True, "skip_download": True, "no_warnings": True}
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )
        return {
            "video_id": video_id,
            "title":     info.get("title", "Unknown Title"),
            "channel":   info.get("uploader", "Unknown Channel"),
            "duration":  info.get("duration", 0),
            "thumbnail": info.get("thumbnail", ""),
            "url":       f"https://www.youtube.com/watch?v={video_id}",
        }
    except Exception as exc:
        logger.warning("Could not retrieve video metadata for %s: %s", video_id, exc)
        return {
            "video_id":  video_id,
            "title":     "Unknown Title",
            "channel":   "Unknown Channel",
            "duration":  0,
            "thumbnail": "",
            "url":       f"https://www.youtube.com/watch?v={video_id}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Transcript
# ─────────────────────────────────────────────────────────────────────────────

def get_video_transcript(video_id: str) -> str:
    """
    Fetch the English transcript for a YouTube video.

    Tries English first, then falls back to any available language
    with English translation. Raises ValueError if no transcript found.
    """
    api = YouTubeTranscriptApi()
    try:
        # Primary: manual or auto English
        transcript = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
    except Exception:
        try:
            # Fallback: any language with auto-translation to English
            transcript_list = api.list(video_id)
            transcript = transcript_list.find_transcript(
                ["en", "en-US", "en-GB"]
            ).fetch()
        except Exception as exc:
            raise ValueError(
                f"No English transcript available for video {video_id}. "
                "Make sure the video has English captions enabled."
            ) from exc

    raw_text = " ".join(snippet.text for snippet in transcript.snippets)
    logger.info("Transcript fetched: %d characters", len(raw_text))
    return raw_text


def clean_transcript(transcript: str) -> str:
    """
    Light cleaning that preserves meaning while removing noise.

    Removes:
      - Timestamp markers like [00:01:23]
      - Square-bracket annotations like [Music], [Applause]
      - Repeated whitespace / newlines
      - Repeated dashes (----)
    """
    text = transcript

    # Remove timestamp markers
    text = re.sub(r"\[\d{2}:\d{2}:\d{2}\]", "", text)

    # Remove annotation markers like [Music]
    text = re.sub(r"\[[^\]]{1,40}\]", "", text)

    # Normalise whitespace around punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:])\s+", r"\1 ", text)

    # Collapse multiple spaces / newlines
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    # Collapse repeated dashes
    text = re.sub(r"-{2,}", "-", text)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_transcript(
    transcript: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split a transcript into overlapping text chunks."""
    chunk_size    = chunk_size    or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(transcript)


def create_documents(
    video_id: str,
    transcript: str,
    video_info: dict,
) -> list[Document]:
    """
    Build a list of LangChain Documents from transcript chunks.
    Each document carries rich metadata for downstream retrieval.
    """
    chunks = chunk_transcript(transcript)
    documents = []

    for i, chunk in enumerate(chunks):
        metadata = {
            "chunk_id":     f"{video_id}_chunk_{i:04d}",
            "chunk_index":  i,
            "total_chunks": len(chunks),
            "video_id":     video_id,
            "title":        video_info.get("title", ""),
            "channel":      video_info.get("channel", ""),
            "url":          video_info.get("url", ""),
            "thumbnail":    video_info.get("thumbnail", ""),
        }
        documents.append(Document(page_content=chunk, metadata=metadata))

    logger.info("Created %d document chunks from transcript", len(documents))
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Public Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def ingest_single_video(video_url_or_id: str) -> tuple[list[Document], dict]:
    """
    Full ingestion pipeline for a single YouTube video.

    Steps:
      1. Parse URL → video ID
      2. Fetch metadata (title, channel, thumbnail)
      3. Fetch & clean English transcript
      4. Chunk transcript
      5. Wrap chunks in LangChain Documents with metadata

    Returns:
        (documents, video_info)

    Raises:
        ValueError: if no transcript is available
    """
    logger.info("Starting ingestion for: %s", video_url_or_id)

    video_id   = extract_video_id(video_url_or_id)
    video_info = get_video_info(video_id)
    logger.info("Video: %r by %r", video_info["title"], video_info["channel"])

    raw_transcript     = get_video_transcript(video_id)
    cleaned_transcript = clean_transcript(raw_transcript)
    logger.info(
        "Transcript cleaned: %d → %d chars",
        len(raw_transcript), len(cleaned_transcript),
    )

    documents = create_documents(video_id, cleaned_transcript, video_info)
    return documents, video_info