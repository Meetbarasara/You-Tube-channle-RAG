"""
Generation module — GPT-4o-mini answer generation with source citations.

Produces grounded answers that reference only the retrieved context.
The LLM is instructed to cite [Source N] tags, which the UI renders.
"""

import logging
from openai import OpenAI

from src.components.config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        settings.validate()
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an intelligent assistant that answers questions about YouTube video content.
You are given relevant excerpts from a video transcript, labelled [Source 1], [Source 2], etc.

Rules:
1. Answer ONLY using information present in the provided sources.
2. Cite the source(s) you used, e.g. "According to [Source 1], ..."
3. If the answer is not found in the sources, say:
   "I couldn't find information about that in this video."
4. Be concise, clear, and helpful. Avoid padding.
5. Do NOT invent information beyond what the sources state.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(
    query:   str,
    context: str,
    video_info: dict | None = None,
) -> dict:
    """
    Generate a grounded answer using GPT-4o-mini.

    Args:
        query:      The user's question.
        context:    Formatted context string from retrieval.format_context().
        video_info: Optional video metadata (used for richer prompt framing).

    Returns:
        {
            "answer":       str,   # LLM-generated grounded answer
            "model":        str,   # Model used
            "tokens_used":  int,   # Total tokens consumed
        }
    """
    client = _get_client()

    # Optionally frame the prompt with video context
    video_frame = ""
    if video_info:
        title   = video_info.get("title",   "Unknown")
        channel = video_info.get("channel", "Unknown")
        video_frame = f'The video is titled "{title}" by "{channel}".\n\n'

    user_message = (
        f"{video_frame}"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    logger.info("Calling %s for generation…", settings.llm_model)

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        answer      = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens if response.usage else 0

        logger.info("Answer generated. Tokens used: %d", tokens_used)
        return {
            "answer":      answer,
            "model":       settings.llm_model,
            "tokens_used": tokens_used,
        }

    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        raise RuntimeError(f"Answer generation failed: {exc}") from exc
