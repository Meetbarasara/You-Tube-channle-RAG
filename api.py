"""
FastAPI Backend — YouTube RAG API

Endpoints:
  GET  /health      Health check
  GET  /status      Current video status + chunk count
  POST /ingest      Load a YouTube video
  POST /ask         Ask a question (RAG)
  POST /evaluate    Run RAGAS evaluation on a Q&A pair

Run:
  uvicorn api:app --reload --host 0.0.0.0 --port 8000
  Docs: http://localhost:8000/docs
"""

import logging
import time

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, field_validator

from main import YouTubeRAGPipeline
from src.components.config import settings
from src.components.evaluation import evaluate_response

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="YouTube RAG API",
    description=(
        "Paste any English YouTube video URL and ask questions about it. "
        "Powered by OpenAI embeddings + GPT-4o-mini generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow Streamlit (and any local dev origin) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared pipeline instance ──────────────────────────────────────────────────
pipeline = YouTubeRAGPipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def url_must_be_youtube(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL cannot be empty.")
        # Allow bare video IDs too
        import re
        is_youtube = "youtube.com" in v or "youtu.be" in v
        is_bare_id = bool(re.match(r"^[a-zA-Z0-9_-]{11}$", v))
        if not (is_youtube or is_bare_id):
            raise ValueError(
                "Must be a YouTube URL (youtube.com/watch?v=... or youtu.be/...) "
                "or an 11-character video ID."
            )
        return v


class IngestResponse(BaseModel):
    status:      str
    video_id:    str
    title:       str
    channel:     str
    thumbnail:   str
    url:         str
    chunk_count: int
    duration_s:  int
    elapsed_s:   float


class AskRequest(BaseModel):
    question: str
    top_k:    int = 4

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_range(cls, v: int) -> int:
        if not (1 <= v <= 10):
            raise ValueError("top_k must be between 1 and 10.")
        return v


class SourceChunk(BaseModel):
    content:          str
    similarity_score: float
    chunk_index:      int


class AskResponse(BaseModel):
    question:    str
    answer:      str
    sources:     list[SourceChunk]
    model:       str
    tokens_used: int
    elapsed_s:   float


class EvaluateRequest(BaseModel):
    question: str
    answer:   str
    contexts: list[str]


class EvaluateResponse(BaseModel):
    faithfulness:     float | None
    answer_relevancy: float | None
    error:            str | None


class StatusResponse(BaseModel):
    ready:       bool
    title:       str
    channel:     str
    url:         str
    thumbnail:   str
    chunk_count: int


class HealthResponse(BaseModel):
    status:  str
    version: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health():
    """Returns 200 OK when the API is running."""
    return HealthResponse(status="ok", version="1.0.0")


@app.get(
    "/status",
    response_model=StatusResponse,
    tags=["System"],
    summary="Current video status",
)
async def status_endpoint():
    """Returns the currently loaded video info and readiness state."""
    info = pipeline.video_info
    return StatusResponse(
        ready=pipeline.is_ready,
        title=info.get("title", ""),
        channel=info.get("channel", ""),
        url=info.get("url", ""),
        thumbnail=info.get("thumbnail", ""),
        chunk_count=pipeline.chunk_count,
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Pipeline"],
    summary="Load a YouTube video",
    status_code=status.HTTP_200_OK,
)
async def ingest(req: IngestRequest):
    """
    Extract the transcript, chunk it, and store embeddings.
    Must be called before /ask.
    """
    t0 = time.perf_counter()
    try:
        result = pipeline.ingest(req.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected ingestion error")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    info = result["video_info"]
    return IngestResponse(
        status="success",
        video_id=info.get("video_id", ""),
        title=info.get("title", ""),
        channel=info.get("channel", ""),
        thumbnail=info.get("thumbnail", ""),
        url=info.get("url", ""),
        chunk_count=result["chunk_count"],
        duration_s=info.get("duration", 0),
        elapsed_s=round(time.perf_counter() - t0, 2),
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["Pipeline"],
    summary="Ask a question about the loaded video",
    status_code=status.HTTP_200_OK,
)
async def ask(req: AskRequest):
    """
    Run the full RAG pipeline: embed query → retrieve → generate answer.
    /ingest must be called first.
    """
    if not pipeline.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No video loaded. Call POST /ingest first.",
        )

    t0 = time.perf_counter()
    try:
        result = pipeline.query(req.question, top_k=req.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected query error")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    sources = [
        SourceChunk(
            content=s["content"],
            similarity_score=s["similarity_score"],
            chunk_index=s["chunk_index"],
        )
        for s in result["sources"]
    ]

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources,
        model=result["model"],
        tokens_used=result["tokens_used"],
        elapsed_s=round(time.perf_counter() - t0, 2),
    )


@app.post(
    "/evaluate",
    response_model=EvaluateResponse,
    tags=["Evaluation"],
    summary="Run RAGAS evaluation on a Q&A pair",
    status_code=status.HTTP_200_OK,
)
async def evaluate(req: EvaluateRequest):
    """
    Compute RAGAS faithfulness and answer_relevancy scores.
    Pass the question, the generated answer, and the source chunk texts.
    """
    metrics = evaluate_response(
        question=req.question,
        answer=req.answer,
        contexts=req.contexts,
    )
    return EvaluateResponse(
        faithfulness=metrics.get("faithfulness"),
        answer_relevancy=metrics.get("answer_relevancy"),
        error=metrics.get("error"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
