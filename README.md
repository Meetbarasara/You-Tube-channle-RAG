# 🎬 YouTube RAG — AI Video Intelligence

> **Ask anything about any YouTube video.** Paste a URL, get instant AI-powered answers grounded in the video transcript — with source citations.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)

---

## Architecture

```
User (Streamlit UI)
       │  HTTP
       ▼
┌──────────────────────────────────────────┐
│          FastAPI Backend (api.py)        │
│                                          │
│  POST /ingest  →  Ingestion pipeline     │
│  POST /ask     →  RAG query pipeline     │
│  GET  /status  →  Loaded video info      │
│  GET  /health  →  Health check           │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│         RAG Pipeline (main.py)           │
│                                          │
│  ingestion.py  → Transcript extraction   │
│  embeddings.py → text-embedding-3-small  │
│  retrieval.py  → NumPy cosine search     │
│  generation.py → GPT-4o-mini + citations │
│  config.py     → Central settings        │
└──────────────────────────────────────────┘
```

> **No database required.** All embeddings live in-memory (NumPy). Perfect for single-video RAG.

---

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Transcript | `youtube-transcript-api` | No OAuth needed |
| Metadata | `yt-dlp` | Title, channel, thumbnail |
| Chunking | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| Embeddings | OpenAI `text-embedding-3-small` | 1536-dim, L2-normalised |
| Vector Search | NumPy cosine similarity | Single-matrix multiply, no DB |
| LLM | OpenAI `gpt-4o-mini` | Grounded answers with `[Source N]` citations |
| Backend | FastAPI + uvicorn | Auto-docs at `/docs` |
| Frontend | Streamlit | Dark glassmorphism theme |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...your-key-here...
```

### 3. Run

Open **two terminals**:

```bash
# Terminal 1 — Start the FastAPI backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Start the Streamlit frontend
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## API Reference

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### `GET /health`
Returns `{"status": "ok"}` when the API is running.

### `GET /status`
Returns the currently loaded video info, chunk count, and readiness state.

### `POST /ingest`
Load a YouTube video for questioning.

```json
// Request
{ "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ" }

// Response
{
  "status":      "success",
  "title":       "Never Gonna Give You Up",
  "channel":     "Rick Astley",
  "chunk_count": 12,
  "duration_s":  213,
  "elapsed_s":   4.7
}
```

### `POST /ask`
Ask a question about the loaded video.

```json
// Request
{ "question": "What is the main message of this video?", "top_k": 4 }

// Response
{
  "answer":      "According to [Source 1], the main message is...",
  "sources":     [{"content": "...", "similarity_score": 0.82, "chunk_index": 3}],
  "model":       "gpt-4o-mini",
  "tokens_used": 412,
  "elapsed_s":   2.1
}
```

---

## How It Works

```
1. URL         →  Extract 11-char video ID
2. video ID    →  Fetch metadata (yt-dlp) + English transcript (youtube-transcript-api)
3. Transcript  →  Clean (remove noise) → Chunk (1000 chars, 200 overlap)
4. Chunks      →  Embed via text-embedding-3-small → L2-normalise → store in NumPy
5. Question    →  Embed query → cosine similarity search → top-K chunks
6. Top chunks  →  Format as [Source N] context → GPT-4o-mini → grounded answer
```

---

## Project Structure

```
1_YouTubbe RAG/
├── .env                    # OpenAI API key (not committed)
├── .gitignore
├── README.md
├── requirements.txt
├── api.py                  # FastAPI backend
├── app.py                  # Streamlit frontend
├── main.py                 # Pipeline orchestrator
└── src/
    ├── exception.py        # Custom exceptions
    ├── logger.py           # Logging setup
    └── components/
        ├── config.py       # Central configuration
        ├── ingestion.py    # YouTube ingestion
        ├── embeddings.py   # OpenAI embeddings
        ├── retrieval.py    # In-memory vector store
        ├── generation.py   # GPT-4o-mini generation
        └── evaluation.py   # RAGAS evaluation (optional)
```

---

## Cost Estimate

| Operation | Model | Cost |
|-----------|-------|------|
| Ingest 1 video (~40 chunks) | text-embedding-3-small | ~$0.0001 |
| Ask 1 question | gpt-4o-mini | ~$0.001 |

Extremely cheap for a demo or portfolio project.

---

## License

MIT — free to use, modify, and distribute.
