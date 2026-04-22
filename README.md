<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-Vector_Store-013243?style=for-the-badge&logo=numpy&logoColor=white" />
</p>

<h1 align="center">🎬 YouTube RAG — AI Video Intelligence</h1>

<p align="center">
  <b>An end-to-end Retrieval-Augmented Generation pipeline that transforms any YouTube video into an interactive, question-answerable knowledge base — powered by OpenAI embeddings, in-memory cosine similarity search, and GPT-4o-mini with source-cited answers.</b>
</p>

<p align="center">
  <i>Paste a YouTube URL → Transcript extracted & chunked → Embeddings generated → Ask anything → Get grounded, cited answers in seconds.</i>
</p>

---

## 🎯 Project Overview

YouTube RAG is a **production-grade** Retrieval-Augmented Generation system that allows users to semantically query any English YouTube video. Unlike traditional keyword search, this pipeline understands meaning — it embeds both the transcript and user queries into a shared vector space, retrieves the most relevant segments via cosine similarity, and generates answers that are **grounded in the actual video content** with traceable `[Source N]` citations.

### Key Highlights

- **Zero-database architecture** — All embeddings stored in-memory using NumPy; no Pinecone, Chroma, or FAISS required
- **Full-stack implementation** — FastAPI backend + Streamlit frontend with a premium dark glassmorphism UI
- **Source-cited answers** — Every response includes `[Source N]` references back to specific transcript segments
- **Sub-5s ingestion** — Transcript extraction, chunking, and embedding generation in under 5 seconds for typical videos
- **RAGAS evaluation** — Optional automated quality scoring (faithfulness & answer relevancy) for pipeline output validation
- **Production patterns** — Pydantic validation, structured logging, retry logic, CORS middleware, and clean error handling

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                    │
│                                                                         │
│   Streamlit UI (app.py)                                                 │
│   ├── Dark glassmorphism theme (Inter font, gradient backgrounds)       │
│   ├── Video card with thumbnail, metadata, and chunk count              │
│   ├── Chat interface with user/AI bubbles                               │
│   └── Expandable source chips with relevance scores                     │
│                         │  HTTP (requests)                              │
└─────────────────────────┼───────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         API LAYER                                       │
│                                                                         │
│   FastAPI Backend (api.py)                                              │
│   ├── POST /ingest     → Ingest a YouTube video                        │
│   ├── POST /ask        → RAG query with cited answer                   │
│   ├── POST /evaluate   → RAGAS quality evaluation                      │
│   ├── GET  /status     → Current video info & readiness                │
│   ├── GET  /health     → Health check                                  │
│   ├── Pydantic request/response models with field validators            │
│   └── CORS middleware for cross-origin access                           │
│                         │                                               │
└─────────────────────────┼───────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER                                  │
│                                                                         │
│   YouTubeRAGPipeline (main.py)                                          │
│   ├── ingest(url)  → Full ingestion lifecycle                           │
│   ├── query(question, top_k) → Full RAG query lifecycle                 │
│   └── Manages VectorStore state & pipeline coordination                 │
│                         │                                               │
└─────────────────────────┼───────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      COMPONENT LAYER (src/components/)                   │
│                                                                         │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│   │  ingestion   │  │  embeddings  │  │  retrieval   │  │ generation │ │
│   │              │  │              │  │              │  │            │ │
│   │ URL parsing  │  │ OpenAI API   │  │ VectorStore  │  │ GPT-4o-   │ │
│   │ yt-dlp meta  │  │ Batch embed  │  │ Cosine sim   │  │ mini +    │ │
│   │ Transcript   │  │ L2 normalize │  │ Top-K search │  │ citations │ │
│   │ Clean & chunk│  │ Retry logic  │  │ Threshold    │  │ Prompt    │ │
│   └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
│                                                                         │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│   │   config     │  │  evaluation  │  │   logger     │                 │
│   │              │  │              │  │              │                 │
│   │ Centralized  │  │ RAGAS        │  │ File + stdout│                 │
│   │ settings     │  │ faithfulness │  │ logging      │                 │
│   │ (dataclass)  │  │ relevancy    │  │              │                 │
│   └──────────────┘  └──────────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 How It Works — RAG Pipeline Deep Dive

### Ingestion Pipeline

```
YouTube URL
    │
    ▼
┌─── 1. URL Parsing ──────────────────────────────────────────────────┐
│   Regex-based extraction supporting 4 URL formats + bare video IDs  │
│   youtube.com/watch?v= | youtu.be/ | /embed/ | /v/ | raw 11-char   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─── 2. Metadata Extraction ──────────────────────────────────────────┐
│   yt-dlp fetches title, channel, duration, and thumbnail URL        │
│   Graceful fallback to defaults if metadata unavailable             │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─── 3. Transcript Retrieval ─────────────────────────────────────────┐
│   youtube-transcript-api (no OAuth required)                        │
│   Priority: en → en-US → en-GB → auto-translated English           │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─── 4. Transcript Cleaning ──────────────────────────────────────────┐
│   Remove: [Music], [Applause], timestamps [00:01:23]                │
│   Normalize: whitespace, punctuation spacing, repeated dashes       │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─── 5. Chunking ─────────────────────────────────────────────────────┐
│   RecursiveCharacterTextSplitter (LangChain)                        │
│   1000 chars per chunk, 200 char overlap                            │
│   Separators: \n\n → \n → ". " → " " → ""                          │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─── 6. Embedding ────────────────────────────────────────────────────┐
│   OpenAI text-embedding-3-small (1536 dimensions)                   │
│   Batch processing (up to 2048 per API call)                        │
│   L2-normalization → enables dot product = cosine similarity        │
│   Exponential backoff retry (3 attempts)                            │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
  [NumPy matrix (N × 1536) + LangChain Documents stored in memory]
```

### Query Pipeline

```
User Question
    │
    ▼
┌─── 1. Query Embedding ─────────────────────────────────────────────┐
│   Embed question via text-embedding-3-small → L2-normalize          │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─── 2. Cosine Similarity Search ────────────────────────────────────┐
│   scores = embedding_matrix @ query_vector  (single matrix multiply)│
│   Top-K selection with configurable similarity threshold (≥0.05)    │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─── 3. Context Formatting ──────────────────────────────────────────┐
│   Retrieved chunks labelled as [Source 1], [Source 2], ...          │
│   Each source includes relevance percentage                         │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─── 4. LLM Generation ──────────────────────────────────────────────┐
│   GPT-4o-mini with structured system prompt:                        │
│   • Answer ONLY from provided sources                               │
│   • Cite [Source N] references                                      │
│   • Admit when information isn't found                              │
│   Temperature: 0.2 | Max tokens: 800                                │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
  Grounded answer with [Source N] citations + token usage metadata
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| **Transcript Extraction** | `youtube-transcript-api` | Fetches English captions without OAuth |
| **Video Metadata** | `yt-dlp` | Extracts title, channel, duration, thumbnail |
| **Text Chunking** | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` with overlap |
| **Document Model** | `langchain-core` | `Document` objects with rich metadata |
| **Embeddings** | OpenAI `text-embedding-3-small` | 1536-dim vectors, L2-normalized |
| **Vector Store** | NumPy (in-memory) | Single matrix multiply for cosine search |
| **LLM** | OpenAI `gpt-4o-mini` | Grounded generation with source citations |
| **Backend** | FastAPI + Uvicorn | Async API with auto-generated OpenAPI docs |
| **Frontend** | Streamlit | Dark glassmorphism UI with chat interface |
| **Validation** | Pydantic v2 | Request/response schemas with field validators |
| **Configuration** | `python-dotenv` + dataclass | Centralized, type-safe settings |
| **Evaluation** | RAGAS (optional) | Faithfulness & answer relevancy metrics |

---

## 📂 Project Structure

```
youtube-rag/
├── api.py                      # FastAPI backend — 5 endpoints with Pydantic models
├── app.py                      # Streamlit frontend — premium dark theme UI
├── main.py                     # Pipeline orchestrator — ingest() and query() lifecycle
├── requirements.txt            # Pinned dependencies
├── .env                        # OpenAI API key (not committed)
├── .gitignore
│
└── src/
    ├── exception.py            # Custom exception class with traceback details
    ├── logger.py               # File + stdout logging configuration
    │
    └── components/
        ├── config.py           # Centralized settings (dataclass, singleton)
        ├── ingestion.py        # URL parsing, metadata, transcript, chunking
        ├── embeddings.py       # OpenAI embeddings with batch + retry logic
        ├── retrieval.py        # In-memory VectorStore with cosine similarity
        ├── generation.py       # GPT-4o-mini generation with citation prompt
        └── evaluation.py       # RAGAS evaluation (faithfulness, relevancy)
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/youtube-rag.git
cd youtube-rag
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure API Key

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

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **API Docs**: Auto-generated Swagger UI available at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 API Reference

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/health` | Health check — returns `{"status": "ok", "version": "1.0.0"}` |
| `GET` | `/status` | Current video info, chunk count, and readiness state |
| `POST` | `/ingest` | Load a YouTube video (extract transcript → chunk → embed) |
| `POST` | `/ask` | Ask a question about the loaded video (full RAG pipeline) |
| `POST` | `/evaluate` | Run RAGAS evaluation on a Q&A pair (optional) |

### `POST /ingest` — Load a Video

```json
// Request
{ "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ" }

// Response (200 OK)
{
  "status":      "success",
  "video_id":    "dQw4w9WgXcQ",
  "title":       "Rick Astley - Never Gonna Give You Up",
  "channel":     "Rick Astley",
  "thumbnail":   "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
  "url":         "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "chunk_count": 12,
  "duration_s":  213,
  "elapsed_s":   4.7
}
```

### `POST /ask` — Ask a Question

```json
// Request
{ "question": "What is the main message of this video?", "top_k": 4 }

// Response (200 OK)
{
  "question":    "What is the main message of this video?",
  "answer":      "According to [Source 1], the main message is...",
  "sources": [
    {
      "content": "Never gonna give you up, never gonna let you down...",
      "similarity_score": 0.82,
      "chunk_index": 3
    }
  ],
  "model":       "gpt-4o-mini",
  "tokens_used": 412,
  "elapsed_s":   2.1
}
```

### `POST /evaluate` — Quality Evaluation

```json
// Request
{
  "question": "What is this video about?",
  "answer":   "According to [Source 1], the video is about...",
  "contexts": ["Source text 1...", "Source text 2..."]
}

// Response (200 OK)
{
  "faithfulness":     0.95,
  "answer_relevancy": 0.88,
  "error":            null
}
```

---

## 🧠 Design Decisions & Trade-offs

| Decision | Rationale |
|:---------|:----------|
| **NumPy over vector databases** | For single-video RAG (~20–60 chunks), a NumPy matrix multiply is faster than any DB round-trip. Eliminates infrastructure dependencies. |
| **L2-normalization at embed time** | Pre-normalizing all vectors converts cosine similarity to a simple dot product (`scores = matrix @ query`), making search a single vectorized operation. |
| **`text-embedding-3-small` over `ada-002`** | 3x cheaper, better retrieval accuracy on benchmarks, and same 1536 dimensions. |
| **`gpt-4o-mini` over `gpt-4o`** | 15x cheaper with sufficient quality for grounded Q&A. Temperature set to 0.2 for factual consistency. |
| **RecursiveCharacterTextSplitter** | Prefers natural boundaries (paragraphs → sentences → words) over fixed-length cuts, preserving semantic coherence. |
| **200-char chunk overlap** | Prevents information loss at chunk boundaries — critical for transcript text where ideas span multiple sentences. |
| **Decoupled API + UI** | FastAPI backend is independently deployable and testable; Streamlit frontend is swappable with any HTTP client. |

---

## 💰 Cost Estimate

| Operation | Model | Approximate Cost |
|:----------|:------|:----------------|
| Ingest 1 video (~40 chunks) | `text-embedding-3-small` | ~$0.0001 |
| Ask 1 question | `gpt-4o-mini` | ~$0.001 |
| Full demo session (ingest + 10 questions) | Both | ~$0.01 |

---

## 🔮 Future Enhancements

- [ ] **Multi-video support** — Query across multiple ingested videos simultaneously
- [ ] **Persistent storage** — PostgreSQL + pgvector for production-scale deployments
- [ ] **Streaming responses** — SSE-based streaming for real-time answer generation
- [ ] **Timestamp linking** — Click a source to jump to the exact video timestamp
- [ ] **Multi-language support** — Extend beyond English transcripts
- [ ] **Hybrid search** — Combine semantic search with BM25 keyword matching

---

## 📄 License

MIT — free to use, modify, and distribute.

---

<p align="center">
  <sub>Built with ❤️ using OpenAI, FastAPI, Streamlit, and NumPy</sub>
</p>
