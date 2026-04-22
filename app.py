"""
YouTube RAG — Premium Streamlit Frontend
Communicates with the FastAPI backend (api.py) via HTTP.

Run:
    # Terminal 1 — Start the API
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

    # Terminal 2 — Start the UI
    streamlit run app.py
"""

import time
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────────
# Page setup — MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="YouTube RAG · AI Video Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling — Dark glassmorphism theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

  .stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #111827 50%, #0d1117 100%);
    min-height: 100vh;
  }

  .block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

  /* ── Hero ── */
  .hero-title {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin-bottom: 0.3rem;
  }
  .hero-subtitle { color: #94a3b8; font-size: 1.05rem; margin-bottom: 2rem; }

  /* ── Step label ── */
  .step-label {
    color: #a78bfa; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.5rem;
  }

  /* ── Video card ── */
  .video-card {
    background: linear-gradient(135deg, rgba(167,139,250,0.08), rgba(96,165,250,0.08));
    border: 1px solid rgba(167,139,250,0.2); border-radius: 16px;
    padding: 1.2rem; display: flex; align-items: flex-start;
    gap: 1rem; margin-bottom: 1.5rem;
  }
  .video-card img { border-radius: 10px; width: 140px; flex-shrink: 0; }
  .video-meta h3 { color: #e2e8f0; font-size: 1rem; font-weight: 600; margin: 0 0 0.3rem; }
  .video-meta p  { color: #94a3b8; font-size: 0.85rem; margin: 0.15rem 0; }
  .badge {
    display: inline-block; background: rgba(167,139,250,0.18);
    color: #a78bfa; border: 1px solid rgba(167,139,250,0.3);
    border-radius: 20px; padding: 2px 10px; font-size: 0.75rem;
    font-weight: 500; margin-top: 0.5rem;
  }

  /* ── Chat bubbles ── */
  .chat-user { display: flex; justify-content: flex-end; margin-bottom: 0.8rem; }
  .chat-user-bubble {
    background: linear-gradient(135deg, #7c3aed, #4f46e5); color: #fff;
    border-radius: 18px 18px 4px 18px; padding: 0.75rem 1.2rem;
    max-width: 70%; font-size: 0.95rem; line-height: 1.5;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3);
  }
  .chat-ai { display: flex; justify-content: flex-start; margin-bottom: 1.2rem; gap: 0.7rem; }
  .chat-avatar {
    width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0; margin-top: 3px;
  }
  .chat-ai-bubble {
    background: rgba(255,255,255,0.055); border: 1px solid rgba(255,255,255,0.1);
    color: #e2e8f0; border-radius: 4px 18px 18px 18px;
    padding: 0.9rem 1.2rem; max-width: 82%; font-size: 0.95rem; line-height: 1.65;
  }
  .chat-meta { color: #64748b; font-size: 0.73rem; margin-top: 0.35rem; }

  /* ── Source chips ── */
  .source-chip {
    background: rgba(52,211,153,0.06); border: 1px solid rgba(52,211,153,0.18);
    border-radius: 8px; padding: 0.65rem 0.9rem;
    margin-bottom: 0.55rem; font-size: 0.82rem;
    color: #94a3b8; line-height: 1.55;
  }
  .score-high { color: #34d399; font-weight: 600; }
  .score-med  { color: #fbbf24; font-weight: 600; }
  .score-low  { color: #f87171; font-weight: 600; }

  /* ── Input — aggressive selectors for Streamlit 1.56+ ── */
  input[type="text"], input[type="search"], textarea,
  .stTextInput input, .stTextInput textarea,
  [data-testid="stTextInput"] input,
  [data-baseweb="input"] input,
  [data-baseweb="base-input"] input {
    background: rgba(20, 20, 40, 0.85) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #e2e8f0 !important;
    caret-color: #a78bfa !important;
    border-radius: 10px !important;
    padding: 0.65rem 1rem !important;
    -webkit-text-fill-color: #e2e8f0 !important;
  }
  input[type="text"]:focus, input[type="search"]:focus,
  [data-baseweb="input"] input:focus,
  [data-baseweb="base-input"] input:focus {
    border-color: rgba(167,139,250,0.6) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
    outline: none !important;
  }
  input::placeholder,
  [data-baseweb="input"] input::placeholder {
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
  }
  /* Base input wrapper background */
  [data-baseweb="input"],
  [data-baseweb="base-input"] {
    background: rgba(20, 20, 40, 0.85) !important;
    border-radius: 10px !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.35) !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(124,58,237,0.5) !important;
  }
  .stButton > button:disabled {
    background: rgba(255,255,255,0.06) !important;
    color: #475569 !important; box-shadow: none !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: rgba(10,10,20,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
  }

  /* ── Status pills ── */
  .status-online {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.3);
    color: #34d399; border-radius: 20px; padding: 3px 12px;
    font-size: 0.78rem; font-weight: 600;
  }
  .status-offline {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.3);
    color: #f87171; border-radius: 20px; padding: 3px 12px;
    font-size: 0.78rem; font-weight: 600;
  }

  /* ── Pipeline diagram ── */
  .pipeline-step { display: flex; align-items: center; gap: 0.5rem; color: #94a3b8; font-size: 0.82rem; padding: 0.3rem 0; }
  .pipeline-icon {
    width: 24px; height: 24px; border-radius: 6px;
    background: rgba(167,139,250,0.14);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; flex-shrink: 0;
  }
  .pipeline-arrow { color: #374151; font-size: 0.65rem; padding-left: 11px; }

  /* ── Empty state ── */
  .empty-state { text-align: center; padding: 3rem 2rem; color: #475569; }
  .empty-icon  { font-size: 3.5rem; margin-bottom: 0.8rem; }

  hr { border-color: rgba(255,255,255,0.07) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def api_status() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/status", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def api_ingest(url: str) -> tuple[dict | None, str | None]:
    try:
        r = requests.post(f"{API_BASE}/ingest", json={"url": url}, timeout=120)
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("detail", r.text)
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API. Is `uvicorn api:app --reload` running?"
    except Exception as exc:
        return None, str(exc)


def api_ask(question: str, top_k: int) -> tuple[dict | None, str | None]:
    try:
        r = requests.post(
            f"{API_BASE}/ask",
            json={"question": question, "top_k": top_k},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("detail", r.text)
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API. Is `uvicorn api:app --reload` running?"
    except Exception as exc:
        return None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

for _k, _v in {
    "chat_history": [],
    "video_info":   None,
    "api_online":   False,
    "last_health":  0.0,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Throttled health + status sync
_now = time.time()
if _now - st.session_state.last_health > 5:
    st.session_state.api_online  = api_health()
    st.session_state.last_health = _now
    if st.session_state.api_online and st.session_state.video_info is None:
        _s = api_status()
        if _s and _s.get("ready"):
            st.session_state.video_info = {
                "title":       _s["title"],
                "channel":     _s["channel"],
                "url":         _s["url"],
                "thumbnail":   _s["thumbnail"],
                "chunk_count": _s["chunk_count"],
                "duration_s":  0,
                "elapsed_s":   "",
            }


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.api_online:
        st.markdown('<div class="status-online">● &nbsp;API Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-offline">● &nbsp;API Offline</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#f87171;font-size:0.8rem;margin-top:0.6rem;line-height:1.6;'>
        Start backend:<br>
        <code style='background:rgba(255,255,255,0.08);padding:2px 6px;border-radius:4px;font-size:0.78rem;'>
        uvicorn api:app --reload
        </code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='step-label'>⚙ Settings</div>", unsafe_allow_html=True)
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=8, value=4,
                      help="Number of transcript segments used per answer.")

    st.markdown("---")
    st.markdown("<div class='step-label'>🔄 Pipeline</div>", unsafe_allow_html=True)
    pipeline_steps = [
        ("🔗", "YouTube URL"), ("📝", "Transcript"), ("✂️", "Chunking"),
        ("🧬", "Embeddings"), ("🔍", "Vector Search"), ("🤖", "GPT-4o-mini"), ("💬", "Cited Answer"),
    ]
    html = ""
    for i, (icon, label) in enumerate(pipeline_steps):
        html += f'<div class="pipeline-step"><div class="pipeline-icon">{icon}</div><span>{label}</span></div>'
        if i < len(pipeline_steps) - 1:
            html += '<div class="pipeline-arrow">↓</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='color:#475569;font-size:0.75rem;line-height:1.7;'>
      <b style='color:#64748b;'>Models</b><br>
      Embeddings: text-embedding-3-small<br>
      Generation: gpt-4o-mini<br><br>
      <b style='color:#64748b;'>Storage</b><br>
      In-memory NumPy · No DB needed
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-title">🎬 YouTube RAG</div>
<div class="hero-subtitle">
  Ask anything about any YouTube video — powered by OpenAI embeddings &amp; GPT-4o-mini
</div>
""", unsafe_allow_html=True)

# ── ① Video Input ─────────────────────────────────────────────────────────────

st.markdown("<div class='step-label'>① Load a Video</div>", unsafe_allow_html=True)

col_url, col_btn = st.columns([5, 1])
with col_url:
    video_url = st.text_input(
        "url", label_visibility="collapsed",
        placeholder="https://www.youtube.com/watch?v=...  or  youtu.be/...",
        key="url_input",
    )
with col_btn:
    st.markdown("<div style='height:0.42rem'></div>", unsafe_allow_html=True)
    analyze = st.button("Analyze ▶", use_container_width=True, key="analyze_btn")

if analyze:
    if not video_url.strip():
        st.error("Please enter a YouTube URL.")
    elif not st.session_state.api_online:
        st.error("API is offline — start `uvicorn api:app --reload` first.")
    else:
        with st.spinner("Fetching transcript & building embeddings…"):
            res, err = api_ingest(video_url.strip())
        if err:
            st.error(f"❌ {err}")
        else:
            st.session_state.video_info = {
                "title":       res["title"],
                "channel":     res["channel"],
                "thumbnail":   res["thumbnail"],
                "url":         res["url"],
                "chunk_count": res["chunk_count"],
                "duration_s":  res.get("duration_s", 0),
                "elapsed_s":   res.get("elapsed_s", ""),
            }
            st.session_state.chat_history = []
            st.success(
                f"✅ **{res['title']}** — "
                f"**{res['chunk_count']} chunks** indexed in **{res.get('elapsed_s', '?')}s**"
            )
            st.rerun()

# ── Video info card ───────────────────────────────────────────────────────────

if st.session_state.video_info:
    info   = st.session_state.video_info
    thumb  = info.get("thumbnail", "")
    title  = info.get("title", "Unknown")
    chan   = info.get("channel", "Unknown")
    chunks = info.get("chunk_count", 0)
    dur    = info.get("duration_s", 0)
    dur_s  = f"{dur // 60}m {dur % 60}s" if dur else "—"
    url    = info.get("url", "#")
    el     = info.get("elapsed_s", "")
    el_s   = f" · indexed in {el}s" if el else ""

    img_html = f'<img src="{thumb}" alt="thumb">' if thumb else ""
    st.markdown(f"""
    <div class="video-card">
      {img_html}
      <div class="video-meta">
        <h3>{title}</h3>
        <p>📺 {chan}</p>
        <p>⏱ {dur_s}</p>
        <p><a href="{url}" target="_blank"
              style="color:#60a5fa;text-decoration:none;">Open on YouTube ↗</a></p>
        <div class="badge">🧩 {chunks} chunks{el_s}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── ② Chat Interface ──────────────────────────────────────────────────────────

st.markdown("<div class='step-label'>② Ask Your Questions</div>", unsafe_allow_html=True)

# Render history
if not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">💬</div>
      <p style="color:#64748b;font-size:0.95rem;font-weight:500;">No questions yet</p>
      <p style="color:#475569;font-size:0.85rem;">Load a video above, then ask anything about it.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for item in st.session_state.chat_history:
        if item["role"] == "user":
            st.markdown(f"""
            <div class="chat-user">
              <div class="chat-user-bubble">{item["content"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            meta    = item.get("meta", {})
            tokens  = meta.get("tokens_used", 0)
            model   = meta.get("model", "gpt-4o-mini")
            elapsed = meta.get("elapsed_s", "")
            el_txt  = f" · {elapsed}s" if elapsed else ""

            st.markdown(f"""
            <div class="chat-ai">
              <div class="chat-avatar">🤖</div>
              <div>
                <div class="chat-ai-bubble">{item["content"]}</div>
                <div class="chat-meta">{model} · {tokens} tokens{el_txt}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            sources = item.get("sources", [])
            if sources:
                with st.expander(f"📚 {len(sources)} source chunks", expanded=False):
                    for i, src in enumerate(sources, 1):
                        score = src.get("similarity_score", 0)
                        pct   = f"{score:.0%}"
                        cls   = "score-high" if score >= 0.6 else ("score-med" if score >= 0.4 else "score-low")
                        preview = src.get("content", "")[:300]
                        if len(src.get("content", "")) > 300:
                            preview += "…"
                        st.markdown(f"""
                        <div class="source-chip">
                          <b>Source {i}</b> &nbsp;
                          <span class="{cls}">{pct} relevance</span><br><br>
                          {preview}
                        </div>""", unsafe_allow_html=True)

# Question input
video_ready = st.session_state.video_info is not None
col_q, col_ask = st.columns([5, 1])
with col_q:
    question = st.text_input(
        "q", label_visibility="collapsed",
        placeholder="What is the main topic of this video?",
        key="q_input", disabled=not video_ready,
    )
with col_ask:
    st.markdown("<div style='height:0.42rem'></div>", unsafe_allow_html=True)
    ask_btn = st.button("Ask →", use_container_width=True, key="ask_btn", disabled=not video_ready)

if not video_ready:
    st.markdown("""
    <div style='text-align:center;color:#475569;font-size:0.84rem;padding:0.6rem;'>
    ↑ Load a YouTube video above to enable the chat
    </div>""", unsafe_allow_html=True)

if ask_btn and question.strip():
    if not st.session_state.api_online:
        st.error("API is offline.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question.strip()})
        with st.spinner("Searching transcript & generating answer…"):
            res, err = api_ask(question.strip(), top_k=top_k)
        if err:
            st.session_state.chat_history.append({
                "role": "assistant", "content": f"⚠️ {err}", "sources": [], "meta": {}
            })
        else:
            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": res["answer"],
                "sources": res.get("sources", []),
                "meta": {
                    "tokens_used": res.get("tokens_used", 0),
                    "model":       res.get("model", "gpt-4o-mini"),
                    "elapsed_s":   res.get("elapsed_s", ""),
                },
            })
        st.rerun()
