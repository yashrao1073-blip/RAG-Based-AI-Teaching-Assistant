from __future__ import annotations

import html
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
EMBEDDINGS_PATH = ROOT / "embeddings.joblib"
VIDEOS_DIR = ROOT / "videos"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
CHAT_MODEL = "llama3.2"
TOP_K = 5
EMBED_BATCH_SIZE = 64

st.set_page_config(
    page_title="Course Companion · Sigma Web Development",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#17211f; --muted:#71807a; --green:#176b52; --mint:#e7f3ed; --line:#e8ece9; --paper:#fbfcfa; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
    .stApp { background:var(--paper); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#f3f6f3; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div { padding-top:1.4rem; }
    .brand { font:800 1.13rem Manrope,sans-serif; letter-spacing:-.04em; color:var(--ink); }
    .brand-mark { display:inline-flex; align-items:center; justify-content:center; width:34px;height:34px;border-radius:11px;background:var(--green);color:white;margin-right:10px;font-size:17px; }
    .eyebrow { color:var(--green); text-transform:uppercase; font-size:.69rem; font-weight:700; letter-spacing:.14em; }
    h1 { font-family:Manrope,sans-serif !important; letter-spacing:-.055em !important; font-weight:700 !important; }
    .hero-note { color:var(--muted); font-size:1.02rem; margin-top:-.5rem; }
    .status-pill { display:inline-flex; align-items:center; gap:8px; padding:7px 11px; border:1px solid var(--line); border-radius:99px; background:white; color:#56645e; font-size:.78rem; }
    .status-dot { width:8px;height:8px;border-radius:50%;background:#d49b43; }
    .status-dot.ok { background:#30a575; }
    .answer-card { background:white; border:1px solid var(--line); border-radius:18px; padding:22px 24px; line-height:1.72; box-shadow:0 8px 28px rgba(20,45,34,.035); }
    .section-label { font:700 .78rem Manrope,sans-serif; text-transform:uppercase; letter-spacing:.1em; color:#697871; }
    .source-card { background:white; border:1px solid var(--line); border-radius:15px; padding:16px 18px; margin:8px 0 12px; }
    .source-title { font-weight:700; font-size:.96rem; margin:0 0 5px; }
    .source-meta { color:var(--muted);font-size:.8rem; }
    .timestamp { display:inline-block;background:var(--mint);color:var(--green);font-weight:700;font-size:.79rem;padding:5px 9px;border-radius:8px;margin-right:8px; }
    .source-text { color:#4e5b55;font-size:.88rem;line-height:1.55;margin-top:11px; }
    .welcome { border:1px solid var(--line);border-radius:20px;background:linear-gradient(135deg,#fff 0%,#f3f8f4 100%);padding:25px 27px; }
    .welcome h3 { font:700 1.08rem Manrope,sans-serif;margin:0 0 8px; }
    .welcome p { color:var(--muted);margin:0;line-height:1.6; }
    div[data-testid="stChatInput"] { border-radius:16px; }
    .stButton > button { border-radius:10px; }
    div[data-testid="stMetric"] { background:white;border:1px solid var(--line);border-radius:13px;padding:12px 15px; }
    hr { border-color:var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_chunks() -> pd.DataFrame:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Could not find {EMBEDDINGS_PATH.name} in the project folder.")
    try:
        chunks = joblib.load(EMBEDDINGS_PATH)
    except (NotImplementedError, ValueError, TypeError, AttributeError) as exc:
        raise RuntimeError(
            "The saved embeddings were created with an incompatible pandas version. "
            "Rebuild them from the transcript JSON files in the sidebar."
        ) from exc
    # Newer versions save plain records rather than a pandas object, so the
    # embeddings can be loaded across pandas versions.
    if not isinstance(chunks, pd.DataFrame):
        chunks = pd.DataFrame.from_records(chunks)
    required = {"embedding", "title", "number", "start", "end", "text"}
    missing = required.difference(chunks.columns)
    if missing:
        raise ValueError("The embeddings file is missing: " + ", ".join(sorted(missing)))
    return chunks


def rebuild_embeddings() -> int:
    """Re-embed transcript JSON chunks and save a pandas-version-neutral file."""
    json_dir = ROOT / "jsons"
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No transcript JSON files were found in the jsons folder.")

    records = []
    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as source:
            records.extend(json.load(source).get("chunks", []))
    if not records:
        raise ValueError("The transcript JSON files contain no chunks to embed.")

    progress = st.progress(0, text="Preparing transcript chunks…")
    try:
        for start in range(0, len(records), EMBED_BATCH_SIZE):
            batch = records[start : start + EMBED_BATCH_SIZE]
            result = ollama_post(
                "/api/embed",
                {"model": EMBED_MODEL, "input": [item["text"] for item in batch]},
            )
            vectors = result.get("embeddings")
            if vectors is None or len(vectors) != len(batch):
                raise RuntimeError("Ollama returned an unexpected number of embeddings.")
            for item, vector in zip(batch, vectors):
                item["embedding"] = vector
            progress.progress(
                min(start + len(batch), len(records)) / len(records),
                text=f"Embedding transcript chunks: {min(start + len(batch), len(records))}/{len(records)}",
            )
        # Store ordinary Python records; pandas can construct the table later
        # without needing to unpickle a pandas-specific StringDtype.
        joblib.dump(records, EMBEDDINGS_PATH)
    finally:
        progress.empty()
    load_chunks.clear()
    return len(records)


def ollama_post(endpoint: str, payload: dict) -> dict:
    try:
        response = requests.post(f"{OLLAMA_URL}{endpoint}", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("I can’t reach Ollama. Start Ollama locally, then try again.") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("Ollama took too long to respond. Please try again.") from exc
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.text[:250] if exc.response is not None else str(exc)
        raise RuntimeError(f"Ollama returned an error: {detail}") from exc


def retrieve(question: str, chunks: pd.DataFrame) -> pd.DataFrame:
    result = ollama_post("/api/embed", {"model": EMBED_MODEL, "input": [question]})
    query_vector = result["embeddings"][0]
    vectors = np.vstack(chunks["embedding"].to_numpy())
    scores = cosine_similarity(vectors, [query_vector]).ravel()
    indices = np.argsort(scores)[::-1][:TOP_K]
    retrieved = chunks.iloc[indices].copy()
    retrieved["similarity"] = scores[indices]
    return retrieved


def answer_question(question: str, retrieved: pd.DataFrame) -> str:
    context = retrieved[["title", "number", "start", "end", "text"]].to_json(orient="records")
    prompt = f"""You are the friendly teaching assistant for the Sigma web development course.
Use only the lesson transcript excerpts below to answer. Explain where the topic is taught and guide the student to the relevant lesson and timestamp. Mention uncertainty if the excerpts only partially answer the question. If the question is unrelated to this course, politely say you can only help with the course.

Transcript excerpts:
{context}

Student question: {question}
"""
    result = ollama_post(
        "/api/generate",
        {"model": CHAT_MODEL, "prompt": prompt, "stream": False},
    )
    return result.get("response", "I couldn’t generate an answer. Please try again.").strip()


def format_time(seconds: float) -> str:
    seconds = max(0, int(float(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02}:{secs:02}" if hours else f"{minutes}:{secs:02}"


def video_for(number: object) -> Path | None:
    if not VIDEOS_DIR.exists():
        return None
    match = re.search(r"\d+", str(number))
    if not match:
        return None
    lesson_number = int(match.group())
    candidates = list(VIDEOS_DIR.glob("*.mp4"))
    return next((path for path in candidates if re.search(rf"Tutorial\s*#\s*0*{lesson_number}(?:\D|$)", path.name, re.I)), None)


def render_sources(sources: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Relevant moments</div>', unsafe_allow_html=True)
    for _, row in sources.iterrows():
        title = html.escape(str(row["title"]))
        number = html.escape(str(row["number"]))
        text = html.escape(str(row["text"]).strip())
        start, end = float(row["start"]), float(row["end"])
        st.markdown(
            f'<div class="source-card"><div class="source-title">{title}</div>'
            f'<span class="timestamp">▶ {format_time(start)}</span>'
            f'<span class="source-meta">Lesson {number} · {format_time(start)}–{format_time(end)}</span>'
            f'<div class="source-text">“{text}”</div></div>',
            unsafe_allow_html=True,
        )
        video = video_for(row["number"])
        if video:
            with st.expander(f"Watch this moment · Lesson {number}", expanded=False):
                st.video(str(video), start_time=int(start))


def main() -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-mark">✦</span>Course Companion</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Your course library**")
        try:
            chunks = load_chunks()
            lesson_count = chunks["number"].astype(str).nunique()
            st.caption(f"{lesson_count} lessons · Searchable transcript library")
            st.markdown('<div class="status-pill"><span class="status-dot ok"></span>Course library ready</div>', unsafe_allow_html=True)
        except Exception as error:
            chunks = None
            st.error(str(error))
            if "incompatible pandas version" in str(error):
                st.caption("This rebuilds the search vectors from your transcript JSON files. Ollama must be running with bge-m3 installed.")
                if st.button("Rebuild course embeddings", use_container_width=True):
                    try:
                        count = rebuild_embeddings()
                        st.success(f"Rebuilt embeddings for {count:,} transcript chunks.")
                        st.rerun()
                    except Exception as rebuild_error:
                        st.error(str(rebuild_error))
        st.markdown("\n---")
        st.markdown("**How to use**")
        st.caption("Ask a question about a web development topic. I’ll find the lesson moments that explain it and show you where to watch.")
        st.markdown("\n---")
        st.caption(f"Embeddings · {EMBED_MODEL}")
        st.caption(f"Answer model · {CHAT_MODEL}")
        st.caption("Powered by your local Ollama")

    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.markdown('<div class="eyebrow">Sigma web development course</div>', unsafe_allow_html=True)
        st.title("Learn right where it’s taught.")
        st.markdown('<div class="hero-note">Ask a course question and jump straight to the lesson moment that explains it.</div>', unsafe_allow_html=True)
    with top_right:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div class="status-pill"><span class="status-dot"></span>Local AI</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])
                if message.get("sources") is not None:
                    render_sources(message["sources"])
            else:
                st.markdown(message["content"])

    if not st.session_state.messages:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="welcome"><h3>What are you learning today?</h3>'
            '<p>Try asking where forms are covered, how CSS selectors work, or what semantic HTML tags do. '
            'You’ll get a clear answer with the lesson and timestamp.</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        prompts = ["Where are HTML forms taught?", "Explain CSS selectors", "What are semantic tags?"]
        cols = st.columns(3)
        for col, prompt in zip(cols, prompts):
            with col:
                if st.button(prompt, use_container_width=True):
                    st.session_state.pending_question = prompt
                    st.rerun()

    question = st.chat_input("Ask about a lesson or web development topic…", disabled=chunks is None)
    if question:
        st.session_state.pending_question = question
        st.rerun()

    pending = st.session_state.pop("pending_question", None)
    if pending:
        st.session_state.messages.append({"role": "user", "content": pending})
        with st.chat_message("user"):
            st.markdown(pending)
        with st.chat_message("assistant"):
            try:
                if chunks is None:
                    raise RuntimeError("The course search library is unavailable. Fix its loading error in the sidebar first.")
                with st.spinner("Searching your course and finding the right moment…"):
                    sources = retrieve(pending, chunks)
                    answer = answer_question(pending, sources)
                st.markdown(answer)
                render_sources(sources)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as error:
                message = str(error)
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message, "sources": None})


if __name__ == "__main__":
    main()
