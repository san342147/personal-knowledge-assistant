"""
Personal Knowledge Assistant — Streamlit UI entrypoint.

Run from project root:
    streamlit run app/main.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Windows: avoid noisy Proactor ConnectionResetError (WinError 10054)
# when the browser closes a websocket/tab. Harmless but scary in the console.
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Ensure project root is on sys.path when launched via `streamlit run app/main.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.core.config import get_settings
from app.core.document_processor import (
    DocumentProcessingError,
    clear_upload_dir,
    process_files,
)
from app.core.rag_chain import RAGService
from app.core.vector_store import (
    VectorStoreError,
    build_vector_store,
    clear_vector_store,
    load_vector_store,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pka")

st.set_page_config(
    page_title="Personal Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; max-width: 1100px; }
      .pka-hero {
        padding: 1rem 1.25rem 0.5rem 0;
        margin-bottom: 0.5rem;
      }
      .pka-hero h1 {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
      }
      .pka-hero p { color: #64748b; margin: 0; }
      .source-card {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.55rem;
        background: #f8fafc;
        font-size: 0.9rem;
      }
      .source-card strong { color: #0f172a; }
      div[data-testid="stChatMessage"] { font-size: 0.98rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "messages": [],          # {role, content, sources?}
        "vector_ready": False,
        "doc_count": 0,
        "chunk_count": 0,
        "indexed_files": [],
        "rag": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _settings():
    return get_settings()


def _ensure_rag(settings) -> RAGService:
    if st.session_state.rag is None:
        st.session_state.rag = RAGService(settings)
    return st.session_state.rag


def _vector_dir_has_data(vector_dir) -> bool:
    try:
        p = Path(vector_dir)
        return p.exists() and any(p.iterdir())
    except Exception:
        return False


def _try_load_existing(settings) -> None:
    """Load on-disk vector store into session if present (safe on failure)."""
    if st.session_state.vector_ready:
        return
    if not _vector_dir_has_data(settings.vector_dir):
        return
    try:
        with st.spinner(
            "Loading knowledge base (first open may take 30–60s for the embedding model)…"
        ):
            store = load_vector_store(settings.vector_dir, settings.embedding_model)
    except Exception as exc:
        # Heavy embedding models can fail transiently; do not crash the UI.
        logger.exception("Failed to load existing vector store")
        st.session_state["_kb_load_error"] = str(exc)
        return
    if store is not None:
        rag = _ensure_rag(settings)
        rag.set_store(store)
        st.session_state.vector_ready = True
        st.session_state.pop("_kb_load_error", None)
        # Approximate counts from collection if possible
        try:
            st.session_state.chunk_count = store._collection.count()  # noqa: SLF001
        except Exception:
            st.session_state.chunk_count = st.session_state.chunk_count or 0


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def action_build_index(uploaded_files, settings) -> None:
    if not uploaded_files:
        st.warning("Please upload at least one PDF or TXT file.")
        return

    progress = st.progress(0, text="Saving & reading documents…")
    try:
        progress.progress(20, text="Chunking documents…")
        chunks = process_files(
            uploaded_files,
            settings.upload_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        progress.progress(45, text="Creating embeddings (first run may download the model)…")
        store = build_vector_store(
            chunks,
            settings.vector_dir,
            settings.embedding_model,
        )
        progress.progress(90, text="Finalizing knowledge base…")

        rag = _ensure_rag(settings)
        rag.set_store(store)

        sources = sorted({c.metadata.get("source", "?") for c in chunks})
        st.session_state.vector_ready = True
        st.session_state.chunk_count = len(chunks)
        st.session_state.doc_count = len(sources)
        st.session_state.indexed_files = sources
        st.session_state.messages = []

        progress.progress(100, text="Done!")
        st.success(
            f"Indexed **{len(sources)}** file(s) into **{len(chunks)}** chunks. "
            "You can start chatting."
        )
    except (DocumentProcessingError, VectorStoreError) as exc:
        st.error(str(exc))
    except Exception as exc:
        logger.exception("Index build failed")
        st.error(f"Unexpected error while building knowledge base: {exc}")
    finally:
        progress.empty()


def action_clear_kb(settings) -> None:
    try:
        clear_vector_store(settings.vector_dir)
        clear_upload_dir(settings.upload_dir)
        st.session_state.vector_ready = False
        st.session_state.doc_count = 0
        st.session_state.chunk_count = 0
        st.session_state.indexed_files = []
        st.session_state.messages = []
        if st.session_state.rag is not None:
            st.session_state.rag.set_store(None)
        st.success("Knowledge base and uploads cleared.")
    except Exception as exc:
        st.error(f"Failed to clear knowledge base: {exc}")


def action_clear_chat() -> None:
    st.session_state.messages = []
    st.toast("Chat history cleared", icon="💬")


def action_ask(question: str, settings) -> None:
    rag = _ensure_rag(settings)
    if not st.session_state.vector_ready:
        # try reload
        _try_load_existing(settings)
    if st.session_state.rag and st.session_state.vector_ready:
        # ensure store attached
        if st.session_state.rag.vector_store is None:
            store = load_vector_store(settings.vector_dir, settings.embedding_model)
            st.session_state.rag.set_store(store)

    st.session_state.messages.append({"role": "user", "content": question, "sources": []})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context & generating answer…"):
            result = rag.ask(question, top_k=settings.top_k)

        if not result.ok:
            st.error(result.error)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ {result.error}", "sources": result.sources}
            )
            return

        st.markdown(result.answer)
        if result.sources:
            with st.expander(f"Sources ({len(result.sources)} chunks)", expanded=True):
                _render_sources(result.sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
            }
        )


def _render_sources(sources) -> None:
    for i, doc in enumerate(sources, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)
        page_bit = f" · page {page}" if page is not None and str(page) != "" else ""
        snippet = (doc.page_content or "").strip()
        if len(snippet) > 420:
            snippet = snippet[:420] + "…"
        st.markdown(
            f"""
            <div class="source-card">
              <strong>[{i}] {src}</strong>{page_bit}<br/>
              <span style="color:#475569">{snippet}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
def main() -> None:
    _init_state()
    settings = _settings()
    _try_load_existing(settings)

    # Header
    st.markdown(
        """
        <div class="pka-hero">
          <h1>📚 Personal Knowledge Assistant</h1>
          <p>Upload PDFs or text files, build a private knowledge base, and chat with RAG-powered answers + sources.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar — knowledge base controls
    with st.sidebar:
        st.header("Knowledge Base")
        st.caption("Files stay on your machine. Embeddings are local.")

        uploaded = st.file_uploader(
            "Upload PDF / TXT",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="You can select multiple files at once.",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            build_clicked = st.button(
                "Build / Rebuild",
                type="primary",
                use_container_width=True,
                disabled=not uploaded,
            )
        with col_b:
            clear_kb = st.button("Clear KB", use_container_width=True)

        if build_clicked:
            action_build_index(uploaded, settings)

        if clear_kb:
            action_clear_kb(settings)

        st.divider()
        st.subheader("Status")
        if st.session_state.vector_ready:
            st.success("Knowledge base ready")
        elif st.session_state.get("_kb_load_error"):
            st.warning(
                "Could not load saved knowledge base. "
                "Try **Build / Rebuild** again.\n\n"
                f"`{st.session_state['_kb_load_error'][:200]}`"
            )
        else:
            st.info("No knowledge base yet")

        st.metric("Documents", st.session_state.doc_count)
        st.metric("Chunks", st.session_state.chunk_count)

        if st.session_state.indexed_files:
            st.caption("Indexed files")
            for name in st.session_state.indexed_files:
                st.markdown(f"- `{name}`")

        st.divider()
        st.subheader("Model settings")
        st.text(f"LLM: {settings.llm_model}")
        st.text(f"API: {settings.base_url}")
        st.text(f"Embed: {settings.embedding_model.split('/')[-1]}")
        st.text(f"Top-K: {settings.top_k} · chunk {settings.chunk_size}")

        if settings.llm_configured:
            st.success("API key detected")
        else:
            st.error("Missing API key — set GROQ_API_KEY in .env")

        st.button("Clear chat", on_click=action_clear_chat, use_container_width=True)

        with st.expander("Setup help"):
            st.markdown(
                """
1. Edit `.env` in the project folder
2. Set `GROQ_API_KEY` from [console.groq.com/keys](https://console.groq.com/keys)
3. Base URL should be `https://api.groq.com/openai/v1`
4. Restart Streamlit after editing `.env`
                """
            )

    # Main chat area
    left, right = st.columns([2.2, 1])
    with right:
        st.subheader("How it works")
        st.markdown(
            """
1. **Upload** PDF/TXT files  
2. Click **Build / Rebuild**  
3. **Ask** questions in the chat  
4. See **sources** under each answer  

**Stack:** Streamlit · LangChain · ChromaDB · sentence-transformers · OpenAI-compatible LLM (Grok / xAI)
            """
        )
        if not st.session_state.vector_ready:
            st.warning("Upload & build a knowledge base to enable chat.")

    with left:
        st.subheader("Chat")
        # History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources", expanded=False):
                        _render_sources(msg["sources"])

        prompt = st.chat_input(
            "Ask a question about your documents…",
            disabled=not st.session_state.vector_ready or not settings.llm_configured,
        )
        if prompt:
            # show user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            action_ask(prompt, settings)
            # Streamlit will rerun; history already includes both messages
            st.rerun()


if __name__ == "__main__":
    main()
