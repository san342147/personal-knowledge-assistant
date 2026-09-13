# Personal Knowledge Assistant

I got tired of grepping through PDFs, so I built a small RAG app that stays on my machine.

Drop in PDF / TXT / Markdown. It chunks the text, embeds it locally with MiniLM, and stores vectors in Chroma. Questions go through similarity search (top-k = 4), then an OpenAI-compatible chat model (Groq or xAI). The UI shows the answer plus the chunks it used, including page numbers.

Nothing in the knowledge base is sent to an embedding API. Only the retrieved snippets and your question leave the box, and that is the LLM call.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain-green)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange)

---

## Features

| Feature | Description |
|--------|-------------|
| Multi-file upload | PDF, TXT, Markdown |
| Smart chunking | Recursive splitter with configurable size/overlap |
| Local embeddings | `sentence-transformers` (no embedding API cost) |
| Persistent vector DB | ChromaDB on disk under `vectorstore/` |
| RAG chat | Top-K retrieval + LLM answer |
| Source display | Document name + page + snippet |
| Session memory | Chat history in Streamlit session state |
| Clear KB / Clear chat | Reset index or conversation |
| OpenAI-compatible LLM | Groq by default (`openai/gpt-oss-20b`); xAI or any OpenAI-style endpoint also works |

---

## Architecture

```
User uploads ──► Document processor (load + chunk)
                        │
                        ▼
              Embeddings (local MiniLM)
                        │
                        ▼
                 Chroma vector store
                        │
User question ──► Similarity search (Top-K)
                        │
                        ▼
              Context + prompt ──► LLM (Groq / OpenAI-compatible)
                        │
                        ▼
              Answer + sources in Streamlit UI
```

---

## Project structure

```
personal-knowledge-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Streamlit UI entrypoint
│   └── core/
│       ├── __init__.py
│       ├── config.py           # Env / settings
│       ├── document_processor.py
│       ├── vector_store.py     # Chroma + embeddings
│       ├── llm.py              # OpenAI-compatible chat client
│       └── rag_chain.py        # Retrieve + generate
├── data/
│   └── uploads/                # Saved uploads (gitignored)
├── vectorstore/                # Chroma persistence (gitignored)
├── requirements.txt
├── .env.example
├── .gitignore
├── start.bat                   # Windows one-click run
└── README.md
```

---

## Quick start (Windows)

### Option A — one click

1. Double-click **`start.bat`**
2. Wait for dependencies (first run downloads embedding model)
3. Edit **`.env`** and set your API key
4. Browser opens at **http://localhost:8501**

### Option B — manual

```bash
cd "C:\Users\SANTHOSH A\GROK\personal-knowledge-assistant"

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

copy .env.example .env
# Edit .env → set GROQ_API_KEY=...

streamlit run app/main.py
```

---

## API key setup (Groq)

1. Create a key at [https://console.groq.com/keys](https://console.groq.com/keys)
2. In `.env`:

```env
GROQ_API_KEY=gsk_xxxxxxxx
OPENAI_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
```

`llama-3.3-70b-versatile` is Enterprise-only on Groq now. Use `openai/gpt-oss-20b` or `openai/gpt-oss-120b` on the free/developer plan.

Also accepted: `XAI_API_KEY`, `GROK_API_KEY`, `OPENAI_API_KEY`.

### Switch to xAI / Grok

```env
XAI_API_KEY=xai-xxxxxxxx
OPENAI_BASE_URL=https://api.x.ai/v1
LLM_MODEL=grok-4.5
```

No code changes required.

---

## Usage

1. **Upload** one or more PDF/TXT files in the sidebar  
2. Click **Build / Rebuild** (creates embeddings + Chroma index)  
3. Ask questions in the chat  
4. Expand **Sources** under each answer  
5. **Clear KB** removes uploads + vector index  
6. **Clear chat** resets conversation only  

---

## Configuration reference

| Variable | Default | Meaning |
|----------|---------|---------|
| `GROQ_API_KEY` | — | Groq API key (recommended) |
| `XAI_API_KEY` | — | xAI / Grok API key (optional) |
| `OPENAI_BASE_URL` | `https://api.groq.com/openai/v1` | Chat API base URL |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Chat model id |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedder |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `4` | Retrieved chunks per question |

---

## Requirements

- Python **3.10+**
- ~1–2 GB disk for venv + embedding model (first install)
- Internet for LLM API calls (embeddings run offline after model download)

---

## Error handling

| Situation | Behavior |
|-----------|----------|
| Bad/empty upload | Clear error, no crash |
| Unsupported type | Skipped with warning |
| Missing API key | Sidebar error; chat disabled |
| LLM / network failure | User-visible message; sources still shown if retrieval worked |
| Empty knowledge base | Guided prompt to upload first |

---

## License

MIT — use freely in portfolios and demos. You are responsible for your own API keys and document privacy.
