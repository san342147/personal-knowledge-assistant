"""UI is an HTTP client; indexing and model inference belong to the API process."""
import os

import httpx
import streamlit as st

API = os.environ.get("GROUNDEDDESK_API_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="GroundedDesk", page_icon="📄", layout="centered")
st.title("GroundedDesk")
st.caption("Ask your PDFs. Get cited answers. Measure whether the system is actually grounded.")
st.info("Files and embeddings stay on this machine. Questions and selected evidence are sent to Groq. "
        "First upload downloads MiniLM. The default sentence reranker reuses that local model.")


def request(method, path, **kwargs):
    try:
        response = httpx.request(method, API + path, timeout=300, **kwargs)
        data = response.json()
        if path == "/health":
            return data
        if response.is_error:
            if "answer" in data:
                return data
            st.error(data.get("detail", "API unavailable"))
            for warning in data.get("warnings", []):
                st.warning(warning)
            return None
        return data
    except (httpx.HTTPError, ValueError):
        st.error("Cannot reach the API. Start GroundedDesk with start.bat and check the API window.")
        return None


with st.form("upload"):
    upload = st.file_uploader("Upload PDF, TXT or MD", type=["pdf", "txt", "md"])
    if st.form_submit_button("Index document") and upload:
        with st.spinner("Extracting and indexing locally..."):
            data = request("POST", "/ingest", files={"file": (upload.name, upload.getvalue(), upload.type)})
        if data:
            st.success(f"{data['chunks_added']} new chunks indexed from {data['filename']}.")
            for warning in data["warnings"]:
                st.warning(warning)

with st.form("question"):
    question = st.text_input("Question", placeholder="How long can I borrow equipment?")
    top_k = st.slider("Evidence chunks", 1, 10, 4)
    submit = st.form_submit_button("Ask", type="primary")
if submit and question.strip():
    with st.spinner("Retrieving, reranking, and checking evidence..."):
        st.session_state["answer"] = request("POST", "/ask", json={"question": question, "top_k": top_k})
if st.session_state.get("answer"):
    answer = st.session_state["answer"]
    (st.warning if answer["refused"] else st.write)(answer["answer"])
    for citation in answer["citations"]:
        with st.expander(f"{citation['filename']} · page {citation['page']} · {citation['chunk_id']}", expanded=True):
            st.text(citation["snippet"])
    st.caption(f"{answer['latency_ms']:.0f} ms · {answer['tokens_in']} input / {answer['tokens_out']} output tokens · "
               f"estimated ${answer['estimated_cost_usd']:.6f} · {answer['reason']}")
    if answer["rewritten_query"]:
        st.caption("One retrieval rewrite: " + answer["rewritten_query"])

with st.expander("System health and metrics"):
    if st.button("Check system"):
        st.json(request("GET", "/health"))
        st.json(request("GET", "/metrics"))
