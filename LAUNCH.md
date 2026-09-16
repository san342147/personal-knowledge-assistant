# GitHub and LinkedIn copy

## Repository

Name: `groundeddesk`

About: Local-first PDF Q&A with hybrid retrieval, local reranking, Groq citations,
and a reproducible evaluation harness. FastAPI + Streamlit.

Topics: `rag`, `fastapi`, `groq`, `evaluation`, `python`, `chromadb`

Local demo: run `start.bat`, add a Groq key to the ignored `.env`, then upload
`data/sample/fieldguide.pdf` at http://localhost:8501. No public hosted demo is claimed.

## LinkedIn draft

I built GroundedDesk to make my document Q&A work easier to inspect and measure.

My earlier RAG project returned source snippets. This version adds a FastAPI backend,
dense + BM25 retrieval, local reranking, cited answers through Groq, and a single
query rewrite when retrieval is weak. It also has a Windows launcher and 27 offline tests.

The evaluation was the most useful part. On a self-authored handbook with 25 questions,
hybrid retrieval improved MRR@4 from 0.950 to 0.975. Sentence reranking did not improve
that ranking score further. Its initial relevance gate rejected all five unsupported
questions, but also rejected three questions the documents could answer.

That is a real limitation. My next step is to calibrate the gate on a separate set
with harder negatives, then measure the tradeoff. I have not measured generated-answer
faithfulness yet; the final report is retrieval-only, and those metrics are marked N/A.

The repo includes the sample PDF, golden questions, evaluation code and measured report.
The default reranker reuses the local embedding model; a cross-encoder is an optional
backend. Documents and indexes stay local, while selected context goes to Groq.

Repository link: add the actual GitHub URL after publishing `groundeddesk`.
