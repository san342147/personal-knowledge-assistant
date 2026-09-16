# GroundedDesk evaluation

Run: 2026-09-16T15:36:52+00:00. Mode: retrieval only; generation NOT measured.

| Metric | Result | Denominator / scope |
|---|---:|---|
| dense recall@4 | 1.000 | 20 supported questions |
| dense MRR@4 | 0.950 | 20 supported questions |
| hybrid recall@4 | 1.000 | 20 supported questions |
| hybrid MRR@4 | 0.975 | 20 supported questions |
| reranked recall@4 | 1.000 | 20 supported questions |
| reranked MRR@4 | 0.975 | 20 supported questions |
| Retrieval gate: supported acceptance | 17/20 | First pass, no generation |
| Retrieval gate: unsupported rejection | 5/5 | First pass, no generation |
| Warm reranked retrieval p50 / p95 | 40 / 48 ms | 25 CPU queries; excludes model loading |
| Faithfulness (LLM judge) | N/A | Groq generation not run |
| Answer relevance | N/A | Groq generation not run |
| End-to-end refusal correctness | N/A | Groq generation not run |
| Unsupported answer / hallucination rate | N/A | Groq generation not run |
| Generation latency and cost | N/A | Groq generation not run |

Corpus: 10 self-authored PDF pages, 10 chunks; 25 questions (20 supported, 5 unsupported).

Corpus SHA-256: `412b3071b8846923b0cb637914535871b3903c0a925dae1cc8cfd4e07db33e52`.

Python 3.12.14 on Windows AMD64. Embedding: `sentence-transformers/all-MiniLM-L6-v2`. Reranker backend: `sentence` (same MiniLM model; sentence cosine + lexical coverage). Chunk size/overlap: 900/150 characters. Candidate pool: 20. Gate score threshold: 0.5. Indexing including embedding model load: 58428 ms.

This is a small development corpus with lexical overlap, not a held-out accuracy claim. Compare the ablations before claiming that hybrid retrieval or reranking helps. See [RUBRIC.md](RUBRIC.md) for metric definitions and limitations.
