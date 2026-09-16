# Evaluation contract

## Retrieval (no API key needed)

Twenty supported cases have filename, PDF page and an exact supporting span.
A retrieved chunk is relevant only if its filename/page match and it contains the
normalized supporting span. Page proximity alone does not count. Recall@k is the
fraction of annotated evidence spans retrieved, macro-averaged across questions.
MRR@k is the reciprocal rank of the first fully supporting chunk (zero for misses).
This corpus has one evidence span per supported question, so recall equals hit rate.

We measure dense-only, reciprocal-rank-fused dense+BM25, and the configured reranking backend
on the same queries. Retrieval numbers are first-pass results before any query rewrite.
The score gate is measured separately on all 25: reject unsupported questions and
accept supported questions. Gate rejection is **not** end-to-end refusal correctness.
The default sentence backend combines best sentence cosine (75%) and query-term
coverage (25%) with an initial 0.5 gate. The optional MS MARCO cross-encoder uses
raw logits with a recommended initial 0.0 gate. Neither score is a probability;
neither threshold is calibrated on a held-out set. Do not transfer thresholds
between rerankers blindly. Reports identify the backend used.

## Generated answers (requires GROQ_API_KEY)

Run the actual service including its maximum-one-rewrite loop. Report:

- Groundedness: mean atomic-claim support fraction on non-refused answers. The
  judge sees only the actual cited chunks. Refusals are excluded, not assigned 1.
- Relevance: mean judge score for supported cases; substantive refusals on those
  cases score 0. Service failures are separately counted and have no judge score.
- Refusal correctness: fraction of all 25 with the correct accept/refuse decision;
  service/key/parse/citation failures count as incorrect, not correct refusals.
- Unsupported answer rate: non-refused answers divided by valid runs on the five
  unsupported cases, with coverage shown. Provider/format errors are excluded;
  zero valid runs means N/A, never a zero hallucination rate. This is a behavioral
  hallucination proxy, not a claim-level hallucination metric.
- Generation and judge failures, valid judgment count, token usage and estimated
  costs are exposed. Judge costs are separate from serving costs.

`judge.py` contains the exact 0..1 rubric. It uses temperature 0 and the configured
Groq model; using the same model as generator creates correlated judge bias.
Record explanations in `eval/runs/latest.json`. Human review and a distinct judge
are next steps. No key means N/A; retrieval scores never stand in for answer quality.

## Limitations

This self-authored 10-page handbook and 25 development questions are small and easy.
The negative set includes explicit absence statements, which can aid refusal.
There is no OCR benchmark, independent holdout, multilingual test, multi-document
reasoning benchmark or prompt-injection red-team suite. High scores here establish
that the implementation works on this corpus, not broad product reliability.
