"""python -m eval.run [--retrieval-only] [--top-k 4] [--update-readme]"""
import argparse
import hashlib
import json
import platform
import re
import statistics
import tempfile
import time
from datetime import datetime, timezone

from app.config import ROOT, Settings
from app.generation import GroqGenerator
from app.ingest import parse_document
from app.retrieval import Retriever
from app.service import DeskService
from app.telemetry import Telemetry
from eval.judge import judge


def normalize(text):
    return " ".join(text.lower().split())


def retrieval_metrics(hits, evidence):
    def supports(hit, item):
        return (hit.chunk.filename == item["filename"] and hit.chunk.page == item["page"]
                and normalize(item["text"]) in normalize(hit.chunk.text))
    covered = sum(any(supports(h, e) for h in hits) for e in evidence)
    rank = next((i for i, h in enumerate(hits, 1) if any(supports(h, e) for e in evidence)), None)
    return covered / len(evidence), 1 / rank if rank else 0


def mean(values):
    return statistics.mean(values) if values else None


def unsupported_answer_metric(rows, failure_reasons):
    valid = [r for r in rows if not r["answerable"] and r["answer"]["reason"] not in failure_reasons]
    rate = sum(not r["answer"]["refused"] for r in valid) / len(valid) if valid else None
    return rate, len(valid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=4, choices=range(1, 11))
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    settings = Settings(top_k=args.top_k)
    golden = json.loads((ROOT / "eval/golden.json").read_text(encoding="utf-8"))
    corpus = (ROOT / "data/sample/fieldguide.pdf").read_bytes()
    run_dir = ROOT / "eval/runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Retain isolated stores for reproducibility. No destructive cleanup and no uploaded files.
    directory = tempfile.mkdtemp(prefix="benchmark-", dir=run_dir)
    settings.vectorstore_path = directory
    from pathlib import Path
    settings.vectorstore_path = Path(directory)
    print("Opening isolated Chroma store...", flush=True)
    retriever = Retriever(settings)
    parsed = parse_document("fieldguide.pdf", corpus, settings.chunk_size, settings.chunk_overlap)
    started = time.perf_counter()
    print(f"Embedding and indexing {len(parsed.chunks)} chunks (first run downloads MiniLM)...", flush=True)
    retriever.add(parsed.chunks)
    indexing_ms = (time.perf_counter()-started)*1000
    generator = GroqGenerator(settings)
    live = generator.available and not args.retrieval_only
    service = DeskService(settings, retriever, generator, Telemetry(run_dir / "logs"))
    rows, latencies = [], []
    stage_scores = {stage: [] for stage in ("dense", "hybrid", "reranked")}
    # Warm model loading once, outside per-query latency samples.
    print(f"Warming retrieval and {settings.reranker_backend} reranker...", flush=True)
    retriever.search("station opening hours", args.top_k)
    for item in golden:
        row = {"id": item["id"], "question": item["question"], "answerable": item["answerable"]}
        for stage in stage_scores:
            started = time.perf_counter()
            hits = retriever.search(item["question"], args.top_k, stage)
            if stage == "reranked":
                latencies.append((time.perf_counter()-started)*1000)
                row["gate_accepts"] = bool(hits and hits[0].score >= settings.rerank_threshold)
                row["top_score"] = hits[0].score if hits else None
            row[stage] = [{"id": h.chunk.id, "page": h.chunk.page, "score": h.score} for h in hits]
            if item["answerable"]:
                score = retrieval_metrics(hits, item["evidence"])
                stage_scores[stage].append(score)
                row[stage + "_metrics"] = {"recall": score[0], "mrr": score[1]}
        if live:
            answer = service.ask(item["question"], args.top_k)
            row["answer"] = answer.model_dump()
            if not answer.refused:
                try:
                    judgment, usage = judge(generator, item["question"], item["expected_answer"], answer)
                    row["judge"] = judgment
                    row["judge_tokens_in"], row["judge_tokens_out"] = usage.tokens_in, usage.tokens_out
                except Exception as exc:
                    row["judge"] = {"error": type(exc).__name__}
        rows.append(row)
        print(f"{item['id']}: gate={'accept' if row['gate_accepts'] else 'reject'} score={row['top_score']:.3f}", flush=True)
    table = ["| Metric | Result | Denominator / scope |", "|---|---:|---|"]
    for stage, values in stage_scores.items():
        table += [f"| {stage} recall@{args.top_k} | {mean([v[0] for v in values]):.3f} | 20 supported questions |",
                  f"| {stage} MRR@{args.top_k} | {mean([v[1] for v in values]):.3f} | 20 supported questions |"]
    positives, negatives = [r for r in rows if r["answerable"]], [r for r in rows if not r["answerable"]]
    table += [f"| Retrieval gate: supported acceptance | {sum(r['gate_accepts'] for r in positives)}/20 | First pass, no generation |",
              f"| Retrieval gate: unsupported rejection | {sum(not r['gate_accepts'] for r in negatives)}/5 | First pass, no generation |",
              f"| Warm reranked retrieval p50 / p95 | {statistics.median(latencies):.0f} / {sorted(latencies)[int(.95*(len(latencies)-1))]:.0f} ms | 25 CPU queries; excludes model loading |"]
    if live:
        failures = {"service_error", "generation_unavailable", "invalid_generation", "invalid_citations"}
        valid = [r for r in rows if r["answer"]["reason"] not in failures]
        judgments = [r["judge"] for r in rows if "faithfulness" in r.get("judge", {})]
        relevance = [0 if r["answer"]["refused"] else r.get("judge", {}).get("relevance")
                     for r in positives if r["answer"]["reason"] not in failures]
        relevance = [v for v in relevance if v is not None]
        def show(value):
            return "N/A" if value is None else f"{value:.3f}"
        correct = sum(r["answer"]["refused"] != r["answerable"] for r in valid)
        unsupported_rate, negative_coverage = unsupported_answer_metric(rows, failures)
        serving_cost = sum(r["answer"]["estimated_cost_usd"] for r in rows)
        judge_cost = sum(r.get("judge_tokens_in", 0)*settings.input_usd_per_million +
                         r.get("judge_tokens_out", 0)*settings.output_usd_per_million for r in rows)/1e6
        table += [f"| Faithfulness (LLM judge) | {show(mean([j['faithfulness'] for j in judgments]))} | {len(judgments)} judged non-refusals |",
                  f"| Answer relevance (LLM judge) | {show(mean(relevance))} | {len(relevance)}/20 supported cases scored |",
                  f"| End-to-end refusal correctness | {correct}/25 | Errors count as incorrect |",
                  f"| Unsupported answer rate | {show(unsupported_rate)} | {negative_coverage}/5 valid negative runs; errors excluded |",
                  f"| Generation failures | {25-len(valid)}/25 | Includes invalid output/citations |",
                  f"| Judge failures | {sum('error' in r.get('judge', {}) for r in rows)} | See raw trace |",
                  f"| Measured serving / judge cost | ${serving_cost:.6f} / ${judge_cost:.6f} | Configured per-token estimates |"]
    else:
        for metric in ["Faithfulness (LLM judge)", "Answer relevance", "End-to-end refusal correctness",
                       "Unsupported answer / hallucination rate", "Generation latency and cost"]:
            table.append(f"| {metric} | N/A | Groq generation not run |")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = f"Run: {stamp}. Mode: {'live Groq + judge' if live else 'retrieval only; generation NOT measured'}.\n\n" + "\n".join(table)
    report = "# GroundedDesk evaluation\n\n" + summary + f"\n\nCorpus: 10 self-authored PDF pages, {len(parsed.chunks)} chunks; 25 questions (20 supported, 5 unsupported).\n"
    report += (f"\nCorpus SHA-256: `{hashlib.sha256(corpus).hexdigest()}`.\n\n"
        f"Python {platform.python_version()} on {platform.system()} {platform.machine()}. "
        f"Embedding: `{settings.embedding_model}`. Reranker backend: `{settings.reranker_backend}` "
        f"({'same MiniLM model; sentence cosine + lexical coverage' if settings.reranker_backend == 'sentence' else settings.reranker_model}). "
        f"Chunk size/overlap: {settings.chunk_size}/{settings.chunk_overlap} characters. "
        f"Candidate pool: {settings.candidate_k}. Gate score threshold: {settings.rerank_threshold}. "
        f"Indexing including embedding model load: {indexing_ms:.0f} ms.\n\n"
        "This is a small development corpus with lexical overlap, not a held-out accuracy claim. "
        "Compare the ablations before claiming that hybrid retrieval or reranking helps. "
        "See [RUBRIC.md](RUBRIC.md) for metric definitions and limitations.\n")
    (ROOT / "eval/results.md").write_text(report, encoding="utf-8")
    trace = {"timestamp": stamp, "live": live, "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
             "llm_model": settings.llm_model, "reranker_backend": settings.reranker_backend, "rows": rows}
    (run_dir / "latest.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    if args.update_readme:
        readme = ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        text, count = re.subn(r"<!-- EVAL:START -->.*?<!-- EVAL:END -->",
                             "<!-- EVAL:START -->\n" + summary + "\n<!-- EVAL:END -->", text, flags=re.S)
        if count != 1:
            raise RuntimeError("README must contain exactly one evaluation marker pair")
        readme.write_text(text, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
