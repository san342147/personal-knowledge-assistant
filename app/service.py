import time
import uuid

from app.generation import Draft
from app.models import Answer, Citation

REFUSAL = "I cannot answer that from the available documents."


def validated_citation_ids(references: list[str], valid_ids: set[str]) -> list[str] | None:
    """Normalize harmless bracket formatting, then fail closed on every other mismatch."""
    if not references:
        return None
    normalized = []
    for reference in references:
        chunk_id = reference.strip()
        if chunk_id.startswith("[") and chunk_id.endswith("]"):
            chunk_id = chunk_id[1:-1].strip()
        if chunk_id not in valid_ids:
            return None
        if chunk_id not in normalized:
            normalized.append(chunk_id)
    return normalized


class DeskService:
    def __init__(self, settings, retriever, generator, telemetry):
        self.settings, self.retriever = settings, retriever
        self.generator, self.telemetry = generator, telemetry

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        started = time.perf_counter()
        result = Answer(request_id=uuid.uuid4().hex, answer=REFUSAL, refused=True, reason="weak_retrieval")
        attempts = []

        def account(completion):
            result.tokens_in += completion.tokens_in
            result.tokens_out += completion.tokens_out

        try:
            hits = self.retriever.search(question, top_k or self.settings.top_k)
            attempts.append([h.chunk.id for h in hits])
            if (hits and hits[0].score < self.settings.rerank_threshold
                    and self.settings.enable_rewrite and self.generator.available):
                rewritten = self.generator.rewrite(question)
                account(rewritten)
                result.rewritten_query = rewritten.text.strip()[:2000]
                if result.rewritten_query:
                    retry = self.retriever.search(result.rewritten_query, top_k or self.settings.top_k)
                    attempts.append([h.chunk.id for h in retry])
                    # Rerank against ORIGINAL intent to avoid accepting a drifting rewrite.
                    if retry:
                        scores = self.retriever.reranker.predict([(question, h.chunk.text) for h in retry])
                        for h, score in zip(retry, scores):
                            h.score = float(score)
                        retry.sort(key=lambda h: h.score, reverse=True)
                        if retry[0].score > hits[0].score:
                            hits = retry
            result.retrieved_chunk_ids = [h.chunk.id for h in hits]
            evidence = [h for h in hits if h.score >= self.settings.rerank_threshold]
            if not evidence:
                return result
            if not self.generator.available:
                result.reason = "generation_unavailable"
                result.answer = "Relevant evidence was found. Add GROQ_API_KEY to .env to generate an answer."
                return result
            completion = self.generator.answer(question, evidence)
            account(completion)
            try:
                draft = Draft.model_validate_json(completion.text)
            except Exception:
                result.reason = "invalid_generation"
                return result
            if draft.refused:
                result.reason = "model_refusal"
                return result
            lookup = {h.chunk.id: h.chunk for h in evidence}
            cited = validated_citation_ids(draft.citations, set(lookup))
            if cited is None:
                result.reason = "invalid_citations"
                return result
            result.answer, result.refused, result.reason = draft.answer, False, "answered"
            result.citations = [Citation(chunk_id=i, filename=lookup[i].filename,
                page=lookup[i].page, snippet=lookup[i].text) for i in cited]
            return result
        except Exception as exc:
            result.reason = "service_error"
            result.answer = f"The request could not complete ({type(exc).__name__}). Check local setup and retry."
            return result
        finally:
            result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            result.estimated_cost_usd = round((result.tokens_in * self.settings.input_usd_per_million
                + result.tokens_out * self.settings.output_usd_per_million) / 1_000_000, 8)
            event = result.model_dump(exclude={"answer", "citations"})
            event.update(endpoint="ask", retrieval_attempts=attempts)
            self.telemetry.record(event)
