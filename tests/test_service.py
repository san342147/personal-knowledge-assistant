import json
from types import SimpleNamespace

from app.config import Settings
from app.generation import Completion
from app.models import Chunk, Hit
from app.service import DeskService, validated_citation_ids

ID = "a" * 24
CHUNK = Chunk(ID, "Loans last seven days.", "manual.pdf", 2, "doc")


class Generator:
    available = True
    calls = 0

    def answer(self, question, hits):
        self.calls += 1
        return Completion(json.dumps({"answer": f"Seven days. [{ID}]", "refused": False,
                                     "citations": [ID]}), 100, 20)

    def rewrite(self, question):
        self.calls += 1
        return Completion("equipment loan duration", 10, 5)


def service(hits, generator=None, **settings):
    log = []
    retriever = SimpleNamespace(search=lambda *args: hits,
                               reranker=SimpleNamespace(predict=lambda pairs: [-5] * len(pairs)))
    return DeskService(Settings(_env_file=None, **settings), retriever,
                       generator or Generator(), SimpleNamespace(record=log.append)), log


def test_empty_refuses_without_llm():
    generator = Generator()
    desk, log = service([], generator)
    result = desk.ask("Who is the director?")
    assert result.refused and result.reason == "weak_retrieval"
    assert generator.calls == 0 and len(log) == 1


def test_weak_retrieval_rewrites_once_then_refuses():
    generator = Generator()
    desk, log = service([Hit(CHUNK, -6)], generator)
    result = desk.ask("loan?")
    assert result.refused and generator.calls == 1
    assert len(log[0]["retrieval_attempts"]) == 2
    assert result.tokens_in == 10 and result.rewritten_query


def test_cited_answer_and_cost():
    desk, _ = service([Hit(CHUNK, 5)])
    result = desk.ask("loan length?")
    assert not result.refused and result.citations[0].page == 2
    assert result.citations[0].snippet == CHUNK.text
    assert result.estimated_cost_usd == 0.0000135


def test_citation_list_is_authoritative_when_answer_has_no_inline_id():
    generator = Generator()
    generator.answer = lambda *args: Completion(
        json.dumps({"answer": "Seven days.", "refused": False, "citations": [ID]}), 1, 1)
    desk, _ = service([Hit(CHUNK, 5)], generator)
    result = desk.ask("loan?")
    assert result.reason == "answered" and result.citations[0].chunk_id == ID


def test_bracketed_citation_id_is_normalized_and_deduplicated():
    assert validated_citation_ids([f"[{ID}]", ID], {ID}) == [ID]


def test_empty_or_unknown_citations_fail_closed():
    assert validated_citation_ids([], {ID}) is None
    assert validated_citation_ids(["b" * 24], {ID}) is None


def test_unknown_citation_fails_closed():
    generator = Generator()
    generator.answer = lambda *args: Completion('{"answer":"Invented [wrong]","refused":false,"citations":["wrong"]}', 1, 1)
    desk, _ = service([Hit(CHUNK, 5)], generator)
    assert desk.ask("loan?").reason == "invalid_citations"


def test_missing_key_is_unavailable_not_success():
    generator = Generator()
    generator.available = False
    desk, _ = service([Hit(CHUNK, 5)], generator)
    assert desk.ask("loan?").reason == "generation_unavailable"


def test_provider_failure_is_logged_without_message_leak():
    generator = Generator()
    def fail(*args):
        raise RuntimeError("secret must not be logged")
    generator.answer = fail
    desk, log = service([Hit(CHUNK, 5)], generator)
    result = desk.ask("loan?")
    assert result.reason == "service_error" and len(log) == 1
    assert "secret" not in str(log)
