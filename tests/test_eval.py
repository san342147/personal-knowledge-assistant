from app.models import Chunk, Hit
from eval.run import retrieval_metrics, unsupported_answer_metric


def test_page_match_without_supporting_span_is_not_recall():
    hits = [Hit(Chunk("1", "Unrelated page text", "a.pdf", 1, "doc"))]
    assert retrieval_metrics(hits, [{"filename": "a.pdf", "page": 1, "text": "seven days"}]) == (0, 0)


def test_recall_and_reciprocal_rank_use_actual_evidence():
    hits = [Hit(Chunk("1", "distractor", "a.pdf", 1, "doc")),
            Hit(Chunk("2", "Loan duration is seven\n days.", "a.pdf", 1, "doc"))]
    assert retrieval_metrics(hits, [{"filename": "a.pdf", "page": 1, "text": "seven days"}]) == (1, .5)


def test_failed_generation_cannot_claim_zero_hallucination():
    rows = [{"answerable": False, "answer": {"reason": "service_error", "refused": True}}]
    assert unsupported_answer_metric(rows, {"service_error"}) == (None, 0)


def test_hallucination_proxy_reports_valid_coverage():
    rows = [{"answerable": False, "answer": {"reason": reason, "refused": refused}}
            for reason, refused in [("answered", False), ("model_refusal", True), ("service_error", True)]]
    assert unsupported_answer_metric(rows, {"service_error"}) == (.5, 2)
