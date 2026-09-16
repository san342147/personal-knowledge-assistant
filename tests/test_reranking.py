import numpy as np

from app.reranking import SentenceReranker, content_terms


class Embedder:
    calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return np.array([[1., 0.] if "loan" in text.lower() else [0., 1.] for text in texts])


def test_supporting_sentence_beats_unrelated_passage_and_caches():
    embedder = Embedder()
    reranker = SentenceReranker(embedder)
    pairs = [("loan duration", "Loan duration is seven days. Samples stay cold."),
             ("loan duration", "Samples stay cold.")]
    scores = reranker.predict(pairs)
    assert scores[0] == 1 and scores[1] == 0
    reranker.predict(pairs)
    assert embedder.calls == 4  # two query batches, two unique passage batches


def test_stopwords_do_not_supply_keyword_evidence():
    assert content_terms("What is the loan period?") == {"loan", "period"}
