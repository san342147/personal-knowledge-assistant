"""Cheap evidence-sentence reranker. No second model or remote inference required."""
import re
from functools import lru_cache

import numpy as np

STOPWORDS = set("a an the and or of to in on at for from by with is are was were be been "
                "what which who when where how why do does did can could should would must "
                "it its this that these those i we you they as about much many long soon".split())


def content_terms(text):
    return set(re.findall(r"\b\w+\b", text.lower())) - STOPWORDS


class SentenceReranker:
    """Rerank whole chunks by their strongest sentence-level support signal.

    This is a bi-encoder + lexical heuristic, NOT a cross-encoder or entailment model.
    Score = .75 * best sentence cosine + .25 * full-passage query-term coverage.
    Sentence-level comparison reduces dilution by unrelated paragraphs in a chunk.
    The bounded cache stores vectors only in memory and is cleared with the process.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    @lru_cache(maxsize=1024)
    def sentence_vectors(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        # Keep long punctuation-free passages within an embedding-sized window.
        spans = [sentence[start:start+700] for sentence in sentences for start in range(0, len(sentence), 550)]
        return self.embedder.encode(spans or [text], normalize_embeddings=True)

    def predict(self, pairs):
        queries = list(dict.fromkeys(question for question, _ in pairs))
        if not queries:
            return np.array([])
        vectors = self.embedder.encode(queries, normalize_embeddings=True)
        lookup = dict(zip(queries, vectors))
        scores = []
        for question, passage in pairs:
            semantic = float(np.max(self.sentence_vectors(passage) @ lookup[question]))
            terms = content_terms(question)
            coverage = len(terms & content_terms(passage)) / len(terms) if terms else 0
            scores.append(.75 * semantic + .25 * coverage)
        return np.array(scores)
