"""Pure functions, unit tested without any services."""
import re

STOP = set("""a an the of and or to in on for by with as at from is are was were be been this that
these those it its which what how did does do their our we you they he she his her them than then
""".split())


def tokenize(text):
    # Keep numbers and tickers. "10-K" -> ["10", "k"], "$391.0" -> ["391", "0"]
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def rrf(rankings, k=60):
    """Reciprocal rank fusion. rankings: list of ranked id lists. Returns ids sorted by fused score.

    RRF ignores raw scores, which is the point: BM25 scores and cosine similarities live on
    different scales and do not need calibration this way.
    """
    scores = {}
    for ranked in rankings:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]


def windows(text, size=300, stride=250):
    """Word windows so a 512-token cross-encoder can still see the whole of an 800-token chunk."""
    words = text.split()
    if len(words) <= size:
        return [text]
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - size + stride), stride)]
