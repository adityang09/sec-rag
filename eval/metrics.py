"""Strategy-independent retrieval metrics.

Gold evidence is stored as a CHARACTER SPAN in the filing text, not as a chunk id. Chunk ids differ
between chunking strategies, so id-based gold would make the comparison meaningless. A retrieved
chunk is a hit for a gold span if it covers at least half of the span (or half of the chunk, if
the span is longer than the chunk).
"""


def overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def is_hit(chunk, span, min_cover=0.5):
    if chunk["filing_id"] != span["filing_id"]:
        return False
    ov = overlap(chunk["char_start"], chunk["char_end"], span["start"], span["end"])
    denom = min(span["end"] - span["start"], chunk["char_end"] - chunk["char_start"])
    return denom > 0 and ov / denom >= min_cover


def recall_at_k(retrieved, gold_spans, k=10):
    if not gold_spans:
        return None
    top = retrieved[:k]
    found = sum(any(is_hit(c, g) for c in top) for g in gold_spans)
    return found / len(gold_spans)


def mrr(retrieved, gold_spans, k=10):
    for rank, c in enumerate(retrieved[:k], 1):
        if any(is_hit(c, g) for g in gold_spans):
            return 1.0 / rank
    return 0.0


def locate(text, quote):
    """Find a quote in the filing text, tolerant of whitespace differences. Returns (start, end) or None."""
    i = text.find(quote)
    if i >= 0:
        return i, i + len(quote)
    import re
    words = quote.split()
    if len(words) < 4:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pattern, text)
    return (m.start(), m.end()) if m else None
