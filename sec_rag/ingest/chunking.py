"""Chunking strategies. All chunks keep char offsets into the filing text, which is what lets the
evaluation compare strategies fairly (see eval/metrics.py).

  section       : split on Item boundaries, pack whole paragraphs to ~800 tokens, 15% overlap,
                  embed with a header "Company (TICKER) 10-K FY2024 | Item 7. MD&A"
  fixed         : naive 800-token sliding window over the whole document, no header
  fixed_header  : same windows, with a document-level header (isolates the effect of the header)
"""
import re
from dataclasses import dataclass

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # offline fallback, close enough for English filings
    _ENC = None


def count_tokens(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    return int(len(re.findall(r"\S+", text)) * 1.3) + 1


@dataclass
class Chunk:
    section: str
    section_title: str
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    n_tokens: int


def _units(text, start, end, target):
    """Yield (s, e) spans: paragraphs, or sentence/line pieces when a paragraph is too long."""
    for m in re.finditer(r"\S(?:.*?\S)??(?=\n\s*\n|\Z)", text[start:end], re.S):
        ps, pe = start + m.start(), start + m.end()
        if count_tokens(text[ps:pe]) <= target:
            yield ps, pe
            continue
        # Split long paragraphs (or big tables) on sentence ends or line breaks
        cur = ps
        for sm in re.finditer(r"(?<=[\.\?\!])\s+|\n", text[ps:pe]):
            cut = ps + sm.end()
            if count_tokens(text[cur:cut]) >= target // 2:
                yield from _hard_split(text, cur, cut, target)
                cur = cut
        if cur < pe:
            yield from _hard_split(text, cur, pe, target)


def _hard_split(text, s, e, target):
    if count_tokens(text[s:e]) <= target:
        yield s, e
        return
    step = target * 4  # ~4 chars per token
    for i in range(s, e, step):
        yield i, min(i + step, e)


def _pack(text, units, target, overlap_tokens):
    chunks, cur = [], []
    cur_tok = 0
    for u in units:
        t = count_tokens(text[u[0]:u[1]])
        if cur and cur_tok + t > target:
            chunks.append((cur[0][0], cur[-1][1]))
            # carry trailing units as overlap, but always make progress
            carry, carry_tok = [], 0
            for cu in reversed(cur[1:]):
                ct = count_tokens(text[cu[0]:cu[1]])
                if carry_tok + ct > overlap_tokens:
                    break
                carry.insert(0, cu)
                carry_tok += ct
            cur, cur_tok = carry, carry_tok
        cur.append(u)
        cur_tok += t
    if cur:
        chunks.append((cur[0][0], cur[-1][1]))
    return chunks


def section_aware_chunks(text, sections, target=800, overlap=0.15, skip=("COVER",)):
    out = []
    for sec in sections:
        if sec.key in skip:
            continue
        units = list(_units(text, sec.start, sec.end, target))
        for s, e in _pack(text, units, target, int(target * overlap)):
            out.append(Chunk(sec.key, sec.title, len(out), s, e, text[s:e], count_tokens(text[s:e])))
    return out


def fixed_chunks(text, target=800, overlap=0.15):
    step = int(target * (1 - overlap))
    out = []
    if _ENC is not None:
        toks = _ENC.encode(text, disallowed_special=())
        _, offsets = _ENC.decode_with_offsets(toks)
        offsets.append(len(text))
        for i in range(0, len(toks), step):
            j = min(i + target, len(toks))
            s, e = offsets[i], offsets[j] if j < len(offsets) else len(text)
            out.append(Chunk("NONE", "", len(out), s, e, text[s:e], j - i))
            if j == len(toks):
                break
    else:
        words = [m.span() for m in re.finditer(r"\S+", text)]
        wt, ws = int(target / 1.3), int(step / 1.3)
        for i in range(0, len(words), ws):
            j = min(i + wt, len(words))
            s, e = words[i][0], words[j - 1][1]
            out.append(Chunk("NONE", "", len(out), s, e, text[s:e], count_tokens(text[s:e])))
            if j == len(words):
                break
    return out


def doc_header(company, ticker, form, fy):
    return f"{company} ({ticker}) {form} fiscal year {fy}"


def embed_text_for(chunk, strategy, company, ticker, form, fy):
    if strategy == "fixed":
        return chunk.text
    head = doc_header(company, ticker, form, fy)
    if strategy == "section":
        head += f" | {chunk.section_title}"
    return head + "\n\n" + chunk.text
