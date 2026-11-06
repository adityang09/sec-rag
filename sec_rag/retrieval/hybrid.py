"""Hybrid retrieval: BM25 + dense (pgvector), fused with RRF, reranked by a cross-encoder.

Why hybrid for filings: dense embeddings handle paraphrase ("how did they do on margins") but are
weak on exact tokens that matter here, like "Item 1A", product names, segment names and numbers.
BM25 catches those. Fusion gets both.
"""
import time

import numpy as np

from .. import llm
from ..config import settings
from .fusion import rrf, tokenize, windows


class HybridRetriever:
    def __init__(self, conn, strategy="section", load_reranker=True):
        from rank_bm25 import BM25Okapi
        self.conn = conn
        self.strategy = strategy
        rows = conn.execute(
            """SELECT c.id, c.filing_id, c.ticker, c.fiscal_year, c.form, c.section, c.section_title,
                      c.char_start, c.char_end, c.text, c.embed_text, f.company, f.url
               FROM chunks c JOIN filings f ON f.id = c.filing_id
               WHERE c.strategy = %s ORDER BY c.id""", (strategy,)).fetchall()
        cols = ["id", "filing_id", "ticker", "fiscal_year", "form", "section", "section_title",
                "char_start", "char_end", "text", "embed_text", "company", "url"]
        self.chunks = [dict(zip(cols, r)) for r in rows]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.ids = np.array([c["id"] for c in self.chunks])
        self.cols = {col: np.array([c[col] for c in self.chunks], dtype=object)
                     for col in ("ticker", "fiscal_year", "form", "section")}
        self.bm25 = BM25Okapi([tokenize(c["embed_text"]) for c in self.chunks])
        self.reranker = None
        if load_reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(settings.reranker_model, max_length=512)

    # ---------- individual retrievers ----------
    def _mask(self, f):
        m = np.ones(len(self.chunks), dtype=bool)
        for key, col in (("tickers", "ticker"), ("fiscal_years", "fiscal_year"),
                         ("forms", "form"), ("sections", "section")):
            vals = f.get(key)
            if vals:
                m &= np.isin(self.cols[col], list(vals))
        return m

    def bm25_search(self, query, f, n):
        scores = self.bm25.get_scores(tokenize(query))
        scores[~self._mask(f)] = -np.inf
        top = np.argsort(-scores)[:n]
        return [int(self.ids[i]) for i in top if np.isfinite(scores[i]) and scores[i] > 0]

    def dense_search(self, query, f, n, qvec=None):
        qvec = qvec if qvec is not None else np.array(llm.embed([query])[0])
        where, params = ["strategy = %s"], [self.strategy]
        for key, col in (("tickers", "ticker"), ("fiscal_years", "fiscal_year"),
                         ("forms", "form"), ("sections", "section")):
            if f.get(key):
                where.append(f"{col} = ANY(%s)")
                params.append(list(f[key]))
        # Filtered HNSW can return fewer than n rows. The connection is tuned for that once, when
        # it is opened (see db.tune_vector_search), instead of on every query.
        sql = f"SELECT id FROM chunks WHERE {' AND '.join(where)} ORDER BY embedding <=> %s LIMIT %s"
        return [r[0] for r in self.conn.execute(sql, (*params, qvec, n)).fetchall()]

    def rerank(self, query, ids, k):
        if not self.reranker or not ids:
            return ids[:k]
        pairs, owner = [], []
        for cid in ids:
            c = self.by_id[cid]
            head = c["embed_text"][: len(c["embed_text"]) - len(c["text"])]
            for w in windows(c["text"]):
                pairs.append((query, head + w))
                owner.append(cid)
        scores = self.reranker.predict(pairs, batch_size=32)
        best = {}
        for cid, s in zip(owner, scores):  # max over windows
            best[cid] = max(best.get(cid, -1e9), float(s))
        return sorted(ids, key=lambda i: -best[i])[:k]

    # ---------- main entry ----------
    def search(self, query, filters=None, k=None, mode="hybrid", use_rerank=True, timings=None):
        if mode not in ("hybrid", "bm25", "dense"):
            raise ValueError(f"unknown retrieval mode: {mode}")
        f = filters or {}
        k = k or settings.top_k
        n = settings.n_candidates
        t0 = time.perf_counter()
        ranked = []
        if mode in ("hybrid", "bm25"):
            ranked.append(self.bm25_search(query, f, n))
        if mode in ("hybrid", "dense"):
            ranked.append(self.dense_search(query, f, n))
        fused = rrf(ranked, settings.rrf_k) if len(ranked) > 1 else ranked[0]
        t1 = time.perf_counter()
        out = self.rerank(query, fused[: settings.n_rerank], k) if use_rerank else fused[:k]
        t2 = time.perf_counter()
        if timings is not None:
            timings["retrieve_ms"] = timings.get("retrieve_ms", 0) + (t1 - t0) * 1000
            timings["rerank_ms"] = timings.get("rerank_ms", 0) + (t2 - t1) * 1000
        return [self.by_id[i] for i in out]

    def search_relaxed(self, query, filters, k=None, **kw):
        """If the router's filters are too strict (wrong year, missing 10-Q), relax step by step."""
        f = dict(filters or {})
        for drop in (None, "sections", "forms", "fiscal_years"):
            if drop:
                f.pop(drop, None)
            res = self.search(query, f, k, **kw)
            if len(res) >= 3:
                return res, f
        return res, f
