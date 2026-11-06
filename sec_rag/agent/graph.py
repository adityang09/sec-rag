"""LangGraph agent.

            +--> numeric_lookup --(no facts)--> single_retrieve --+
  route ----+--> single_retrieve ---------------------------------+--> generate --> check_citations
            +--> compare_retrieve (one retrieval per company, in parallel) --+
"""
import contextvars
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .. import llm
from ..config import settings
from . import financial_tool
from .prompts import ANSWER
from .router import route as route_question


class State(TypedDict, total=False):
    question: str
    plan: dict
    facts: list
    derived: list
    evidence: list
    answer: str
    citations: list
    uncited: bool
    timings: dict


def timed(name):
    def deco(fn):
        def wrapper(self, state):
            t = time.perf_counter()
            out = fn(self, state)
            timings = dict(state.get("timings", {}))
            timings[name + "_ms"] = round((time.perf_counter() - t) * 1000, 1)
            for k, v in out.pop("_extra_timings", {}).items():
                timings[k] = round(timings.get(k, 0) + v, 1)
            out["timings"] = timings
            return out
        return wrapper
    return deco


class SecAgent:
    def __init__(self, conn, retriever):
        self.conn = conn
        self.ret = retriever
        rows = conn.execute("SELECT DISTINCT ticker, company FROM filings").fetchall()
        self.universe = {t: c for t, c in rows}
        self.years = {r[0] for r in conn.execute("SELECT DISTINCT fiscal_year FROM filings").fetchall()}
        self.graph = self._build()

    # ---------------- nodes ----------------
    @timed("route")
    def route(self, s):
        return {"plan": route_question(s["question"], self.universe, self.years)}

    @timed("numeric")
    def numeric(self, s):
        p = s["plan"]
        facts = financial_tool.lookup(self.conn, p["tickers"], p["metrics"], p["fiscal_years"])
        return {"facts": facts, "derived": financial_tool.derived(facts)}

    def _filters(self, p, tickers=None):
        return {"tickers": tickers if tickers is not None else p["tickers"],
                "fiscal_years": p["fiscal_years"], "forms": p["forms"]}

    @timed("single_retrieve")
    def single(self, s):
        extra = {}
        hits, _ = self.ret.search_relaxed(s["question"], self._filters(s["plan"]), k=8, timings=extra)
        return {"evidence": hits, "_extra_timings": extra}

    @timed("compare_retrieve")
    def compare(self, s):
        p = s["plan"]
        per = max(3, 10 // len(p["tickers"]))

        def one(t):
            # Rewrite the query per company so each retrieval is focused on one filer
            q = f"{self.universe[t]} {s['question']}"
            tm = {}
            hits, _ = self.ret.search_relaxed(q, self._filters(p, [t]), k=per, timings=tm)
            return hits, tm

        with ThreadPoolExecutor(max_workers=len(p["tickers"])) as ex:
            # copy context so token/cost tracking works inside threads
            futures = [ex.submit(contextvars.copy_context().run, one, t) for t in p["tickers"]]
            results = [f.result() for f in futures]
        hits = [h for r, _ in results for h in r]
        # Per-stage times are summed across the parallel searches (work done, not wall clock).
        # compare_retrieve_ms is the wall-clock time of the whole node.
        extra = {}
        for _, tm in results:
            for k, v in tm.items():
                extra[k] = extra.get(k, 0) + v
        return {"evidence": hits, "_extra_timings": extra}

    @timed("generate")
    def generate(self, s):
        blocks = []
        for i, f in enumerate(s.get("facts", []), 1):
            blocks.append(f"[F{i}] {f.render()}")
        for d in s.get("derived", []):
            blocks.append(f"[computed] {d}")
        for i, c in enumerate(s.get("evidence", []), 1):
            blocks.append(f"[S{i}] ({c['company']} {c['form']} FY{c['fiscal_year']}, {c['section_title'] or 'n/a'})\n"
                          f"{c['text']}")
        if not blocks:
            return {"answer": "I could not find relevant information in the indexed filings."}
        msg = ANSWER.format(question=s["question"], sources="\n\n".join(blocks))
        return {"answer": llm.chat([{"role": "user", "content": msg}], model=settings.answer_model)}

    @timed("check_citations")
    def check(self, s):
        """Drop citation tags that point at nothing, flag answers with no citations at all."""
        n_s, n_f = len(s.get("evidence", [])), len(s.get("facts", []))
        answer = s["answer"]

        def ok(m):
            kind, num = m.group(1), int(m.group(2))
            return m.group(0) if (kind == "S" and num <= n_s) or (kind == "F" and num <= n_f) else ""

        answer = re.sub(r"\[(S|F)(\d+)\]", ok, answer)
        used = sorted(set(re.findall(r"\[(S|F)(\d+)\]", answer)), key=lambda x: (x[0], int(x[1])))
        cites = []
        for kind, num in used:
            if kind == "S":
                c = s["evidence"][int(num) - 1]
                cites.append({"tag": f"S{num}", "ticker": c["ticker"], "form": c["form"],
                              "fiscal_year": c["fiscal_year"], "section": c["section_title"],
                              "chunk_id": c["id"], "url": c["url"]})
            else:
                f = s["facts"][int(num) - 1]
                cites.append({"tag": f"F{num}", "ticker": f.ticker, "metric": f.label,
                              "fiscal_year": f.fiscal_year, "value": f.value, "url": f.source_url})
        return {"answer": answer, "citations": cites,
                "uncited": not cites and "could not find" not in answer.lower()}

    # ---------------- wiring ----------------
    def _build(self):
        g = StateGraph(State)
        g.add_node("route", self.route)
        g.add_node("numeric_lookup", self.numeric)
        g.add_node("single_retrieve", self.single)
        g.add_node("compare_retrieve", self.compare)
        g.add_node("generate", self.generate)
        g.add_node("check_citations", self.check)
        g.set_entry_point("route")
        g.add_conditional_edges("route", lambda s: s["plan"]["route"],
                                {"numeric": "numeric_lookup", "single": "single_retrieve",
                                 "compare": "compare_retrieve"})
        # graceful degradation: no XBRL match -> fall back to text retrieval
        g.add_conditional_edges("numeric_lookup", lambda s: "ok" if s.get("facts") else "fallback",
                                {"ok": "generate", "fallback": "single_retrieve"})
        g.add_edge("single_retrieve", "generate")
        g.add_edge("compare_retrieve", "generate")
        g.add_edge("generate", "check_citations")
        g.add_edge("check_citations", END)
        return g.compile()

    def ask(self, question) -> dict[str, Any]:
        t = time.perf_counter()
        out = self.graph.invoke({"question": question, "timings": {}})
        out["timings"]["total_ms"] = round((time.perf_counter() - t) * 1000, 1)
        return out
