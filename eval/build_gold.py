"""Build the gold set: 120 single-company, 50 comparison, 30 numeric questions.

  python -m eval.build_gold generate --out eval/gold_draft.jsonl
  python -m eval.build_gold review   --inp eval/gold_draft.jsonl --out eval/gold.jsonl

Fairness detail: passages are sampled as random windows of the FILING TEXT, not from any
chunking strategy's chunks. Sampling from section chunks would quietly favor that strategy.
The LLM drafts and you review every item. Only reviewed items are used in evaluation.
"""
import argparse
import json
import random
import re

from sec_rag import llm
from sec_rag.agent.financial_tool import METRICS, lookup
from sec_rag.config import settings
from sec_rag.db import connect
from sec_rag.ingest.parse import split_sections

from .metrics import locate

SINGLE_PROMPT = """Below is a passage from {company}'s {form} for fiscal year {fy}.
Write ONE question a financial analyst would realistically ask that this passage answers.
The question must name the company and fiscal year, and must NOT mention "the passage".
Avoid trivia (addresses, page numbers, exhibit lists). If the passage is boilerplate, a table of
contents, or not useful, return {{"skip": true}}.
Return JSON: {{"question": ..., "answer": short answer, "quote": an EXACT verbatim sentence or two
copied from the passage that supports the answer}}

Passage:
{passage}"""

COMPARE_PROMPT = """Below are passages from two companies' filings for the same fiscal year.
Write ONE question comparing the two companies that needs BOTH passages to answer. Name both
companies and the fiscal year. If no meaningful comparison exists, return {{"skip": true}}.
Return JSON: {{"question": ..., "answer": short answer, "quote_a": exact verbatim text from A,
"quote_b": exact verbatim text from B}}

A ({a_company}, {form} FY{fy}):
{a}

B ({b_company}, {form} FY{fy}):
{b}"""


def useful(window):
    alpha = sum(ch.isalpha() for ch in window) / max(1, len(window))
    return alpha > 0.6 and window.count("|") < 20 and len(window.split()) > 120


def sample_window(text, form, rng, words=320, section=None):
    secs = [s for s in split_sections(text, form) if s.key not in ("COVER", "15", "II-6")]
    if section:
        secs = [s for s in secs if s.key == section] or secs
    for _ in range(20):
        s = rng.choice(secs)
        seg = text[s.start:s.end]
        ws = [m.start() for m in re.finditer(r"\S+", seg)]
        if len(ws) < words + 10:
            continue
        i = rng.randrange(0, len(ws) - words)
        a, b = s.start + ws[i], s.start + ws[i + words]
        if useful(text[a:b]):
            return a, b, s.key
    return None


def generate(args):
    rng = random.Random(7)
    conn = connect()
    filings = conn.execute("SELECT id, ticker, company, form, fiscal_year, text FROM filings").fetchall()
    out = open(args.out, "w")
    n = {"single": 0, "compare": 0, "numeric": 0}

    while n["single"] < args.n_single:
        fid, t, comp, form, fy, text = rng.choice(filings)
        w = sample_window(text, form, rng)
        if not w:
            continue
        r = llm.chat([{"role": "user", "content": SINGLE_PROMPT.format(
            company=comp, form=form, fy=fy, passage=text[w[0]:w[1]])}],
            model=settings.answer_model, json_mode=True)
        if r.get("skip"):
            continue
        loc = locate(text, r.get("quote", ""))
        if not loc:
            continue
        n["single"] += 1
        out.write(json.dumps({"id": f"s{n['single']}", "type": "single", "question": r["question"],
                              "answer": r["answer"], "tickers": [t], "fiscal_years": [fy],
                              "evidence": [{"filing_id": fid, "start": loc[0], "end": loc[1],
                                            "quote": r["quote"]}], "reviewed": False}) + "\n")

    tenks = [f for f in filings if f[3] == "10-K"]
    while n["compare"] < args.n_compare:
        a = rng.choice(tenks)
        peers = [f for f in tenks if f[4] == a[4] and f[1] != a[1]]
        if not peers:
            continue
        b = rng.choice(peers)
        sec = rng.choice(["1", "1A", "7", "7A"])
        wa, wb = sample_window(a[5], "10-K", rng, section=sec), sample_window(b[5], "10-K", rng, section=sec)
        if not wa or not wb:
            continue
        r = llm.chat([{"role": "user", "content": COMPARE_PROMPT.format(
            a_company=a[2], b_company=b[2], form="10-K", fy=a[4], a=a[5][wa[0]:wa[1]], b=b[5][wb[0]:wb[1]])}],
            model=settings.answer_model, json_mode=True)
        if r.get("skip"):
            continue
        la, lb = locate(a[5], r.get("quote_a", "")), locate(b[5], r.get("quote_b", ""))
        if not la or not lb:
            continue
        n["compare"] += 1
        out.write(json.dumps({"id": f"c{n['compare']}", "type": "compare", "question": r["question"],
                              "answer": r["answer"], "tickers": [a[1], b[1]], "fiscal_years": [a[4]],
                              "evidence": [{"filing_id": a[0], "start": la[0], "end": la[1], "quote": r["quote_a"]},
                                           {"filing_id": b[0], "start": lb[0], "end": lb[1], "quote": r["quote_b"]}],
                              "reviewed": False}) + "\n")

    combos = sorted({(f[1], f[2], f[4]) for f in tenks})
    tries = 0
    while n["numeric"] < args.n_numeric and tries < 500:
        tries += 1
        t, comp, fy = rng.choice(combos)
        m = rng.choice(list(METRICS))
        facts = lookup(conn, [t], [m], [fy])
        if not facts:
            continue
        n["numeric"] += 1
        f = facts[0]
        out.write(json.dumps({"id": f"n{n['numeric']}", "type": "numeric",
                              "question": f"What was {comp}'s {METRICS[m]['label'].lower()} in fiscal year {fy}?",
                              "answer": f.render(), "expected_value": f.value, "metric": m,
                              "tickers": [t], "fiscal_years": [fy], "evidence": [], "reviewed": False}) + "\n")
    out.close()
    print(n, "->", args.out)


def review(args):
    items = [json.loads(l) for l in open(args.inp)]
    kept = []
    for i, it in enumerate(items, 1):
        print(f"\n[{i}/{len(items)}] {it['type']}  {it['tickers']} FY{it['fiscal_years']}")
        print("Q:", it["question"])
        print("A:", it["answer"])
        for e in it["evidence"]:
            print("  evidence:", e["quote"][:300])
        c = input("(k)eep  (e)dit question  (s)kip  (q)uit > ").strip().lower()
        if c == "q":
            break
        if c == "e":
            it["question"] = input("new question: ").strip() or it["question"]
            c = "k"
        if c == "k":
            it["reviewed"] = True
            kept.append(it)
    with open(args.out, "w") as f:
        for it in kept:
            f.write(json.dumps(it) + "\n")
    print(f"kept {len(kept)} -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", default="eval/gold_draft.jsonl")
    g.add_argument("--n-single", type=int, default=140)
    g.add_argument("--n-compare", type=int, default=60)
    g.add_argument("--n-numeric", type=int, default=35)
    r = sub.add_parser("review")
    r.add_argument("--inp", default="eval/gold_draft.jsonl")
    r.add_argument("--out", default="eval/gold.jsonl")
    a = ap.parse_args()
    generate(a) if a.cmd == "generate" else review(a)
