"""End-to-end answer quality: route accuracy, faithfulness, citation accuracy, numeric exactness,
plus p50/p95 latency and cost per query.

  python -m eval.answer_eval --gold eval/gold.jsonl

Faithfulness  = share of atomic claims in the answer supported by the retrieved sources
Citation acc. = share of [S#]/[F#] citations whose cited source actually supports the sentence
Numeric exact = for numeric questions, the XBRL tool returned the expected value (within 0.5%)
Use a judge model that is different from, or at least as strong as, the answer model.
"""
import argparse
import json

import numpy as np

from sec_rag import llm
from sec_rag.agent.graph import SecAgent
from sec_rag.config import settings
from sec_rag.db import connect
from sec_rag.retrieval.hybrid import HybridRetriever

JUDGE = """You are grading an answer produced from SEC filing excerpts.

SOURCES:
{sources}

ANSWER:
{answer}

Tasks:
1. Split the answer into atomic factual claims. For each, decide if the SOURCES support it.
2. For each citation tag in the answer (like [S2] or [F1]), decide if THAT source supports the
   sentence it is attached to.
Return JSON: {{"claims": [{{"claim": str, "supported": bool}}],
               "citations": [{{"tag": str, "supported": bool}}]}}"""


def sources_text(out):
    parts = [f"[F{i}] {f.render()}" for i, f in enumerate(out.get("facts", []), 1)]
    parts += [f"[S{i}] {c['text']}" for i, c in enumerate(out.get("evidence", []), 1)]
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/gold.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="eval/answer_results.jsonl")
    args = ap.parse_args()

    gold = [json.loads(l) for l in open(args.gold) if json.loads(l).get("reviewed")]
    if args.limit:
        gold = gold[: args.limit]
    conn = connect()
    agent = SecAgent(conn, HybridRetriever(conn, "section"))
    rows = []
    with open(args.out, "w") as f:
        for g in gold:
            usage = llm.start_tracking()
            out = agent.ask(g["question"])
            cost, lat = usage.cost_usd, out["timings"]["total_ms"]
            j = llm.chat([{"role": "user", "content": JUDGE.format(sources=sources_text(out), answer=out["answer"])}],
                         model=settings.judge_model, json_mode=True, max_tokens=1500)
            claims, cites = j.get("claims", []), j.get("citations", [])
            row = {"id": g["id"], "type": g["type"], "route": out["plan"]["route"],
                   "route_ok": out["plan"]["route"] == g["type"],
                   "faithfulness": np.mean([c["supported"] for c in claims]) if claims else None,
                   "citation_acc": np.mean([c["supported"] for c in cites]) if cites else None,
                   "uncited": out.get("uncited", False), "latency_ms": lat, "cost_usd": cost}
            if g["type"] == "numeric":
                vals = [x.value for x in out.get("facts", [])]
                row["numeric_exact"] = any(abs(v - g["expected_value"]) <= 0.005 * abs(g["expected_value"]) for v in vals)
            rows.append(row)
            f.write(json.dumps({**row, "question": g["question"], "answer": out["answer"]}, default=float) + "\n")
            print(g["id"], row["route"], row["faithfulness"], row["citation_acc"], f"{lat:.0f}ms")

    def avg(key, subset=rows):
        v = [r[key] for r in subset if r.get(key) is not None]
        return round(float(np.mean(v)), 3) if v else None

    lats = [r["latency_ms"] for r in rows]
    print("\n=== summary ===")
    print("route accuracy     ", avg("route_ok"))
    print("faithfulness       ", avg("faithfulness"))
    print("citation accuracy  ", avg("citation_acc"))
    print("uncited answers    ", avg("uncited"))
    print("numeric exact      ", avg("numeric_exact", [r for r in rows if r["type"] == "numeric"]))
    print("latency p50 / p95  ", round(np.percentile(lats, 50)), "/", round(np.percentile(lats, 95)), "ms")
    print("avg cost per query  $", round(float(np.mean([r["cost_usd"] for r in rows])), 4))
    for t in ("single", "compare", "numeric"):
        sub = [r for r in rows if r["type"] == t]
        if sub:
            print(f"  {t:8s} n={len(sub)} route_acc={avg('route_ok', sub)} faith={avg('faithfulness', sub)}")


if __name__ == "__main__":
    main()
