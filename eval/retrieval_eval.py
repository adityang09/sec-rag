"""Ablation grid: chunking strategy x retrieval mode x reranker, measured by recall@10 and MRR.

  python -m eval.retrieval_eval --gold eval/gold.jsonl

Filters use the gold tickers and years ("oracle filters") so this measures retrieval alone, not
the router. Pass --no-filters to see how retrieval does across the whole corpus.
Comparison questions retrieve 5 chunks per company, the same way the agent does.
"""
import argparse
import json

import numpy as np

from sec_rag import llm
from sec_rag.db import connect
from sec_rag.retrieval.hybrid import HybridRetriever

from .metrics import mrr, recall_at_k

GRID = [
    ("fixed", "dense", False),
    ("fixed", "hybrid", False),
    ("fixed_header", "hybrid", False),
    ("section", "dense", False),
    ("section", "bm25", False),
    ("section", "hybrid", False),
    ("section", "hybrid", True),
    ("fixed", "hybrid", True),
]


def run(ret, item, mode, rerank, use_filters, k=10):
    if item["type"] == "compare":
        hits = []
        for t in item["tickers"]:
            f = {"tickers": [t], "fiscal_years": item["fiscal_years"]} if use_filters else {}
            hits += ret.search(item["question"], f, k=k // 2, mode=mode, use_rerank=rerank)
        return hits
    f = {"tickers": item["tickers"], "fiscal_years": item["fiscal_years"]} if use_filters else {}
    return ret.search(item["question"], f, k=k, mode=mode, use_rerank=rerank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/gold.jsonl")
    ap.add_argument("--no-filters", action="store_true")
    ap.add_argument("--out", default="eval/retrieval_results.json")
    args = ap.parse_args()

    gold = [json.loads(l) for l in open(args.gold)]
    gold = [g for g in gold if g.get("reviewed") and g["evidence"]]
    conn = connect()
    usage = llm.start_tracking()
    results, cache = [], {}
    for strategy, mode, rerank in GRID:
        if strategy not in cache:
            cache = {strategy: HybridRetriever(conn, strategy, load_reranker=True)}
        ret = cache[strategy]
        rec, rr, by_type = [], [], {}
        for g in gold:
            hits = run(ret, g, mode, rerank, not args.no_filters)
            r = recall_at_k(hits, g["evidence"], 10)
            rec.append(r)
            rr.append(mrr(hits, g["evidence"], 10))
            by_type.setdefault(g["type"], []).append(r)
        row = {"strategy": strategy, "mode": mode, "rerank": rerank, "n": len(gold),
               "recall@10": round(float(np.mean(rec)), 3), "mrr@10": round(float(np.mean(rr)), 3),
               **{f"recall@10_{t}": round(float(np.mean(v)), 3) for t, v in by_type.items()}}
        results.append(row)
        print(row)

    print("\n| strategy | mode | rerank | recall@10 | MRR@10 |\n|---|---|---|---|---|")
    for r in results:
        print(f"| {r['strategy']} | {r['mode']} | {r['rerank']} | {r['recall@10']} | {r['mrr@10']} |")
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nquery embedding cost: ${usage.cost_usd:.3f}")


if __name__ == "__main__":
    main()
