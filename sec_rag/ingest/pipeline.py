"""Ingestion CLI.

python -m sec_rag.ingest.pipeline --tickers AAPL MSFT NVDA JPM BK GS AMZN XOM JNJ WMT \
    --years 2022 2023 2024 --forms 10-K 10-Q --strategies section fixed fixed_header
"""
import argparse
from datetime import date

from .. import llm
from ..config import settings
from ..db import connect, init_db
from . import edgar
from .chunking import embed_text_for, fixed_chunks, section_aware_chunks
from .parse import html_to_text, split_sections
from ..agent.financial_tool import ALL_CONCEPTS


def ingest_filing(conn, ticker, cik, company, f, strategies):
    row = conn.execute("SELECT id FROM filings WHERE accession=%s", (f["accession"],)).fetchone()
    if row:
        filing_id = row[0]
        text = conn.execute("SELECT text FROM filings WHERE id=%s", (filing_id,)).fetchone()[0]
    else:
        text = html_to_text(edgar.fetch_document(f["url"]))
        filing_id = conn.execute(
            """INSERT INTO filings (ticker, cik, company, form, fiscal_year, period_end, filed, accession, url, text)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (ticker, cik, company, f["form"], f["fiscal_year"], f["period_end"], f["filed"],
             f["accession"], f["url"], text)).fetchone()[0]

    sections = split_sections(text, f["form"])
    for strategy in strategies:
        done = conn.execute("SELECT 1 FROM chunks WHERE filing_id=%s AND strategy=%s LIMIT 1",
                            (filing_id, strategy)).fetchone()
        if done:
            continue
        if strategy == "section":
            chunks = section_aware_chunks(text, sections, settings.chunk_target_tokens, settings.chunk_overlap)
        else:
            chunks = fixed_chunks(text, settings.chunk_target_tokens, settings.chunk_overlap)
        embed_texts = [embed_text_for(c, strategy, company, ticker, f["form"], f["fiscal_year"]) for c in chunks]
        vectors = llm.embed(embed_texts)
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (filing_id, strategy, ticker, form, fiscal_year, section, section_title,
                       chunk_index, char_start, char_end, n_tokens, text, embed_text, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                [(filing_id, strategy, ticker, f["form"], f["fiscal_year"], c.section, c.section_title,
                  c.chunk_index, c.char_start, c.char_end, c.n_tokens, c.text, et, v)
                 for c, et, v in zip(chunks, embed_texts, vectors)])
        print(f"  {ticker} {f['form']} FY{f['fiscal_year']} {strategy}: {len(chunks)} chunks")


def ingest_facts(conn, ticker, cik):
    facts = edgar.company_facts(cik).get("facts", {}).get("us-gaap", {})
    rows = []
    for concept in ALL_CONCEPTS:
        for unit, vals in facts.get(concept, {}).get("units", {}).items():
            for v in vals:
                if v.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                    continue
                rows.append((ticker, cik, concept, unit, v["val"], v.get("start"), v["end"],
                             v.get("fy"), v.get("fp"), v.get("form"), v.get("filed"), v["accn"]))
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO financial_facts (ticker, cik, concept, unit, value, period_start, period_end,
                   fy, fp, form, filed, accn)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""", rows)
    print(f"  {ticker}: {len(rows)} XBRL facts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[date.today().year - 2, date.today().year - 1])
    ap.add_argument("--forms", nargs="+", default=["10-K"])
    ap.add_argument("--max-10q-per-year", type=int, default=1)
    ap.add_argument("--strategies", nargs="+", default=["section", "fixed", "fixed_header"])
    args = ap.parse_args()

    init_db()
    conn = connect()
    usage = llm.start_tracking()
    tmap = edgar.ticker_map()
    for t in args.tickers:
        t = t.upper()
        cik, company = tmap[t]
        print(f"{t} -> CIK {cik} {company}")
        filings = edgar.list_filings(cik, forms=tuple(args.forms), fiscal_years=set(args.years))
        seen_q = {}
        for f in filings:
            if f["form"] == "10-Q":
                seen_q[f["fiscal_year"]] = seen_q.get(f["fiscal_year"], 0) + 1
                if seen_q[f["fiscal_year"]] > args.max_10q_per_year:
                    continue
            ingest_filing(conn, t, cik, company, f, args.strategies)
        ingest_facts(conn, t, cik)
    print(f"embedding tokens: {usage.embedding_tokens:,}  cost: ${usage.cost_usd:.2f}")


if __name__ == "__main__":
    main()
