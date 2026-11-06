"""SEC EDGAR client: ticker lookup, filing lists, documents, XBRL company facts.

SEC rules: send a descriptive User-Agent with an email, stay under 10 requests/sec.
Everything downloaded is cached under data/raw so re-runs are free. Filing documents never change,
so they are cached forever. Index endpoints (ticker map, submissions, company facts) change as new
filings arrive, so their cache expires after INDEX_MAX_AGE seconds.
"""
import hashlib
import json
import time
from pathlib import Path

import requests

from ..config import settings

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

INDEX_MAX_AGE = 24 * 3600

_session = None
_last = 0.0


def _get(url, as_json=False, max_age=None):
    global _session, _last
    key = hashlib.sha1(url.encode()).hexdigest()
    cached = RAW / (key + (".json" if as_json else ".html"))
    fresh = cached.exists() and (max_age is None or time.time() - cached.stat().st_mtime < max_age)
    if fresh:
        text = cached.read_text(encoding="utf-8", errors="ignore")
        return json.loads(text) if as_json else text
    if not settings.sec_user_agent:
        raise RuntimeError("Set SEC_USER_AGENT in .env (name + email), SEC blocks anonymous clients")
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": settings.sec_user_agent,
                                 "Accept-Encoding": "gzip, deflate"})
    wait = 0.12 - (time.time() - _last)
    if wait > 0:
        time.sleep(wait)
    r = _session.get(url, timeout=60)
    _last = time.time()
    r.raise_for_status()
    cached.write_text(r.text, encoding="utf-8")
    return r.json() if as_json else r.text


def ticker_map():
    data = _get("https://www.sec.gov/files/company_tickers.json", as_json=True, max_age=INDEX_MAX_AGE)
    return {row["ticker"].upper(): (str(row["cik_str"]).zfill(10), row["title"]) for row in data.values()}


def _filing_tables(sub, min_year=None):
    """Yield the columnar filing tables, newest first.

    The submissions API only inlines the ~1000 most recent filings under "recent". Heavy filers
    (banks issuing thousands of 424B2 prospectuses a year) push older 10-Ks out of that window,
    so the rest are paged in from the extra files listed under "files".
    """
    yield sub["filings"]["recent"]
    for f in sub["filings"].get("files", []):
        if min_year and f.get("filingTo", "9999")[:4] < str(min_year):
            continue  # this page only holds filings older than anything we want
        # Older pages are historical and do not change, so they are cached forever
        yield _get(f"https://data.sec.gov/submissions/{f['name']}", as_json=True)


def list_filings(cik, forms=("10-K",), fiscal_years=None, max_per_form=None):
    """Return filings from the submissions API, newest first.

    fiscal_year is taken as the calendar year of the report period end. That matches how most
    companies label their fiscal year (Apple FY2024 ends Sep 2024, Walmart FY2025 ends Jan 2025).
    """
    sub = _get(f"https://data.sec.gov/submissions/CIK{cik}.json", as_json=True, max_age=INDEX_MAX_AGE)
    rows, counts, seen = [], {}, set()
    for table in _filing_tables(sub, min(fiscal_years) if fiscal_years else None):
        for i, form in enumerate(table["form"]):
            if form not in forms or not table["reportDate"][i]:
                continue
            acc = table["accessionNumber"][i]
            if acc in seen:
                continue
            fy = int(table["reportDate"][i][:4])
            if fiscal_years and fy not in fiscal_years:
                continue
            if max_per_form and counts.get(form, 0) >= max_per_form:
                continue
            seen.add(acc)
            counts[form] = counts.get(form, 0) + 1
            url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                   f"{acc.replace('-', '')}/{table['primaryDocument'][i]}")
            rows.append({"form": form, "accession": acc, "filed": table["filingDate"][i],
                         "period_end": table["reportDate"][i], "fiscal_year": fy, "url": url})
    return rows


def fetch_document(url):
    return _get(url)


def company_facts(cik):
    return _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", as_json=True,
                max_age=INDEX_MAX_AGE)
