"""Numeric lookups over XBRL facts with a constrained, parameterized SQL tool.

Why not free-form text-to-SQL: the LLM only picks a metric NAME from a whitelist plus tickers and
years. The SQL is fixed and parameterized, so there is no injection surface and no hallucinated
column names. The hard part is in the data, handled here:

* The same metric uses different XBRL concepts across companies (revenue alone has 4 common
  tags). Each metric lists candidate concepts in priority order.
* A 10-K for FY2024 also reports FY2023 and FY2022 comparatives, all tagged fy=2024. So the "fy"
  field is the FILING year, not the value's year. We select by period_end and period length
  instead, and take the most recently filed value (restatements win).
* 10-Ks also contain quarterly values. Annual duration metrics must span roughly 1 year.
"""
from dataclasses import dataclass

METRICS = {
    "revenue": {"label": "Revenue", "kind": "duration", "unit": "USD",
                "concepts": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                             "SalesRevenueNet", "RevenuesNetOfInterestExpense"]},
    "net_income": {"label": "Net income", "kind": "duration", "unit": "USD", "concepts": ["NetIncomeLoss"]},
    "operating_income": {"label": "Operating income", "kind": "duration", "unit": "USD",
                         "concepts": ["OperatingIncomeLoss"]},
    "gross_profit": {"label": "Gross profit", "kind": "duration", "unit": "USD", "concepts": ["GrossProfit"]},
    "rd_expense": {"label": "R&D expense", "kind": "duration", "unit": "USD",
                   "concepts": ["ResearchAndDevelopmentExpense"]},
    "eps_diluted": {"label": "Diluted EPS", "kind": "duration", "unit": "USD/shares",
                    "concepts": ["EarningsPerShareDiluted"]},
    "operating_cash_flow": {"label": "Operating cash flow", "kind": "duration", "unit": "USD",
                            "concepts": ["NetCashProvidedByUsedInOperatingActivities"]},
    "capex": {"label": "Capital expenditures", "kind": "duration", "unit": "USD",
              "concepts": ["PaymentsToAcquirePropertyPlantAndEquipment"]},
    "total_assets": {"label": "Total assets", "kind": "instant", "unit": "USD", "concepts": ["Assets"]},
    "total_liabilities": {"label": "Total liabilities", "kind": "instant", "unit": "USD",
                          "concepts": ["Liabilities"]},
    "stockholders_equity": {"label": "Stockholders' equity", "kind": "instant", "unit": "USD",
                            "concepts": ["StockholdersEquity"]},
    "cash": {"label": "Cash and cash equivalents", "kind": "instant", "unit": "USD",
             "concepts": ["CashAndCashEquivalentsAtCarryingValue"]},
    "long_term_debt": {"label": "Long-term debt", "kind": "instant", "unit": "USD",
                       "concepts": ["LongTermDebtNoncurrent", "LongTermDebt"]},
}
ALL_CONCEPTS = sorted({c for m in METRICS.values() for c in m["concepts"]})

ANNUAL_SQL = """
SELECT concept, value, period_start, period_end, accn, cik, filed
FROM financial_facts
WHERE ticker = %(ticker)s AND concept = %(concept)s AND unit = %(unit)s
  AND form LIKE '10-K%%'
  AND EXTRACT(YEAR FROM period_end) = %(fy)s
  AND (%(instant)s OR (period_end - period_start) BETWEEN 340 AND 380)
ORDER BY period_end DESC, filed DESC
LIMIT 1
"""


@dataclass
class Fact:
    ticker: str
    metric: str
    label: str
    fiscal_year: int
    value: float
    unit: str
    concept: str
    period_end: str
    source_url: str

    def render(self):
        v = self.value
        if self.unit == "USD":
            s = f"${v / 1e9:,.2f}B" if abs(v) >= 1e9 else f"${v / 1e6:,.1f}M"
        else:
            s = f"{v:,.2f} {self.unit}"
        return (f"{self.ticker} {self.label} FY{self.fiscal_year} = {s} "
                f"(XBRL {self.concept}, period ending {self.period_end})")


def lookup(conn, tickers, metrics, fiscal_years):
    facts = []
    for t in tickers:
        for m in metrics:
            spec = METRICS.get(m)
            if not spec:
                continue
            for fy in fiscal_years:
                for concept in spec["concepts"]:
                    row = conn.execute(ANNUAL_SQL, {"ticker": t, "concept": concept, "unit": spec["unit"],
                                                    "fy": fy, "instant": spec["kind"] == "instant"}).fetchone()
                    if row:
                        c, val, _, pend, accn, cik, _ = row
                        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn.replace('-', '')}/"
                        facts.append(Fact(t, m, spec["label"], fy, float(val), spec["unit"], c, str(pend), url))
                        break
    return facts


def derived(facts):
    """Compute growth in code, never ask the LLM to do arithmetic."""
    out = []
    by_key = {}
    for f in facts:
        by_key.setdefault((f.ticker, f.metric), []).append(f)
    for (t, m), fs in by_key.items():
        fs = sorted(fs, key=lambda x: x.fiscal_year)
        for a, b in zip(fs, fs[1:]):
            if a.value:
                pct = (b.value - a.value) / abs(a.value) * 100
                out.append(f"{t} {a.label} change FY{a.fiscal_year} to FY{b.fiscal_year}: {pct:+.1f}%")
    return out
