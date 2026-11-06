"""Offline tests for the parts that decide retrieval quality. Run: python -m unittest -v"""
import random
import unittest

from eval.metrics import is_hit, locate, recall_at_k
from sec_rag.ingest.chunking import count_tokens, fixed_chunks, section_aware_chunks
from sec_rag.ingest.parse import html_to_text, split_sections
from sec_rag.retrieval.fusion import rrf, tokenize, windows

rng = random.Random(0)
WORDS = "revenue growth margin customers products services demand supply cloud data risk market".split()


def para(n=60):
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def fake_10k():
    toc = ("<table><tr><td>Item 1.</td><td>Business</td><td>3</td></tr>"
           "<tr><td>Item 1A.</td><td>Risk Factors</td><td>9</td></tr>"
           "<tr><td>Item 7.</td><td>MD&amp;A</td><td>30</td></tr></table>")
    body = "<p>PART I</p>"
    body += "<p>Item 1. Business</p>" + "".join(f"<p>{para()}</p>" for _ in range(30))
    body += "<p>Item 1A. Risk Factors</p>" + "".join(f"<p>{para()}</p>" for _ in range(40))
    body += "<p>PART II</p><p>Item 7. Management's Discussion and Analysis</p>"
    body += "".join(f"<p>{para()}</p>" for _ in range(25))
    body += ("<table><tr><td>Revenue</td><td>$</td><td>391,035</td><td>$</td><td>383,285</td></tr>"
             "<tr><td>Net income</td><td>$</td><td>93,736</td><td>$</td><td>96,995</td></tr></table>")
    body += "".join(f"<p>{para()}</p>" for _ in range(5))
    return f"<html><body><div style='display:none'>hidden xbrl</div>{toc}{body}</body></html>"


class TestParse(unittest.TestCase):
    def setUp(self):
        self.text = html_to_text(fake_10k())
        self.secs = split_sections(self.text, "10-K")

    def test_hidden_removed_and_tables_flattened(self):
        self.assertNotIn("hidden xbrl", self.text)
        self.assertIn("Revenue | 391,035 | 383,285", self.text)

    def test_toc_is_skipped(self):
        keys = [s.key for s in self.secs]
        self.assertEqual(keys, ["COVER", "1", "1A", "7"])
        risk = next(s for s in self.secs if s.key == "1A")
        self.assertGreater(risk.end - risk.start, 5000)  # real section, not the TOC line

    def test_sections_do_not_overlap(self):
        for a, b in zip(self.secs, self.secs[1:]):
            self.assertLessEqual(a.end, b.start)


class TestChunking(unittest.TestCase):
    def setUp(self):
        self.text = html_to_text(fake_10k())
        self.secs = split_sections(self.text, "10-K")

    def test_section_chunks_stay_inside_sections(self):
        chunks = section_aware_chunks(self.text, self.secs, target=300, overlap=0.15)
        self.assertGreater(len(chunks), 5)
        bounds = {s.key: s for s in self.secs}
        for c in chunks:
            s = bounds[c.section]
            self.assertTrue(s.start <= c.char_start < c.char_end <= s.end)
            self.assertEqual(self.text[c.char_start:c.char_end], c.text)
            self.assertLessEqual(c.n_tokens, 300 * 1.1)

    def test_overlap_and_progress(self):
        chunks = [c for c in section_aware_chunks(self.text, self.secs, target=300, overlap=0.15) if c.section == "1A"]
        for a, b in zip(chunks, chunks[1:]):
            self.assertGreater(b.char_start, a.char_start)       # always moves forward
            self.assertLess(b.char_start, a.char_end)            # paragraphs carried as overlap

    def test_table_not_split_mid_row(self):
        chunks = section_aware_chunks(self.text, self.secs, target=300)
        self.assertTrue(any("Revenue | 391,035 | 383,285\nNet income | 93,736 | 96,995" in c.text for c in chunks))

    def test_fixed_chunks_cover_document(self):
        chunks = fixed_chunks(self.text, target=300, overlap=0.15)
        self.assertEqual(chunks[0].char_start, 0)
        self.assertGreaterEqual(chunks[-1].char_end, len(self.text.rstrip()) - 1)


class TestRetrievalUtils(unittest.TestCase):
    def test_rrf_rewards_agreement(self):
        fused = rrf([[1, 2, 3], [3, 1, 4]])
        self.assertEqual(fused[0], 1)  # ranked 1st and 2nd beats 3rd and 1st
        self.assertEqual(set(fused), {1, 2, 3, 4})

    def test_tokenize_keeps_numbers(self):
        self.assertIn("1a", tokenize("Item 1A Risk Factors"))
        self.assertIn("391", tokenize("$391 billion"))

    def test_windows_cover_long_text(self):
        text = " ".join(str(i) for i in range(700))
        ws = windows(text, size=300, stride=250)
        self.assertTrue(ws[-1].endswith("699"))


class TestMetrics(unittest.TestCase):
    def test_hit_and_recall(self):
        gold = [{"filing_id": 1, "start": 100, "end": 200}]
        good = {"filing_id": 1, "char_start": 50, "char_end": 180}   # covers 80%
        weak = {"filing_id": 1, "char_start": 170, "char_end": 900}  # covers 30%
        other = {"filing_id": 2, "char_start": 100, "char_end": 200}
        self.assertTrue(is_hit(good, gold[0]))
        self.assertFalse(is_hit(weak, gold[0]))
        self.assertFalse(is_hit(other, gold[0]))
        self.assertEqual(recall_at_k([weak, other, good], gold, k=2), 0.0)
        self.assertEqual(recall_at_k([weak, other, good], gold, k=3), 1.0)

    def test_locate_whitespace_tolerant(self):
        text = "Net sales increased\n  due to higher iPhone   revenue in Greater China."
        self.assertIsNotNone(locate(text, "increased due to higher iPhone revenue"))


class TestRouterValidate(unittest.TestCase):
    UNI = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corp"}
    YEARS = {2023, 2024}

    def validate(self, plan):
        from sec_rag.agent.router import validate
        return validate(plan, self.UNI, self.YEARS)

    def test_null_and_scalar_fields(self):
        p = self.validate({"route": "single", "tickers": None, "fiscal_years": "2024",
                           "forms": None, "metrics": None})
        self.assertEqual(p["tickers"], [])
        self.assertEqual(p["fiscal_years"], [2024])

    def test_unknown_route_and_tickers_dropped(self):
        p = self.validate({"route": "banana", "tickers": ["aapl", "AMD", "AAPL"]})
        self.assertEqual(p["route"], "single")
        self.assertEqual(p["tickers"], ["AAPL"])

    def test_compare_needs_two_known_tickers(self):
        self.assertEqual(self.validate({"route": "compare", "tickers": ["NVDA", "AMD"]})["route"], "single")
        self.assertEqual(self.validate({"route": "compare", "tickers": ["AAPL", "MSFT"]})["route"], "compare")

    def test_numeric_defaults_to_latest_year(self):
        p = self.validate({"route": "numeric", "tickers": ["AAPL"], "metrics": ["revenue"]})
        self.assertEqual((p["route"], p["fiscal_years"]), ("numeric", [2024]))
        self.assertEqual(self.validate({"route": "numeric", "tickers": ["AAPL"]})["route"], "single")


class TestEdgarPaging(unittest.TestCase):
    def test_older_filings_are_paged_in(self):
        from unittest import mock
        from sec_rag.ingest import edgar

        def table(rows):
            keys = ["form", "reportDate", "accessionNumber", "primaryDocument", "filingDate"]
            return {k: [r[i] for r in rows] for i, k in enumerate(keys)}

        recent = table([("424B2", "", "a-1", "x.htm", "2024-12-01"),
                        ("10-K", "2024-12-31", "a-2", "k24.htm", "2025-02-20")])
        older = table([("10-K", "2024-12-31", "a-2", "k24.htm", "2025-02-20"),  # duplicate
                       ("10-K", "2023-12-31", "a-3", "k23.htm", "2024-02-20")])
        ancient = table([("10-K", "2010-12-31", "a-4", "k10.htm", "2011-02-20")])
        sub = {"filings": {"recent": recent, "files": [
            {"name": "page-001.json", "filingTo": "2024-06-30"},
            {"name": "page-002.json", "filingTo": "2012-01-01"}]}}
        fetched = []

        def fake_get(url, as_json=False, max_age=None):
            fetched.append(url)
            if url.endswith("CIK0000000001.json"):
                return sub
            return {"page-001.json": older, "page-002.json": ancient}[url.rsplit("/", 1)[1]]

        with mock.patch.object(edgar, "_get", fake_get):
            rows = edgar.list_filings("0000000001", forms=("10-K",), fiscal_years={2023, 2024})
        self.assertEqual([r["accession"] for r in rows], ["a-2", "a-3"])
        self.assertFalse(any("page-002" in u for u in fetched))  # too old, never downloaded


if __name__ == "__main__":
    unittest.main()
