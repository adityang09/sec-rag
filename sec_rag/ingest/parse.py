"""Turn filing HTML into clean text, then find the Item sections.

Two details that matter a lot for retrieval quality:
1. Tables are flattened row by row with " | " between cells, so a revenue table stays one
   readable block instead of hundreds of one-number lines.
2. Every 10-K starts with a table of contents that repeats every "Item X" header. We keep, for
   each item, the occurrence that starts the LONGEST span. TOC entries are only a line long,
   so the real section always wins.
"""
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString

BLOCK_TAGS = ["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"]

TENK_TITLES = {
    "1": "Item 1. Business", "1A": "Item 1A. Risk Factors", "1B": "Item 1B. Unresolved Staff Comments",
    "1C": "Item 1C. Cybersecurity", "2": "Item 2. Properties", "3": "Item 3. Legal Proceedings",
    "4": "Item 4. Mine Safety Disclosures", "5": "Item 5. Market for Common Equity",
    "6": "Item 6. Reserved", "7": "Item 7. Management's Discussion and Analysis (MD&A)",
    "7A": "Item 7A. Market Risk Disclosures", "8": "Item 8. Financial Statements",
    "9": "Item 9. Changes in Accountants", "9A": "Item 9A. Controls and Procedures",
    "9B": "Item 9B. Other Information", "9C": "Item 9C. Foreign Jurisdiction Inspections",
    "10": "Item 10. Directors and Governance", "11": "Item 11. Executive Compensation",
    "12": "Item 12. Security Ownership", "13": "Item 13. Related Transactions",
    "14": "Item 14. Accountant Fees", "15": "Item 15. Exhibits", "16": "Item 16. Form 10-K Summary",
}
TENQ_TITLES = {
    "I-1": "Part I Item 1. Financial Statements", "I-2": "Part I Item 2. MD&A",
    "I-3": "Part I Item 3. Market Risk", "I-4": "Part I Item 4. Controls and Procedures",
    "II-1": "Part II Item 1. Legal Proceedings", "II-1A": "Part II Item 1A. Risk Factors",
    "II-2": "Part II Item 2. Equity Sales and Repurchases", "II-5": "Part II Item 5. Other Information",
    "II-6": "Part II Item 6. Exhibits",
}

ITEM_RE = re.compile(r"^[ \t]*item[ \t]+(\d{1,2}[a-c]?)\b[ \t]*[\.\:\-\u2013\u2014]?(.{0,150})$",
                     re.IGNORECASE | re.MULTILINE)
PART_RE = re.compile(r"^[ \t]*part[ \t]+(iv|iii|ii|i)\b", re.IGNORECASE | re.MULTILINE)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    # Inline XBRL hides a big metadata block in <ix:header>
    for tag in soup.find_all(lambda t: t.name and t.name.lower() in ("ix:header",)):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()

    for table in soup.find_all("table"):
        lines = []
        for tr in table.find_all("tr"):
            cells = [re.sub(r"\s+", " ", td.get_text(" ")).strip() for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c and c not in ("$", "%", ")")]
            if cells:
                lines.append(" | ".join(cells))
        table.replace_with(NavigableString("\n\n" + "\n".join(lines) + "\n\n"))

    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_after(NavigableString("\n"))

    text = soup.get_text()
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class Section:
    key: str
    title: str
    start: int
    end: int


def split_sections(text: str, form: str):
    is_q = form.upper().startswith("10-Q")
    parts = [(m.start(), m.group(1).upper()) for m in PART_RE.finditer(text)]
    headers = []
    for m in ITEM_RE.finditer(text):
        item = m.group(1).upper()
        part = "I"
        for pos, p in parts:
            if pos <= m.start():
                part = p
            else:
                break
        key = f"{part}-{item}" if is_q else item
        headers.append((m.start(), key))

    if not headers:
        return [Section("FULL", "Full document", 0, len(text))]

    # Candidate spans between consecutive headers, keep the longest per key
    best = {}
    for i, (pos, key) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        if key not in best or (end - pos) > (best[key][1] - best[key][0]):
            best[key] = (pos, end)

    titles = TENQ_TITLES if is_q else TENK_TITLES
    spans = sorted((s, e, k) for k, (s, e) in best.items())
    sections = []
    # Make spans non-overlapping: each section ends where the next kept one starts
    for i, (s, e, k) in enumerate(spans):
        nxt = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        end = min(e, nxt) if e > s else nxt
        sections.append(Section(k, titles.get(k, f"Item {k}"), s, max(end, s)))
    if sections and sections[0].start > 0:
        sections.insert(0, Section("COVER", "Cover page and table of contents", 0, sections[0].start))
    return [s for s in sections if s.end - s.start > 0]
