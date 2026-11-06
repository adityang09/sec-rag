# Agentic RAG Assistant for SEC Filings

Ask questions about 10-K and 10-Q filings in plain English. A LangGraph agent decides whether the
question needs a reported number, an explanation from one company's filing, or a comparison
across companies, and sends it down a different path for each. Every answer cites its sources.

```
                      +--> numeric_lookup (SQL over XBRL facts) ---------(facts found)---+
                      |          |                                                        |
                      |      (no match)                                                   |
                      |          v                                                        v
question --> route ---+--> single_retrieve (hybrid search, 1 company) -------------> generate --> check_citations
                      |                                                                   ^
                      +--> compare_retrieve (one search per company, in parallel) --------+
```

Stack: Python, LangGraph, FastAPI, PostgreSQL + pgvector, rank-bm25, sentence-transformers
cross-encoder, Azure OpenAI, Docker.

---

## 1. Quick start

Requirements: Python 3.11+, Docker, and an Azure OpenAI (or OpenAI) key. The SEC asks every client
to send a User-Agent with a real name and email, so set `SEC_USER_AGENT` in `.env`.

```bash
cp .env.example .env            # fill in Azure OpenAI keys and SEC_USER_AGENT
docker compose up -d db
pip install -r requirements.txt
python -m unittest -v           # offline tests, no keys needed

# Ingest 10 companies, 3 fiscal years, 10-K plus one 10-Q per year, all 3 chunking strategies
python -m sec_rag.ingest.pipeline \
  --tickers AAPL MSFT NVDA AMZN JPM BK GS XOM JNJ WMT \
  --years 2022 2023 2024 --forms 10-K 10-Q

uvicorn sec_rag.api.main:app --reload
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "How do NVIDIA and Microsoft describe supply chain risk in their fiscal 2024 10-Ks?"}'
```

On Windows PowerShell, use `copy` instead of `cp`, put the ingest command on one line (or replace
the trailing `\` with a backtick), and send the request with:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType 'application/json' `
  -Body '{"question": "How do NVIDIA and Microsoft describe supply chain risk in their fiscal 2024 10-Ks?"}'
```

Only companies you ingested can be asked about. The router drops tickers it does not know, so
a question about a company outside the index is answered as a single-company question.

SEC responses are cached under `data/raw/`. Filing documents are cached forever. The ticker map,
submissions and XBRL company facts expire after a day so new filings are picked up.

Rough cost of a full run: embeddings for ~60 filings across 3 strategies is about 12 to 15M tokens,
roughly $0.25 to $0.30 with text-embedding-3-small. Building the gold set and running the answer
eval is a few dollars on gpt-4o. Check current Azure pricing and update `PRICES` in `config.py`.

---

## 2. Design decisions

### 2.1 Fetching

* **Full filing history.** The SEC submissions API only inlines the ~1000 most recent filings.
  Banks like JPMorgan and Goldman Sachs file thousands of prospectuses a year, which pushes
  older 10-Ks out of that window. The client pages through the older submission files, skipping
  pages that only hold filings older than the requested years.

### 2.2 Parsing

* **Tables are flattened row by row** with ` | ` between cells, and empty cells, lone `$` and `%`
  are dropped. Without this, a revenue table becomes hundreds of one-number lines and retrieval
  can never match "revenue 2024" to the right number.
* **Inline XBRL metadata is removed.** Modern filings hide a large `<ix:header>` block and
  `display:none` divs that would otherwise pollute the text.
* **Table of contents handling.** Every 10-K repeats each "Item X" header in its table of
  contents. The parser finds all header candidates and keeps, for each item, the one that starts
  the longest span. TOC entries are one line long, so the real section always wins. 10-Qs reuse
  item numbers across Part I and Part II, so 10-Q keys are prefixed (`I-2` is MD&A, `II-1A` is
  risk factors).

### 2.3 Chunking: section-aware, ~800 tokens, 15% overlap

* **Boundaries follow Items.** A chunk never spans Risk Factors and MD&A. Mixed chunks embed to a
  blurry average of two topics and rank poorly for both.
* **Whole paragraphs are packed** up to ~800 tokens. Paragraphs over the limit are split on
  sentence ends or line breaks, so table rows are not cut in half.
* **Why ~800 tokens.** Filing paragraphs are long, and analyst questions usually need a full
  argument ("why did margin fall") rather than one sentence. Smaller chunks (200 to 300) lose
  that context. Much larger chunks dilute the embedding. 800 is a starting point. Try 400 and
  1200 in the ablation and keep what wins on your gold set.
* **15% overlap** is carried as whole trailing paragraphs, so a claim that straddles a boundary
  appears whole in at least one chunk.
* **Contextual header.** Each chunk is embedded as
  `Apple Inc. (AAPL) 10-K fiscal year 2024 | Item 1A. Risk Factors` followed by the text. Raw
  chunks rarely say which company or year they belong to, so without the header "Apple risk
  factors 2024" and "Microsoft risk factors 2024" look nearly identical to the embedder.
* **Three strategies are indexed** so the effect can be measured:
  `fixed` (plain 800-token windows), `fixed_header` (same windows plus the document header), and
  `section` (section boundaries plus the section header). This separates "the header helped"
  from "the boundaries helped".

### 2.4 Retrieval: hybrid BM25 + dense, RRF fusion, cross-encoder rerank

1. **Dense**: pgvector cosine search, top 50, with metadata filters in SQL.
2. **BM25**: in-memory `rank_bm25` over the embed text, top 50. Catches exact tokens that
   embeddings blur, like "Item 1A", segment names, product names and numbers.
3. **Reciprocal Rank Fusion** with k=60. RRF uses ranks, not scores, so BM25 scores and cosine
   similarities never need calibration against each other.
4. **Cross-encoder rerank** of the top 30 fused results with `ms-marco-MiniLM-L-6-v2`, keep 10.
   The model reads at most 512 tokens but chunks are ~800, so each chunk is scored in ~300-word
   windows and the max window score is used. Otherwise the reranker never sees the second half
   of a chunk.
5. **Filtered HNSW gotcha.** pgvector filters after the HNSW graph search, so a selective filter
   (one company, one year) can return fewer rows than asked for. Each connection checks the
   installed pgvector version once: 0.8+ gets iterative scans, older versions get a larger
   `ef_search`.
6. **Filter relaxation.** If the router's filters return fewer than 3 chunks (wrong year, no
   10-Q that quarter), it retries dropping the form filter, then the year filter.

### 2.5 Routing

The router is a small, cheap model (gpt-4o-mini) returning JSON: route, tickers, fiscal years,
forms and metrics. Its output is **never trusted directly**. `router.validate` clamps every field
to what exists in the database:

* malformed output (null fields, a bare string instead of a list, an unknown route) is normalized
* a "numeric" plan with no recognized metric or no known ticker becomes "single"
* a "compare" plan with fewer than two known tickers becomes "single"
* unknown tickers and years are dropped
* a numeric question with no year gets the latest year

| Route   | Example | Path |
|---|---|---|
| numeric | "What was JPMorgan's net income in FY2023?" | SQL tool over XBRL facts |
| single  | "What cybersecurity risks did Microsoft disclose in 2024?" | Hybrid retrieval for one company |
| compare | "How do Apple and Microsoft describe AI investment?" | One retrieval per company, in parallel |

### 2.6 The numeric tool (SQL over XBRL)

* **Constrained, not text-to-SQL.** The LLM picks a metric name from a whitelist of 13. The SQL
  is fixed and parameterized. No injection risk, no invented columns.
* **Concept mapping.** One metric maps to several XBRL tags in priority order. Revenue alone has
  `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `SalesRevenueNet` and, for
  banks, `RevenuesNetOfInterestExpense`.
* **The fiscal year trap.** In XBRL company facts, the `fy` field is the year of the FILING, not
  of the value. A FY2024 10-K also reports FY2023 and FY2022 numbers, all tagged `fy=2024`.
  The tool selects by `period_end` instead and requires a ~1 year duration for flow metrics, so
  quarterly values inside a 10-K are not picked up. The latest filed value wins, so restatements
  are respected.
* **No LLM arithmetic.** Growth rates are computed in Python and passed in as `[computed]` lines.
* **Graceful fallback.** If no XBRL fact matches, the graph falls back to text retrieval.

### 2.7 Generation and citations

* Sources are numbered `[S1]...` for text chunks and `[F1]...` for XBRL facts. The prompt requires
  a tag on every claim and says XBRL numbers win over numbers in text.
* `check_citations` removes tags that point at nothing (a common LLM failure) and flags answers
  with no citations at all. The API returns structured citations with filing URLs.

### 2.8 Observability

Every request records tokens and cost per model call (via a context variable that is copied into
the parallel compare threads) and latency per graph node, plus retrieval and rerank time. Everything
goes to the `query_log` table. `GET /stats` returns p50 and p95 latency, average cost and the route
mix. The API uses a Postgres connection pool, so concurrent requests and the parallel compare
searches do not queue behind one connection.

---

## 3. Evaluation

### 3.1 Gold set (target 200: 120 single, 50 compare, 30 numeric)

```bash
python -m eval.build_gold generate     # drafts ~235 candidates
python -m eval.build_gold review       # you keep, edit or skip each one
```

* Passages are sampled as random windows of the **raw filing text**, not from any strategy's
  chunks. Sampling from section chunks would quietly favor the section strategy.
* Gold evidence is a **character span** in the filing, found by locating the verbatim quote.
  This is what makes strategies comparable, because chunk ids differ across strategies. A chunk
  counts as a hit if it covers at least half the span.
* Numeric questions are built from XBRL facts, so their expected values are exact.
* **Review every item.** LLM-drafted questions are sometimes vague or answerable from many
  places. Only reviewed items are used in evaluation.

### 3.2 Retrieval ablation

```bash
python -m eval.retrieval_eval
```

Runs 8 configurations (strategy x mode x rerank) and prints recall@10 and MRR@10, overall and per
question type. Filters use the gold company and year, so it measures retrieval alone. Pass `--no-filters` to
search the whole corpus. The headline comparison is the `fixed / dense` row (the naive baseline)
against the `section / hybrid / rerank` row (the full pipeline).

### 3.3 End-to-end answers

```bash
python -m eval.answer_eval
```

Reports route accuracy, faithfulness (share of claims supported by sources), citation accuracy
(share of citations whose source supports the sentence), numeric exact match, p50/p95 latency and
cost per query. Judge with a model at least as strong as the answer model, and spot-check 20
judgments by hand to make sure the judge agrees with you.

### Results

Results are not published yet. Both eval scripts print a Markdown table and write their raw
output to `eval/*results*`, which is git-ignored.

---|---|---|
| fixed, dense | | |
| fixed_header, hybrid | | |
| section, hybrid | | |
| section, hybrid, rerank | | |

| Metric | Value |
|---|---|
| Route accuracy | |
| Faithfulness | |
| Citation accuracy | |
| Numeric exact | |
| Latency p50 / p95 | |
| Cost per query | |

---

## 4. Deploy on Azure

```bash
RG=sec-rag LOC=eastus
az group create -n $RG -l $LOC

# Postgres Flexible Server with pgvector
az postgres flexible-server create -g $RG -n secrag-db --tier Burstable --sku-name Standard_B2s \
  --version 16 --admin-user rag --admin-password '<strong-password>' --public-access 0.0.0.0
az postgres flexible-server parameter set -g $RG -s secrag-db --name azure.extensions --value VECTOR
az postgres flexible-server db create -g $RG -s secrag-db -d secrag

# Image and container app
az acr create -g $RG -n secragacr --sku Basic --admin-enabled true
az acr build -r secragacr -t sec-rag:latest .
az containerapp up -g $RG -n sec-rag-api --image secragacr.azurecr.io/sec-rag:latest \
  --ingress external --target-port 8000 --env-vars DATABASE_URL=<azure-pg-url> LLM_PROVIDER=azure \
  AZURE_OPENAI_ENDPOINT=<...> AZURE_OPENAI_API_KEY=<...>
```

Run ingestion once from your laptop pointed at the Azure database URL. Give the container app at
least 2 GiB memory, since the BM25 index and reranker live in memory. Azure's pgvector version
may be older than 0.8, which is why the retriever falls back to `ef_search`.

---

## 5. Known limitations and next steps

* BM25 lives in memory and is rebuilt at startup. Fine for ~30k chunks. At larger scale, move it
  to Postgres full-text search or a search engine like OpenSearch.
* The numeric tool only handles annual values. Quarterly values need fiscal-quarter mapping per
  company, since fiscal calendars differ.
* The numeric tool maps each metric to a short list of common XBRL tags. Companies that report
  under other tags (for example cash including restricted cash) fall back to text retrieval.
* The router is a single LLM call. A few-shot prompt or a small classifier trained on logged
  queries would cut cost and latency further.
* No query rewriting or multi-hop retrieval yet ("what caused the drop in the segment that grew
  fastest last year" needs two steps).

---

## 6. Repo map

```
sec_rag/
  config.py               settings, model names, prices
  db.py                   schema, connection pool, pgvector tuning
  llm.py                  Azure/OpenAI wrapper with token and cost tracking
  ingest/edgar.py         SEC API client with caching and rate limiting
  ingest/parse.py         HTML to text, table flattening, Item section detection
  ingest/chunking.py      section-aware, fixed and fixed_header chunkers
  ingest/pipeline.py      ingestion CLI
  retrieval/fusion.py     tokenizer, RRF, reranker windows (pure, tested)
  retrieval/hybrid.py     BM25 + pgvector + RRF + cross-encoder
  agent/router.py         LLM router with validation
  agent/financial_tool.py constrained SQL tool over XBRL facts
  agent/graph.py          LangGraph wiring, parallel compare, citation checks
  api/main.py             FastAPI: /ask, /stats, /health
eval/
  build_gold.py           gold set generation and review CLI
  metrics.py              span-based recall@k, MRR, quote locating
  retrieval_eval.py       ablation grid
  answer_eval.py          faithfulness, citation accuracy, latency, cost
tests/test_core.py        offline tests for parsing, chunking, fusion, metrics, routing and EDGAR paging
```

---

## License

MIT. See [LICENSE](LICENSE).
