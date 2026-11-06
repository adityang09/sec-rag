from .config import settings

SCHEMA = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    """CREATE TABLE IF NOT EXISTS filings (
        id SERIAL PRIMARY KEY,
        ticker TEXT NOT NULL,
        cik TEXT NOT NULL,
        company TEXT NOT NULL,
        form TEXT NOT NULL,
        fiscal_year INT NOT NULL,
        period_end DATE,
        filed DATE,
        accession TEXT UNIQUE NOT NULL,
        url TEXT NOT NULL,
        text TEXT NOT NULL
    )""",
    f"""CREATE TABLE IF NOT EXISTS chunks (
        id BIGSERIAL PRIMARY KEY,
        filing_id INT NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
        strategy TEXT NOT NULL,
        ticker TEXT NOT NULL,
        form TEXT NOT NULL,
        fiscal_year INT NOT NULL,
        section TEXT,
        section_title TEXT,
        chunk_index INT NOT NULL,
        char_start INT NOT NULL,
        char_end INT NOT NULL,
        n_tokens INT NOT NULL,
        text TEXT NOT NULL,
        embed_text TEXT NOT NULL,
        embedding vector({settings.embed_dim})
    )""",
    "CREATE INDEX IF NOT EXISTS chunks_filter_idx ON chunks (strategy, ticker, fiscal_year, form)",
    "CREATE INDEX IF NOT EXISTS chunks_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops)",
    """CREATE TABLE IF NOT EXISTS financial_facts (
        ticker TEXT NOT NULL,
        cik TEXT NOT NULL,
        concept TEXT NOT NULL,
        unit TEXT NOT NULL,
        value NUMERIC NOT NULL,
        period_start DATE,
        period_end DATE NOT NULL,
        fy INT,
        fp TEXT,
        form TEXT,
        filed DATE,
        accn TEXT NOT NULL,
        UNIQUE NULLS NOT DISTINCT (ticker, concept, unit, period_start, period_end, accn)
    )""",
    "CREATE INDEX IF NOT EXISTS facts_lookup_idx ON financial_facts (ticker, concept, period_end)",
    """CREATE TABLE IF NOT EXISTS query_log (
        id BIGSERIAL PRIMARY KEY,
        ts TIMESTAMPTZ DEFAULT now(),
        question TEXT,
        route TEXT,
        plan JSONB,
        latency_ms JSONB,
        prompt_tokens INT,
        completion_tokens INT,
        embedding_tokens INT,
        cost_usd NUMERIC,
        answer TEXT
    )""",
]


def tune_vector_search(conn):
    """Make filtered HNSW searches return enough rows.

    pgvector applies WHERE filters AFTER the HNSW graph search, so a selective filter (one
    company, one year) can return fewer rows than LIMIT. pgvector >= 0.8 fixes this with iterative
    scans. Older versions (common on managed Postgres) only get a larger ef_search. The version is
    checked explicitly, because on some setups SET of an unknown hnsw.* name succeeds silently.
    """
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    if not row:
        return
    version = tuple(int(x) for x in row[0].split(".")[:2] if x.isdigit())
    if version >= (0, 8):
        conn.execute("SET hnsw.iterative_scan = relaxed_order")
    else:
        conn.execute("SET hnsw.ef_search = 200")


def _configure(conn):
    from pgvector.psycopg import register_vector
    register_vector(conn)
    tune_vector_search(conn)


def connect():
    """A single connection, for scripts (ingestion, evaluation)."""
    import psycopg
    conn = psycopg.connect(settings.database_url, autocommit=True)
    _configure(conn)
    return conn


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class PooledDB:
    """Connection pool with the same .execute(...).fetchone()/.fetchall() surface as a connection.

    The API serves requests from a thread pool and the compare route searches companies in
    parallel threads, so a single shared connection would serialize every query. Rows are fully
    fetched before the connection goes back to the pool.
    """

    def __init__(self, url=None, min_size=1, max_size=10):
        from psycopg_pool import ConnectionPool
        self.pool = ConnectionPool(url or settings.database_url, min_size=min_size, max_size=max_size,
                                   kwargs={"autocommit": True}, configure=_configure, open=True)

    def execute(self, sql, params=None):
        with self.pool.connection() as conn:
            cur = conn.execute(sql, params)
            return _Rows(cur.fetchall() if cur.description else [])

    def close(self):
        self.pool.close()


def init_db():
    import psycopg
    conn = psycopg.connect(settings.database_url, autocommit=True)
    for stmt in SCHEMA:
        conn.execute(stmt)
    conn.close()


if __name__ == "__main__":
    init_db()
    print("schema ready")
