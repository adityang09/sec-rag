import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from .. import llm
from ..agent.graph import SecAgent
from ..db import PooledDB
from ..retrieval.hybrid import HybridRetriever

state = {}


@asynccontextmanager
async def lifespan(app):
    db = PooledDB()
    state["conn"] = db
    state["agent"] = SecAgent(db, HybridRetriever(db, strategy="section"))
    yield
    db.close()


app = FastAPI(title="SEC Filings Agentic RAG", lifespan=lifespan)


class Ask(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True, "chunks": len(state["agent"].ret.chunks)}


@app.post("/ask")
def ask(req: Ask):
    usage = llm.start_tracking()
    out = state["agent"].ask(req.question)
    state["conn"].execute(
        """INSERT INTO query_log (question, route, plan, latency_ms, prompt_tokens, completion_tokens,
               embedding_tokens, cost_usd, answer) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (req.question, out["plan"]["route"], json.dumps(out["plan"]), json.dumps(out["timings"]),
         usage.prompt_tokens, usage.completion_tokens, usage.embedding_tokens, usage.cost_usd, out["answer"]))
    return {"answer": out["answer"], "route": out["plan"]["route"], "plan": out["plan"],
            "citations": out.get("citations", []), "uncited": out.get("uncited", False),
            "latency_ms": out["timings"], "cost_usd": round(usage.cost_usd, 5),
            "tokens": {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens,
                       "embedding": usage.embedding_tokens}}


@app.get("/stats")
def stats(last: int = 200):
    row = state["conn"].execute(
        """SELECT count(*),
                  percentile_cont(0.5) WITHIN GROUP (ORDER BY (latency_ms->>'total_ms')::float),
                  percentile_cont(0.95) WITHIN GROUP (ORDER BY (latency_ms->>'total_ms')::float),
                  avg(cost_usd)
           FROM (SELECT * FROM query_log ORDER BY id DESC LIMIT %s) q""", (last,)).fetchone()
    routes = state["conn"].execute(
        "SELECT route, count(*) FROM (SELECT route FROM query_log ORDER BY id DESC LIMIT %s) q GROUP BY route",
        (last,)).fetchall()
    return {"queries": row[0], "p50_ms": row[1], "p95_ms": row[2],
            "avg_cost_usd": float(row[3] or 0), "routes": dict(routes)}
