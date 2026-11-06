import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "azure")
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    openai_key: str = os.getenv("OPENAI_API_KEY", "")

    router_model: str = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
    answer_model: str = os.getenv("ANSWER_MODEL", "gpt-4o")
    judge_model: str = os.getenv("JUDGE_MODEL", "gpt-4o")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    embed_dim: int = int(os.getenv("EMBED_DIM", "1536"))
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    database_url: str = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/secrag")
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "")

    chunk_target_tokens: int = int(os.getenv("CHUNK_TARGET_TOKENS", "800"))
    chunk_overlap: float = float(os.getenv("CHUNK_OVERLAP", "0.15"))

    # Retrieval knobs
    n_candidates: int = 50     # per retriever, before fusion
    n_rerank: int = 30         # fused candidates sent to the cross-encoder
    top_k: int = 10            # chunks returned after reranking
    rrf_k: int = 60            # reciprocal rank fusion constant


settings = Settings()

# USD per 1M tokens. Update to your actual Azure pricing.
PRICES = {
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "text-embedding-3-small": {"in": 0.02, "out": 0.0},
    "text-embedding-3-large": {"in": 0.13, "out": 0.0},
}
