"""Thin LLM wrapper that records tokens, cost and calls for every request."""
import contextvars
import json
import time
from dataclasses import dataclass, field

from .config import PRICES, settings

_client = None


def client():
    global _client
    if _client is None:
        if settings.llm_provider == "azure":
            from openai import AzureOpenAI
            _client = AzureOpenAI(api_key=settings.azure_key,
                                  api_version=settings.azure_api_version,
                                  azure_endpoint=settings.azure_endpoint)
        else:
            from openai import OpenAI
            _client = OpenAI(api_key=settings.openai_key)
    return _client


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    cost_usd: float = 0.0
    calls: list = field(default_factory=list)

    def add(self, model, tin, tout, kind):
        p = PRICES.get(model, {"in": 0.0, "out": 0.0})
        self.cost_usd += tin / 1e6 * p["in"] + tout / 1e6 * p["out"]
        if kind == "embed":
            self.embedding_tokens += tin
        else:
            self.prompt_tokens += tin
            self.completion_tokens += tout
        self.calls.append({"model": model, "kind": kind, "in": tin, "out": tout})


_usage: contextvars.ContextVar = contextvars.ContextVar("usage", default=None)


def start_tracking() -> Usage:
    u = Usage()
    _usage.set(u)
    return u


def _record(model, tin, tout, kind):
    u = _usage.get()
    if u is not None:
        u.add(model, tin, tout, kind)


def chat(messages, model=None, json_mode=False, temperature=0.0, max_tokens=1200, retries=3):
    model = model or settings.answer_model
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(retries):
        try:
            r = client().chat.completions.create(**kwargs)
            _record(model, r.usage.prompt_tokens, r.usage.completion_tokens, "chat")
            text = r.choices[0].message.content or ""
            return json.loads(text) if json_mode else text
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def embed(texts, model=None, batch_size=128):
    model = model or settings.embed_model
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(5):
            try:
                r = client().embeddings.create(model=model, input=batch)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        _record(model, r.usage.prompt_tokens, 0, "embed")
        out.extend(d.embedding for d in r.data)
    return out
