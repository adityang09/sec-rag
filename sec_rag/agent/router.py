from .. import llm
from ..config import settings
from .financial_tool import METRICS
from .prompts import ROUTER


def route(question, universe, years):
    """universe: {ticker: company}. Returns a validated plan dict."""
    uni = ", ".join(f"{t} ({c})" for t, c in sorted(universe.items()))
    plan = llm.chat(
        [{"role": "system", "content": ROUTER.format(universe=uni, years=sorted(years), metrics=", ".join(METRICS))},
         {"role": "user", "content": question}],
        model=settings.router_model, json_mode=True, max_tokens=300)
    return validate(plan, universe, years)


def _as_list(v):
    """The model sometimes returns null, a bare string, or a list. Normalize to a list."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def validate(plan, universe, years):
    """Never trust the router's JSON blindly. Clamp everything to what exists."""
    if not isinstance(plan, dict):
        plan = {}
    tickers = list(dict.fromkeys(str(t).upper() for t in _as_list(plan.get("tickers"))
                                 if t and str(t).upper() in universe))
    fys = [int(y) for y in _as_list(plan.get("fiscal_years")) if str(y).isdigit() and int(y) in years]
    forms = [f for f in _as_list(plan.get("forms")) if f in ("10-K", "10-Q")]
    metrics = [m for m in _as_list(plan.get("metrics")) if m in METRICS]
    r = plan.get("route") if plan.get("route") in ("numeric", "single", "compare") else "single"
    if r == "numeric" and (not metrics or not tickers):
        r = "single"
    if r == "compare" and len(tickers) < 2:
        r = "single"
    if r == "numeric" and not fys and years:
        fys = [max(years)]
    return {"route": r, "tickers": tickers, "fiscal_years": fys, "forms": forms,
            "metrics": metrics}
