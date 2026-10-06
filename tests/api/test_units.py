"""Pure units: chart suggestion, rate limiting, the daily budget, session expiry, settings and glossary gating."""
import pandas as pd
import pytest

from backend.chart import suggest_chart
from backend.config import Settings
from backend.glossary import GATE, matched_definitions
from backend.ratelimit import DailyBudget, SlidingWindowLimiter
from backend.sessions import SessionStore, load_session_dataset, make_session


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# --- suggest_chart -----------------------------------------------------------------------
@pytest.mark.parametrize("intent, df, sql, expected", [
    ("count", pd.DataFrame({"n": [42]}), "SELECT COUNT(*) AS n FROM data", "stat"),
    ("aggregate", pd.DataFrame({"avg": [1.5]}), "", "stat"),
    ("filter", pd.DataFrame({"name": ["a", "b"], "age": [1, 2]}), "", "table"),
    ("aggregate", pd.DataFrame(columns=["k", "v"]), "", "table"),
    ("aggregate", pd.DataFrame({"a": ["x"], "b": ["y"]}), "", "table"),
    ("trend", pd.DataFrame({"month": ["2023-01", "2023-02"], "n": [3, 4]}), "", "line"),
    ("count", pd.DataFrame({"year": ["2019", "2020", "2021"], "n": [1, 2, 3]}), "", "line"),
    ("trend", pd.DataFrame({"year": [2019, 2020], "n": [1, 2]}), "", "line"),
    ("compare", pd.DataFrame({"g": ["M", "F"], "total": [10, 20]}), "SELECT g, SUM(x)", "pie"),
    ("compare", pd.DataFrame({"g": ["M", "F"], "avg": [10, 20]}), "SELECT g, AVG(x)", "bar"),
    ("compare", pd.DataFrame({"g": ["M", "F"], "d": [-1, 20]}), "SELECT g, SUM(x)", "bar"),
    ("aggregate", pd.DataFrame({"g": list("abcdefgh"), "v": range(8)}), "", "bar"),
    ("aggregate", pd.DataFrame({"g": [str(i) + "x" for i in range(60)], "v": range(60)}), "", "table"),
])
def test_suggest_chart(intent, df, sql, expected):
    assert suggest_chart(intent, df, sql)["type"] == expected


def test_suggest_chart_axes():
    chart = suggest_chart("aggregate", pd.DataFrame({"region": ["E", "W"], "rev": [1.0, 2.0], "n": [3, 4]}))
    assert (chart["type"], chart["x"], chart["y"]) == ("bar", "region", "rev")
    stat = suggest_chart("count", pd.DataFrame({"n": [5]}))
    assert stat["x"] is None and stat["y"] == "n"


# --- rate limiting -----------------------------------------------------------------------
def test_sliding_window_limiter():
    clock = Clock()
    lim = SlidingWindowLimiter(2, 60, clock)
    assert lim.hit("ip1") == (True, 0)
    assert lim.hit("ip1") == (True, 0)
    allowed, retry = lim.hit("ip1")
    assert not allowed and retry == 60
    assert lim.hit("ip2") == (True, 0)            # other clients are unaffected
    clock.t += 60.5
    assert lim.hit("ip1") == (True, 0)            # the window slid past the old hits


def test_daily_budget_resets_at_utc_midnight():
    day = ["2026-10-05"]
    budget = DailyBudget(2, today=lambda: day[0])
    assert budget.try_consume() and budget.try_consume()
    assert not budget.try_consume() and budget.remaining == 0
    day[0] = "2026-10-06"
    assert budget.remaining == 2 and budget.try_consume()


# --- sessions ------------------------------------------------------------------------------
def _session():
    return make_session(load_session_dataset(b"a,b\n1,x\n2,y\n"), "t.csv")


def test_session_ttl_expires_idle_uploads_and_use_refreshes_it():
    clock = Clock()
    store = SessionStore(max_uploads=5, ttl_s=100, clock=clock)
    s1, s2 = store.add_upload(_session()), store.add_upload(_session())
    clock.t += 60
    assert store.get(s1.id) is s1                 # refreshes s1 only
    clock.t += 60
    assert store.get(s2.id) is None               # idle for 120 s > ttl
    assert store.get(s1.id) is s1


def test_session_cap():
    store = SessionStore(max_uploads=2, ttl_s=100, clock=Clock())
    s = [store.add_upload(_session()) for _ in range(3)]
    assert store.get(s[0].id) is None and len(store) == 2


def test_upload_connection_is_usable_from_another_thread():
    import threading
    session = _session()
    out = []
    t = threading.Thread(target=lambda: out.append(session.dataset.conn.execute("SELECT COUNT(*) FROM data").fetchone()))
    t.start(); t.join()
    assert out == [(2,)]


# --- settings and glossary gating ------------------------------------------------------------
def test_llm_provider_and_model_settings(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "CEREBRAS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test")
    s = Settings.from_env()
    assert (s.llm_provider, s.llm_model, s.llm_enabled, s.glossary_default) == ("groq", "openai/gpt-oss-120b", True, True)
    assert s.llm_fallback_models == ("openai/gpt-oss-20b",)

    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "")                 # set but empty: no fallback
    s = Settings.from_env()
    assert (s.llm_model, s.llm_fallback_models) == ("openai/gpt-oss-20b", ())
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "a/one, a/two")
    assert Settings.from_env().llm_fallback_models == ("a/one", "a/two")

    monkeypatch.delenv("LLM_MODEL")
    monkeypatch.delenv("LLM_FALLBACK_MODELS")
    monkeypatch.setenv("LLM_PROVIDER", "cerebras")
    s = Settings.from_env()
    assert (s.llm_provider, s.llm_model, s.llm_enabled) == ("cerebras", "gpt-oss-120b", False)   # no Cerebras key
    assert s.llm_fallback_models == ()                                # Cerebras doesn't serve the 20b

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_gate_uses_the_frozen_stage3c_settings():
    assert GATE["doc_method"] == "gated" and GATE["max_chunks"] is None and GATE["retrieval_fallback"] is None


def test_app_gating_config_matches_the_frozen_benchmark_config():
    import json

    from backend import ROOT_DIR
    from backend.glossary import GATING_CONFIG

    assert GATING_CONFIG == ROOT_DIR / "config" / "glossary_gating.json"
    app = json.loads(GATING_CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads((ROOT_DIR / "eval" / "stage3c_config.json").read_text(encoding="utf-8"))
    assert app == frozen


@pytest.mark.parametrize("question, glossary, expected", [
    ("what is our ARR?", "saas", [("ss_arr", "ARR")]),
    ("how many active accounts do we have?", "saas", [("ss_active_account", "Active account")]),
    ("total MRR by plan", "saas", []),
    ("revenue in FY2024", "retail", [("rt_fiscal_year", "FY")]),                     # FY<year> matches FY
    ("how many high-cost admissions were there", "healthcare", [("hc_high_cost", "High-cost admission")]),
])
def test_matched_definitions(question, glossary, expected):
    assert [(d["id"], d["matched"]) for d in matched_definitions(question, glossary)] == expected
