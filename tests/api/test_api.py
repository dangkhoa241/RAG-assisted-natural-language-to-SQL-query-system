"""Endpoint behaviour: samples, uploads, query modes, fallbacks, limits and error shapes."""
import rag_sql
from rag_sql import DailyLimitError, LLMError

from api_helpers import upload

SMALL_CSV = "city,sales,region\nParis,10,EU\nLyon,20,EU\nBoston,30,US\n"


def ask(client, question="how many patients by gender", dataset_id="healthcare", **body):
    return client.post("/api/query", json={"dataset_id": dataset_id, "question": question, **body})


# --- metadata ----------------------------------------------------------------------------
def test_health_and_config(client):
    assert client.get("/api/health").json() == {"status": "ok", "intent_model": "bert", "llm_configured": True}
    cfg = client.get("/api/config").json()
    assert cfg["default_mode"] == "auto" and cfg["glossary_default"] is False
    assert cfg["llm_model"] == "openai/gpt-oss-120b"


def test_samples_list_both_datasets_with_schema_and_examples(client):
    samples = {s["dataset_id"]: s for s in client.get("/api/samples").json()["samples"]}
    assert set(samples) == {"healthcare", "retail"}
    hc = samples["healthcare"]
    assert hc["rows"] > 0 and hc["has_glossary"] is True and hc["example_questions"]
    types = {c["name"]: c["type"] for c in hc["schema"]}
    assert types["Age"] == "numeric" and types["Gender"] == "categorical" and types["Date of Admission"] == "date"


# --- query: auto mode ----------------------------------------------------------------------
def test_auto_uses_llm_zero_shot_by_default(client, llm, docs):
    r = ask(client)
    assert r.status_code == 200
    body = r.json()
    assert body["generator"]["used"] == "llm" and body["generator"]["llm_strategy"] == "zero_shot"
    assert body["generator"]["fallback_reason"] is None
    assert body["intent"] == {"label": "count", "confidence": 0.91, "source": "bert"}
    assert body["sql"] == llm.reply
    assert [c["name"] for c in body["columns"]] == ["Gender", "n"]
    assert body["columns"][1]["type"] == "number"
    assert body["chart"]["type"] in ("pie", "bar") and body["chart"]["x"] == "Gender"
    assert set(body["latency_ms"]) == {"intent", "retrieval", "generation", "execution", "total"}
    assert docs == []                                          # glossary is off by default
    assert llm.calls[0]["model"] == "openai/gpt-oss-120b"


def test_auto_falls_back_when_llm_fails_without_leaking_the_error(client, llm):
    llm.reply = LLMError("401 invalid api key gsk_secret123 at groq.internal")
    body = ask(client).json()
    assert body["generator"]["used"] == "rule_based"
    assert body["generator"]["fallback_reason"] == "llm_error"
    assert "gsk_secret123" not in str(body) and "groq.internal" not in str(body)
    assert body["rows"]


def test_auto_falls_back_on_provider_quota(client, llm):
    llm.reply = DailyLimitError("Rate limit reached ... tokens per day (TPD)")
    assert ask(client).json()["generator"]["fallback_reason"] == "provider_quota"


def test_auto_falls_back_on_unsafe_sql(client, llm):
    llm.reply = "DROP TABLE data"
    gen = ask(client).json()["generator"]
    assert gen["used"] == "rule_based" and gen["fallback_reason"] == "unsafe_sql"
    assert gen["rejected_sql"] == "DROP TABLE data" and "safety check" in gen["fallback_detail"]


def test_auto_falls_back_when_llm_sql_fails_to_run(client, llm):
    llm.reply = "SELECT `No Such Column` FROM data"
    gen = ask(client).json()["generator"]
    assert gen["fallback_reason"] == "execution_error" and "no such column" in gen["fallback_detail"]


# --- query: llm and rule_based modes -----------------------------------------------------
def test_llm_mode_reports_failures_as_errors(client, llm):
    llm.reply = LLMError("boom")
    r = ask(client, mode="llm")
    assert r.status_code == 502 and r.json()["error"]["code"] == "llm_error"
    llm.reply = "DELETE FROM data"
    r = ask(client, mode="llm")
    assert r.status_code == 422 and r.json()["error"]["code"] == "unsafe_sql"
    llm.reply = "SELECT nope FROM data"
    r = ask(client, mode="llm")
    assert r.status_code == 422 and r.json()["error"]["code"] == "query_failed"


def test_rule_based_mode_never_calls_the_llm(client, llm):
    body = ask(client, "average billing amount by insurance provider", mode="rule_based").json()
    assert llm.calls == []
    assert body["generator"]["used"] == "rule_based" and body["generator"]["fallback_reason"] is None
    assert body["chart"]["type"] == "bar"


def test_server_default_mode_applies_when_mode_is_omitted(make_client, llm):
    client = make_client(default_mode="rule_based")
    assert ask(client).json()["generator"]["used"] == "rule_based" and llm.calls == []


# --- glossary ------------------------------------------------------------------------------
def test_glossary_toggle_retrieves_definitions_for_samples(client, llm, docs):
    llm.reply = "SELECT COUNT(*) AS n FROM data"
    body = ask(client, "how many repeat customers", dataset_id="retail", use_glossary=True).json()
    assert docs == [("how many repeat customers", "retail")]
    assert body["generator"]["llm_strategy"] == "glossary_rag"
    assert body["context"]["glossary"][0]["term"] == "Repeat customer"
    assert "Repeat customer: 3 or more orders." in llm.last_prompt


def test_glossary_toggle_is_ignored_for_uploads(client, llm, docs):
    ds = upload(client, SMALL_CSV).json()["dataset_id"]
    llm.reply = "SELECT region, SUM(sales) AS total FROM data GROUP BY region"
    body = ask(client, "total sales by region", dataset_id=ds, use_glossary=True).json()
    assert docs == [] and body["context"]["glossary"] == []
    assert body["generator"]["llm_strategy"] == "zero_shot"
    assert any("Glossary" in n for n in body["notes"])


def test_glossary_default_is_configurable(make_client, docs):
    client = make_client(glossary_default=True)
    ask(client, dataset_id="retail")
    assert len(docs) == 1


# --- limits ----------------------------------------------------------------------------------
def test_per_ip_rate_limit(make_client, llm):
    client = make_client(llm_rate_limit_per_min=2)
    assert ask(client).json()["generator"]["used"] == "llm"
    assert ask(client).json()["generator"]["used"] == "llm"
    gen = ask(client).json()["generator"]
    assert gen["used"] == "rule_based" and gen["fallback_reason"] == "rate_limited"
    r = ask(client, mode="llm")
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"
    assert int(r.headers["Retry-After"]) >= 1
    assert ask(client, mode="rule_based").status_code == 200    # rule-based isn't limited
    assert len(llm.calls) == 2


def test_daily_budget(make_client, llm):
    client = make_client(llm_daily_budget=1)
    assert ask(client).json()["generator"]["used"] == "llm"
    gen = ask(client).json()["generator"]
    assert gen["used"] == "rule_based" and gen["fallback_reason"] == "daily_budget"
    r = ask(client, mode="llm")
    assert r.status_code == 429 and r.json()["error"]["code"] == "daily_budget"
    assert client.get("/api/config").json()["llm_budget_remaining"] == 0
    assert len(llm.calls) == 1


def test_no_llm_configured(make_client, llm):
    client = make_client(llm_enabled=False)
    assert ask(client).json()["generator"]["fallback_reason"] == "llm_unavailable"
    assert ask(client, mode="llm").status_code == 503
    assert llm.calls == []


def test_result_rows_are_capped(make_client, llm):
    client = make_client(max_rows_returned=7)
    llm.reply = "SELECT * FROM data"
    body = ask(client, "show all patients").json()
    assert len(body["rows"]) == 7 and body["truncated"] is True and body["row_count"] > 7


# --- uploads -------------------------------------------------------------------------------
def test_upload_returns_schema_and_is_queryable(client, llm):
    r = upload(client, SMALL_CSV, "sales.csv")
    assert r.status_code == 201
    body = r.json()
    assert body["rows"] == 3 and body["name"] == "sales.csv" and body["has_glossary"] is False
    assert {c["name"]: c["type"] for c in body["schema"]} == {"city": "categorical", "sales": "numeric",
                                                               "region": "categorical"}
    llm.reply = "SELECT SUM(sales) AS total FROM data"
    q = ask(client, "total sales", dataset_id=body["dataset_id"]).json()
    assert q["rows"] == [[60]] and q["chart"]["type"] == "stat"


def test_upload_rejects_non_csv(client):
    r = upload(client, SMALL_CSV, "data.xlsx")
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_type"


def test_upload_rejects_oversized_file(make_client):
    client = make_client(max_upload_bytes=1024)
    r = upload(client, "a,b\n" + "1,2\n" * 2000)
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_large"


def test_upload_rejects_unparseable_and_empty_files(client):
    for content in ["", "a,b\n", '"unterminated\n1,2\n"x,"y\n']:
        r = upload(client, content)
        assert r.status_code == 422, content
        assert r.json()["error"]["code"] == "invalid_csv"
    r = client.post("/api/datasets", files={"file": ("bin.csv", b"\xff\xfe\x00\x81\x82" * 50, "text/csv")})
    assert r.status_code == 422


def test_upload_rejects_too_many_columns(make_client):
    client = make_client(max_columns=3)
    r = upload(client, "a,b,c,d\n1,2,3,4\n")
    assert r.status_code == 422 and "columns" in r.json()["error"]["message"]


def test_upload_rate_limit(make_client):
    client = make_client(upload_rate_limit_per_min=1)
    assert upload(client, SMALL_CSV).status_code == 201
    r = upload(client, SMALL_CSV)
    assert r.status_code == 429 and "Retry-After" in r.headers


def test_upload_session_cap_evicts_least_recently_used(make_client):
    client = make_client(max_sessions=2)
    ids = [upload(client, SMALL_CSV).json()["dataset_id"] for _ in range(3)]
    assert ask(client, dataset_id=ids[0], mode="rule_based").status_code == 404
    assert ask(client, dataset_id=ids[1], mode="rule_based").status_code == 200
    assert ask(client, dataset_id=ids[2], mode="rule_based").status_code == 200
    assert ask(client, dataset_id="healthcare", mode="rule_based").status_code == 200   # samples never evicted


# --- errors, CORS, body size ---------------------------------------------------------------
def test_unknown_dataset_is_404(client):
    r = ask(client, dataset_id="nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "dataset_not_found"


def test_validation_errors_are_concise(client):
    for body in [{"dataset_id": "healthcare", "question": "x", "mode": "turbo"},
                 {"dataset_id": "healthcare", "question": "   "},
                 {"dataset_id": "healthcare", "question": "x" * 501},
                 {"dataset_id": "../etc", "question": "x"},
                 {"dataset_id": "healthcare", "question": "x", "sql": "DROP TABLE data"}]:
        r = client.post("/api/query", json=body)
        assert r.status_code == 422, body
        assert r.json()["error"]["code"] == "invalid_request"


def test_unexpected_errors_hide_details(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail at C:\\path\\file.py line 42")
    monkeypatch.setattr(client.app.state.service, "run", boom)
    r = ask(client)
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "internal_error", "message": "Something went wrong on the server."}}


def test_cors_allows_only_the_frontend_origin(client):
    ok = client.options("/api/query", headers={"Origin": "http://localhost:5173",
                                               "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.get("/api/samples", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_json_body_size_limit(client):
    r = client.post("/api/query", content=b'{"dataset_id":"healthcare","question":"' + b"x" * 20000 + b'"}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_security_headers(client):
    r = client.get("/api/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["Cache-Control"] == "no-store"


def test_never_uses_gpt_oss_20b():
    assert "20b" not in rag_sql.GROQ_MODEL.replace("120b", "")


def test_execution_fallback_keeps_the_llm_strategy(client, llm, docs):
    llm.reply = "SELECT nope FROM data"
    gen = ask(client, dataset_id="retail", use_glossary=True).json()["generator"]
    assert gen["fallback_reason"] == "execution_error" and gen["llm_strategy"] == "glossary_rag"


def test_upload_rejects_too_many_rows(make_client):
    client = make_client(max_rows=5)
    r = upload(client, "a\n" + "1\n" * 6)
    assert r.status_code == 422 and "rows" in r.json()["error"]["message"]
    assert upload(client, "a\n" + "1\n" * 5).status_code == 201


def test_blob_results_are_described_not_dumped(client, llm):
    llm.reply = "SELECT randomblob(16) AS b"
    assert ask(client).json()["rows"] == [["<16 bytes>"]]


def test_example_rag_strategy_reports_retrieved_examples(make_client, llm, monkeypatch):
    example = {"question": "count rows per city", "sql": "SELECT city, COUNT(*) FROM data GROUP BY city",
               "table_sql": "SELECT city, COUNT(*) FROM shops GROUP BY city",
               "table_schema": "Table `shops` columns: city", "intent": "count", "score": 0.71}
    monkeypatch.setattr(rag_sql, "retrieve_examples", lambda *a, **k: [example])
    client = make_client(llm_strategy="example_rag")
    body = ask(client).json()
    assert body["generator"]["llm_strategy"] == "example_rag"
    assert body["context"]["examples"] == [{"question": "count rows per city",
                                            "sql": "SELECT city, COUNT(*) FROM shops GROUP BY city", "score": 0.71}]
    assert "count rows per city" in llm.last_prompt
