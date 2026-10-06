"""The model chain: gpt-oss-120b first, gpt-oss-20b when the 120b is out of quota or rate-limited, then rule-based.
Both providers' errors are faked; no test makes a network call."""
from rag_sql import DailyLimitError, LLMError, RateLimitError

BIG, SMALL = "openai/gpt-oss-120b", "openai/gpt-oss-20b"
QUOTA = DailyLimitError("Rate limit reached for model ... tokens per day (TPD). Please try again in 7m12s")
RATE = RateLimitError("gave up after 1 rate-limited attempts: tokens per minute (TPM)")


def ask(client, question="how many patients by gender", dataset_id="healthcare", **body):
    return client.post("/api/query", json={"dataset_id": dataset_id, "question": question, **body})


def models_called(llm):
    return [c["model"] for c in llm.calls]


def test_120b_quota_falls_back_to_20b(client, llm):
    llm.by_model[BIG] = QUOTA
    gen = ask(client).json()["generator"]
    assert models_called(llm) == [BIG, SMALL]
    assert gen["used"] == "llm" and gen["model"] == SMALL and gen["fallback_reason"] is None
    assert gen["model_note"] == "gpt-oss-20b (120b quota exhausted)"
    assert gen["models_tried"] == [{"model": BIG, "outcome": "quota_exhausted"}, {"model": SMALL, "outcome": "answered"}]


def test_primary_fails_over_on_the_first_429_and_only_the_last_model_retries(client, llm):
    llm.by_model[BIG] = RATE
    gen = ask(client).json()["generator"]
    assert [c["max_retries"] for c in llm.calls] == [1, None]   # None = the configured LLM_MAX_RETRIES
    assert gen["model"] == SMALL and gen["model_note"] == "gpt-oss-20b (120b rate-limited)"


def test_both_quotas_exhausted_fall_back_to_rule_based(client, llm):
    llm.by_model.update({BIG: QUOTA, SMALL: QUOTA})
    body = ask(client).json()
    gen = body["generator"]
    assert models_called(llm) == [BIG, SMALL]
    assert gen["used"] == "rule_based" and gen["fallback_reason"] == "provider_quota"
    assert gen["model"] is None and gen["model_note"] is None
    assert [m["outcome"] for m in gen["models_tried"]] == ["quota_exhausted", "quota_exhausted"]
    assert body["rows"] and "TPD" not in str(body)          # the provider's message never reaches the client


def test_quota_then_rate_limit_reports_the_last_failure(client, llm):
    llm.by_model.update({BIG: QUOTA, SMALL: RATE})
    gen = ask(client).json()["generator"]
    assert gen["used"] == "rule_based" and gen["fallback_reason"] == "provider_rate_limited"


def test_llm_mode_uses_the_fallback_model_and_errors_only_when_both_are_out(client, llm):
    llm.by_model[BIG] = QUOTA
    r = ask(client, mode="llm")
    assert r.status_code == 200 and r.json()["generator"]["model"] == SMALL
    llm.by_model[SMALL] = QUOTA
    r = ask(client, mode="llm")
    assert r.status_code == 429 and r.json()["error"]["code"] == "provider_quota"


def test_other_llm_errors_do_not_try_the_fallback_model(client, llm):
    llm.by_model[BIG] = LLMError("500 internal error")
    gen = ask(client).json()["generator"]
    assert models_called(llm) == [BIG]
    assert gen["used"] == "rule_based" and gen["fallback_reason"] == "llm_error"
    assert gen["models_tried"] == [{"model": BIG, "outcome": "failed"}]


def test_unsafe_sql_from_the_primary_does_not_try_the_fallback_model(client, llm):
    llm.by_model[BIG] = "DROP TABLE data"
    gen = ask(client).json()["generator"]
    assert models_called(llm) == [BIG]
    assert gen["fallback_reason"] == "unsafe_sql"


def test_no_fallback_models_configured(make_client, llm):
    client = make_client(llm_fallback_models=())
    llm.by_model[BIG] = QUOTA
    gen = ask(client).json()["generator"]
    assert models_called(llm) == [BIG] and llm.calls[0]["max_retries"] is None
    assert gen["used"] == "rule_based" and gen["fallback_reason"] == "provider_quota"


def test_fallback_model_gets_the_same_prompt_including_glossary_definitions(client, llm):
    llm.by_model[BIG] = QUOTA
    body = ask(client, "what is our ARR?", dataset_id="saas").json()
    assert body["generator"]["llm_strategy"] == "glossary_rag"
    assert llm.calls[0]["messages"] == llm.calls[1]["messages"]
    assert "ARR" in llm.calls[1]["messages"][-1]["content"]


def test_a_failed_over_query_counts_once_against_the_daily_budget(make_client, llm):
    client = make_client(llm_daily_budget=1)
    llm.by_model[BIG] = QUOTA
    assert ask(client).json()["generator"]["model"] == SMALL
    assert ask(client).json()["generator"]["fallback_reason"] == "daily_budget"
