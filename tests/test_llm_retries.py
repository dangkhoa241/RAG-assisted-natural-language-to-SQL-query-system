"""rag_sql.call_llm's handling of provider 429s, with a fake Groq client (no network)."""
import groq
import httpx
import pytest

import rag_sql


def rate_limit_error(message):
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.test/v1/chat/completions"))
    return groq.RateLimitError(message, response=response, body=None)


class FakeClient:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        raise self.errors.pop(0)


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rag_sql.time, "sleep", sleeps.append)
    return sleeps


def use(monkeypatch, client):
    monkeypatch.setattr(rag_sql, "_client_for", lambda provider: client)


def test_a_daily_quota_429_raises_daily_limit_error_immediately(monkeypatch, no_sleep):
    client = FakeClient([rate_limit_error("Rate limit reached on tokens per day (TPD)")])
    use(monkeypatch, client)
    with pytest.raises(rag_sql.DailyLimitError):
        rag_sql.call_llm([{"role": "user", "content": "q"}], model="openai/gpt-oss-120b", max_retries=3)
    assert client.calls == 1 and no_sleep == []


def test_one_attempt_fails_over_on_the_first_per_minute_429_without_waiting(monkeypatch, no_sleep):
    client = FakeClient([rate_limit_error("Rate limit reached on tokens per minute (TPM)")])
    use(monkeypatch, client)
    with pytest.raises(rag_sql.RateLimitError):
        rag_sql.call_llm([{"role": "user", "content": "q"}], model="openai/gpt-oss-120b", max_retries=1)
    assert client.calls == 1 and no_sleep == []


def test_retries_back_off_between_attempts_but_not_after_the_last(monkeypatch, no_sleep):
    client = FakeClient([rate_limit_error("tokens per minute (TPM)")] * 2)
    use(monkeypatch, client)
    with pytest.raises(rag_sql.RateLimitError):
        rag_sql.call_llm([{"role": "user", "content": "q"}], model="openai/gpt-oss-20b", max_retries=2)
    assert client.calls == 2 and len(no_sleep) == 1


def test_generate_sql_detailed_reports_the_error_kind(monkeypatch, no_sleep):
    from data_context import load_dataset
    from rag_sql import ROOT_DIR

    ds = load_dataset(ROOT_DIR / "data" / "retail_sales.csv")
    use(monkeypatch, FakeClient([rate_limit_error("tokens per minute (TPM)")]))
    gen = rag_sql.generate_sql_detailed("total revenue by region", ds, mode="llm_zero_shot", max_retries=1)
    assert gen.source == "fallback" and gen.error_kind == "rate_limit"
    use(monkeypatch, FakeClient([rate_limit_error("tokens per day (TPD)")]))
    gen = rag_sql.generate_sql_detailed("total revenue by region", ds, mode="llm_zero_shot")
    assert gen.source == "fallback" and gen.error_kind == "quota"
