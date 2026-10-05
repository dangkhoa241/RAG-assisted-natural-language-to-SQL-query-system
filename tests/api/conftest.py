"""API test fixtures. The LLM, the intent model and the glossary embeddings are all faked:
no test makes a network call or loads a model."""
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import rag_sql  # noqa: E402  (src/ is on the path via tests/conftest.py)
from fastapi.testclient import TestClient  # noqa: E402

from backend.config import Settings  # noqa: E402
from backend.main import create_app  # noqa: E402
from intent import guess_intent_by_keywords  # noqa: E402


class FakeLLM:
    """Stands in for rag_sql.call_llm. `reply` is the SQL to return, or an exception to raise."""

    def __init__(self):
        self.reply = "SELECT `Gender`, COUNT(*) AS n FROM data GROUP BY `Gender`"
        self.calls = []

    def __call__(self, messages, cache=None, model=rag_sql.GROQ_MODEL):
        self.calls.append({"messages": messages, "model": model})
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    @property
    def last_prompt(self) -> str:
        return self.calls[-1]["messages"][-1]["content"]


def fake_intent(question):
    return [{"label": guess_intent_by_keywords(question), "score": 0.91}]


FAKE_DOCS = [{"id": "rt_repeat_customer", "term": "Repeat customer", "definition": "3 or more orders.",
              "text": "Repeat customer: 3 or more orders."}]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("tests must not create a real LLM client")
    monkeypatch.setattr(rag_sql, "_groq_client", refuse)


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(rag_sql, "call_llm", fake)
    return fake


@pytest.fixture
def docs(monkeypatch):
    calls = []

    def fake_retrieve(question, dataset_name, k=3, method="hybrid"):
        calls.append((question, dataset_name))
        return FAKE_DOCS
    monkeypatch.setattr(rag_sql, "retrieve_docs", fake_retrieve)
    return calls


def make_settings(**overrides) -> Settings:
    base = dict(preload_models=False, llm_enabled=True, llm_rate_limit_per_min=100, llm_daily_budget=1000,
                upload_rate_limit_per_min=100, frontend_origins=("http://localhost:5173",))
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def make_client(llm, docs):
    def _make(**overrides):
        app = create_app(make_settings(**overrides), intent_classifier=fake_intent)
        return TestClient(app, raise_server_exceptions=False)
    return _make


@pytest.fixture
def client(make_client):
    return make_client()

