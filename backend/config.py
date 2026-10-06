"""Settings, read from environment variables (and .env) so defaults can change without code edits.

Every variable is optional:
  DEFAULT_MODE              auto | llm | rule_based                     (auto)
  GLOSSARY_DEFAULT          whether the glossary toggle starts on       (true)
  LLM_STRATEGY              zero_shot | example_rag, when no glossary term matches (zero_shot)
  LLM_PROVIDER              groq | cerebras                             (groq)
  LLM_MODEL                 the provider's model ID                     (groq: openai/gpt-oss-20b,
                                                                         cerebras: gpt-oss-120b)
  LLM_RATE_LIMIT_PER_MIN    LLM queries per client IP per minute        (10)
  LLM_DAILY_BUDGET          LLM queries per UTC day, all clients        (500)
  LLM_MAX_RETRIES           attempts per LLM call on rate limits         (2)
  UPLOAD_RATE_LIMIT_PER_MIN uploads per client IP per minute            (10)
  MAX_UPLOAD_MB             largest accepted CSV                        (10)
  MAX_COLUMNS               widest accepted CSV                         (100)
  MAX_ROWS                  longest accepted CSV                        (200000)
  MAX_SESSIONS              uploaded datasets kept in memory            (20)
  SESSION_TTL_MIN           idle minutes before an upload is dropped    (30)
  MAX_ROWS_RETURNED         result rows sent to the client              (500)
  FRONTEND_ORIGINS          comma-separated CORS origins                (http://localhost:5173)
  PRELOAD_MODELS            load the intent model (and, for example_rag, the embedding model) at startup (true)

The provider's API key comes from GROQ_API_KEY or CEREBRAS_API_KEY.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from rag_sql import PROVIDER_API_KEYS

from backend import ROOT_DIR

load_dotenv(ROOT_DIR / ".env")

MODES = ("auto", "llm", "rule_based")
LLM_STRATEGIES = ("zero_shot", "example_rag")
PROVIDERS = tuple(PROVIDER_API_KEYS)
# Stage 3 found gpt-oss-20b matches gpt-oss-120b once the right definitions are in the prompt.
# Cerebras doesn't serve the 20b, so its default is the 120b.
DEFAULT_MODELS = {"groq": "openai/gpt-oss-20b", "cerebras": "gpt-oss-120b"}


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw)


def _choice(name: str, default: str, allowed: tuple) -> str:
    value = os.environ.get(name, default).strip()
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value


@dataclass
class Settings:
    default_mode: str = "auto"
    glossary_default: bool = True
    llm_strategy: str = "zero_shot"
    llm_provider: str = "groq"
    llm_model: str = DEFAULT_MODELS["groq"]
    llm_rate_limit_per_min: int = 10
    llm_daily_budget: int = 500
    llm_max_retries: int = 2
    upload_rate_limit_per_min: int = 10
    max_upload_bytes: int = 10 * 1024 * 1024
    max_columns: int = 100
    max_rows: int = 200_000
    max_sessions: int = 20
    session_ttl_s: int = 30 * 60
    max_rows_returned: int = 500
    frontend_origins: tuple = ("http://localhost:5173",)
    preload_models: bool = True
    llm_enabled: bool = True     # False when the provider's API key isn't set: auto mode then always uses rule_based

    @classmethod
    def from_env(cls) -> "Settings":
        origins = os.environ.get("FRONTEND_ORIGINS", "http://localhost:5173")
        provider = _choice("LLM_PROVIDER", "groq", PROVIDERS)
        return cls(
            default_mode=_choice("DEFAULT_MODE", "auto", MODES),
            glossary_default=_bool("GLOSSARY_DEFAULT", True),
            llm_strategy=_choice("LLM_STRATEGY", "zero_shot", LLM_STRATEGIES),
            llm_provider=provider,
            llm_model=os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODELS[provider],
            llm_rate_limit_per_min=_int("LLM_RATE_LIMIT_PER_MIN", 10),
            llm_daily_budget=_int("LLM_DAILY_BUDGET", 500),
            llm_max_retries=_int("LLM_MAX_RETRIES", 2),
            upload_rate_limit_per_min=_int("UPLOAD_RATE_LIMIT_PER_MIN", 10),
            max_upload_bytes=_int("MAX_UPLOAD_MB", 10) * 1024 * 1024,
            max_columns=_int("MAX_COLUMNS", 100),
            max_rows=_int("MAX_ROWS", 200_000),
            max_sessions=_int("MAX_SESSIONS", 20),
            session_ttl_s=_int("SESSION_TTL_MIN", 30) * 60,
            max_rows_returned=_int("MAX_ROWS_RETURNED", 500),
            frontend_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
            preload_models=_bool("PRELOAD_MODELS", True),
            llm_enabled=bool(os.environ.get(PROVIDER_API_KEYS[provider])),
        )
