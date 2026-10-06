"""One question -> intent -> SQL (LLM and/or rule-based) -> safe execution -> chart suggestion.

The LLM step tries a chain of models on one provider: the primary (LLM_MODEL), then each of
LLM_FALLBACK_MODELS when the previous one is out of quota or rate-limited. Any other failure (an API error,
SQL the safety layer rejects) goes straight to the rule-based generator, since another model wouldn't help.

Generation, retrieval and execution all reuse src/: rag_sql.generate_sql_detailed (Groq or Cerebras),
doc_retrieval.match_terms (via backend.glossary), rag_sql.retrieve_examples, sql_builder.build_sql and
sql_safety.run_safe_query.
"""
import json
import logging
import sqlite3
import time
from dataclasses import dataclass

import rag_sql
from intent import guess_intent_by_keywords
from sql_builder import build_sql
from sql_safety import MAX_RESULT_ROWS, QUERY_TIMEOUT_S, QueryTimeoutError, UnsafeSQLError, run_safe_query

from backend.chart import suggest_chart
from backend.config import Settings
from backend.glossary import matched_definitions
from backend.ratelimit import DailyBudget, SlidingWindowLimiter
from backend.sessions import Session

log = logging.getLogger("backend.query")

# Why "auto" fell back to the rule-based generator. The messages are fixed text, never provider errors.
FALLBACK_MESSAGES = {
    "daily_budget": "The app's daily LLM budget is used up, so the rule-based generator answered.",
    "rate_limited": "Too many LLM questions from your address in the last minute, so the rule-based generator answered.",
    "llm_unavailable": "No LLM is configured on the server, so the rule-based generator answered.",
    "provider_quota": "The LLM provider's daily quota is exhausted, so the rule-based generator answered.",
    "provider_rate_limited": "The LLM provider is rate-limiting requests, so the rule-based generator answered.",
    "llm_error": "The LLM call failed, so the rule-based generator answered.",
    "unsafe_sql": "The LLM's SQL was rejected by the safety check ({detail}), so the rule-based generator answered.",
    "execution_error": "The LLM's SQL failed to run ({detail}), so the rule-based generator answered.",
}


class QueryError(Exception):
    """A request that can't be answered. `message` is safe to show to the user."""

    def __init__(self, status: int, code: str, message: str, retry_after: int = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after


@dataclass
class _Generated:
    sql: str
    used: str                    # "llm" or "rule_based"
    intent: str                  # the rule-based builder may downgrade the intent (e.g. to "filter")
    llm_strategy: str = None
    fallback_reason: str = None
    fallback_detail: str = None
    rejected_sql: str = None
    model: str = None            # the model whose SQL is used (LLM answers only)
    model_note: str = None       # e.g. "gpt-oss-20b (120b quota exhausted)" when a fallback model answered
    models_tried: list = None    # [{"model", "outcome"}] in the order tried


MODEL_OUTCOMES = {"quota": "quota_exhausted", "rate_limit": "rate_limited"}
OUTCOME_TEXT = {"quota_exhausted": "quota exhausted", "rate_limited": "rate-limited"}


def _short(model: str) -> str:
    return model.split("/")[-1]


def model_note(primary: str, answered: str, first_outcome: str) -> str:
    """'gpt-oss-20b (120b quota exhausted)': who answered, and why the primary didn't."""
    label = _short(primary)
    prefix = "gpt-oss-"
    if label.startswith(prefix) and _short(answered).startswith(prefix):
        label = label[len(prefix):]
    return f"{_short(answered)} ({label} {OUTCOME_TEXT.get(first_outcome, 'failed')})"


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def _execution_message(e: Exception) -> str:
    if isinstance(e, QueryTimeoutError):
        return f"query took longer than {QUERY_TIMEOUT_S:g} s"
    if isinstance(e, UnsafeSQLError):
        return f"rejected by the safety check: {e}"
    return str(e).split("\n")[0][:200]   # SQLite's own message, e.g. "no such column: Foo"


class QueryService:
    def __init__(self, settings: Settings, intent_classifier=None):
        self.settings = settings
        self.intent_classifier = intent_classifier
        self.ip_limiter = SlidingWindowLimiter(settings.llm_rate_limit_per_min, 60.0)
        self.budget = DailyBudget(settings.llm_daily_budget)
        rag_sql.MAX_LLM_RETRIES = settings.llm_max_retries   # an API request shouldn't back off for minutes

    # --- intent --------------------------------------------------------------
    def predict_intent(self, question: str) -> dict:
        if self.intent_classifier is not None:
            try:
                top = self.intent_classifier(question)[0]
                return {"label": top["label"], "confidence": round(float(top["score"]), 4), "source": "bert"}
            except Exception:
                log.exception("intent classifier failed; using keywords")
        return {"label": guess_intent_by_keywords(question), "confidence": None, "source": "keywords"}

    # --- LLM gate ------------------------------------------------------------
    def _llm_gate(self, client_ip: str):
        """None if an LLM call may go ahead (and is counted), else (code, retry_after_s)."""
        if not self.settings.llm_enabled:
            return "llm_unavailable", None
        allowed, retry_after = self.ip_limiter.hit(client_ip)
        if not allowed:
            return "rate_limited", retry_after
        if not self.budget.try_consume():
            return "daily_budget", None
        return None

    # --- generation ----------------------------------------------------------
    def _rule_based(self, question: str, intent: str, session: Session, reason: str = None,
                    detail: str = None, rejected_sql: str = None) -> _Generated:
        used_intent, sql = build_sql(question, intent, session.dataset)
        sql = "\n".join(line.strip() for line in sql.strip().splitlines() if line.strip())  # drop the template indent
        return _Generated(sql, "rule_based", used_intent, fallback_reason=reason,
                          fallback_detail=detail, rejected_sql=rejected_sql)

    def _fallback(self, question, intent, session, reason, detail=None, rejected_sql=None) -> _Generated:
        message = FALLBACK_MESSAGES[reason].format(detail=detail or "")
        return self._rule_based(question, intent, session, reason, message, rejected_sql)

    def run(self, session: Session, question: str, mode: str, use_glossary: bool, client_ip: str) -> dict:
        t_start = time.perf_counter()
        latency = {"intent": 0, "retrieval": 0, "generation": 0, "execution": 0}
        notes = []
        context = {"glossary_checked": False, "glossary": [], "examples": []}

        t = time.perf_counter()
        intent = self.predict_intent(question)
        latency["intent"] = _ms(t)

        if use_glossary and not session.glossary:
            notes.append("Glossary definitions exist only for the built-in sample datasets, so none were used.")
        glossary = session.glossary if use_glossary else None
        if glossary and mode == "rule_based" and matched_definitions(question, glossary):
            notes.append("The question uses a glossary term, but the rule-based generator doesn't use definitions.")

        if mode == "rule_based":
            t = time.perf_counter()
            gen = self._rule_based(question, intent["label"], session)
            latency["generation"] = _ms(t)
        else:
            gate = self._llm_gate(client_ip)
            if gate is not None:
                code, retry_after = gate
                if mode == "llm":
                    raise self._gate_error(code, retry_after)
                gen = self._fallback(question, intent["label"], session, code)
            else:
                gen = self._generate_llm(question, intent["label"], session, glossary, mode, context, latency)

        t = time.perf_counter()
        try:
            df = run_safe_query(gen.sql, session.dataset.conn, max_rows=MAX_RESULT_ROWS)
        except (UnsafeSQLError, QueryTimeoutError, sqlite3.Error) as e:
            if gen.used == "llm" and mode == "auto":
                detail = _execution_message(e)
                strategy, tried = gen.llm_strategy, gen.models_tried
                gen = self._fallback(question, intent["label"], session, "execution_error", detail, gen.sql)
                gen.llm_strategy, gen.models_tried = strategy, tried
                try:
                    df = run_safe_query(gen.sql, session.dataset.conn, max_rows=MAX_RESULT_ROWS)
                except (UnsafeSQLError, QueryTimeoutError, sqlite3.Error) as e2:
                    raise QueryError(422, "query_failed", f"The query couldn't be run: {_execution_message(e2)}.")
            else:
                raise QueryError(422, "query_failed", f"The query couldn't be run: {_execution_message(e)}.")
        latency["execution"] = _ms(t)

        truncated = bool(df.attrs.get("truncated")) or len(df) > self.settings.max_rows_returned
        shown = df.head(self.settings.max_rows_returned)
        shown = shown.apply(lambda col: col.map(lambda v: f"<{len(v):,} bytes>" if isinstance(v, bytes) else v)
                            if col.dtype == object else col)   # blobs: describe, don't dump
        payload = json.loads(shown.to_json(orient="split", index=False, date_format="iso"))
        columns = [{"name": c, "type": "number" if shown[c].dtype.kind in "iuf" else "text"} for c in shown.columns]
        latency["total"] = _ms(t_start)

        log.info("query mode=%s used=%s fallback=%s rows=%d total_ms=%d",
                 mode, gen.used, gen.fallback_reason, len(df), latency["total"])
        return {
            "question": question,
            "intent": intent,
            "generator": {
                "used": gen.used,
                "requested_mode": mode,
                "llm_strategy": gen.llm_strategy,
                "model": gen.model if gen.used == "llm" else None,
                "model_note": gen.model_note if gen.used == "llm" else None,
                "models_tried": gen.models_tried or [],
                "provider": self.settings.llm_provider if gen.used == "llm" else None,
                "fallback_reason": gen.fallback_reason,
                "fallback_detail": gen.fallback_detail,
                "rejected_sql": gen.rejected_sql,
            },
            "sql": gen.sql,
            "columns": columns,
            "rows": payload["data"],
            "row_count": len(df),
            "truncated": truncated,
            "chart": suggest_chart(gen.intent if gen.used == "rule_based" else intent["label"], df, gen.sql),
            "context": context,
            "latency_ms": latency,
            "notes": notes,
        }

    def _generate_llm(self, question, intent, session, glossary, mode, context, latency) -> _Generated:
        docs = []
        if glossary:
            t = time.perf_counter()
            docs = matched_definitions(question, glossary)
            latency["retrieval"] = _ms(t)
            context["glossary_checked"] = True
            context["glossary"] = [{"id": d["id"], "term": d["term"], "matched": d["matched"],
                                    "definition": d["definition"]} for d in docs]
        if docs:   # no matched term: no definitions, so the prompt is the plain zero-shot (or example) one
            strategy, rag_mode = "glossary_rag", "llm_doc_rag"
        elif self.settings.llm_strategy == "example_rag":
            strategy, rag_mode = "example_rag", "llm_rag"
        else:
            strategy, rag_mode = "zero_shot", "llm_zero_shot"

        chain = (self.settings.llm_model,) + tuple(
            m for m in self.settings.llm_fallback_models if m != self.settings.llm_model)
        tried = []
        t = time.perf_counter()
        for i, model in enumerate(chain):
            last = i == len(chain) - 1
            # Only the last model waits out a per-minute limit; earlier ones fail over on the first 429.
            result = rag_sql.generate_sql_detailed(
                question, session.dataset, mode=rag_mode, intent=intent, schema_context=session.schema_context,
                dataset_name=glossary, docs=docs or None, model=model, provider=self.settings.llm_provider,
                max_retries=None if last else 1)
            if result.source == "llm":
                tried.append({"model": model, "outcome": "answered"})
                break
            if result.rejected:
                tried.append({"model": model, "outcome": "rejected"})
                break
            outcome = MODEL_OUTCOMES.get(result.error_kind, "failed")
            tried.append({"model": model, "outcome": outcome})
            log.warning("LLM %s failed (%s): %s", model, outcome, result.error)   # raw errors stay in the server log
            if outcome == "failed":
                break
        latency["generation"] = _ms(t)
        if result.examples:
            context["examples"] = [{"question": e["question"], "sql": e.get("table_sql", e["sql"]),
                                    "score": round(e["score"], 3)} for e in result.examples]

        if result.source == "llm":
            note = model_note(chain[0], model, tried[0]["outcome"]) if model != chain[0] else None
            return _Generated(result.sql, "llm", intent, llm_strategy=strategy, model=model, model_note=note,
                              models_tried=tried)

        if result.rejected:
            reason, detail = "unsafe_sql", result.error.split(": ", 1)[-1]
        elif tried[-1]["outcome"] == "quota_exhausted":
            reason, detail = "provider_quota", None
        elif tried[-1]["outcome"] == "rate_limited":
            reason, detail = "provider_rate_limited", None
        else:
            reason, detail = "llm_error", None
        if mode == "llm":
            raise self._llm_failure(reason, detail)
        gen = self._fallback(question, intent, session, reason, detail, result.llm_sql if result.rejected else None)
        gen.llm_strategy, gen.models_tried = strategy, tried
        return gen

    @staticmethod
    def _gate_error(code: str, retry_after) -> QueryError:
        if code == "rate_limited":
            return QueryError(429, code, f"Too many LLM questions. Try again in {retry_after} s.", retry_after)
        if code == "daily_budget":
            return QueryError(429, code, "The daily LLM budget is used up. Use auto or rule-based mode, or try again tomorrow.")
        return QueryError(503, code, "No LLM is configured on the server. Use auto or rule-based mode.")

    @staticmethod
    def _llm_failure(reason: str, detail: str) -> QueryError:
        if reason == "unsafe_sql":
            return QueryError(422, reason, f"The LLM's SQL was rejected by the safety check ({detail}).")
        if reason == "provider_quota":
            return QueryError(429, reason, "The LLM provider's daily quota is exhausted. Try auto or rule-based mode.")
        if reason == "provider_rate_limited":
            return QueryError(429, reason, "The LLM provider is rate-limiting requests. Try again shortly, "
                                           "or use auto or rule-based mode.")
        return QueryError(502, reason, "The LLM call failed. Try again, or use auto or rule-based mode.")
