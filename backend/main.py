"""FastAPI app: datasets (samples + CSV uploads) and natural-language queries over them.

Run from the repo root:  uvicorn backend.main:app --port 8000
(`app` is built on first access, so importing this module, e.g. in tests, doesn't load models or data;
`uvicorn backend.main:create_app --factory` works too.)
"""
import logging
import os

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

import rag_sql

from backend.config import Settings
from backend.glossary import glossary_chunks
from backend.query_service import QueryError, QueryService
from backend.ratelimit import SlidingWindowLimiter
from backend.samples import SAMPLES
from backend.schemas import QueryRequest
from backend.sessions import DatasetError, SessionStore, load_session_dataset, make_session

log = logging.getLogger("backend")
_DEFAULT = object()
MULTIPART_OVERHEAD = 64 * 1024
MAX_JSON_BODY = 16 * 1024

class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict = None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers


def _error(status: int, code: str, message: str, headers: dict = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


class BodySizeLimit:
    """Rejects bodies over the limit while they stream in, without trusting Content-Length alone.
    Uploads get the CSV limit plus multipart overhead; every other request gets MAX_JSON_BODY."""

    def __init__(self, app, max_upload_bytes: int):
        self.app = app
        self.max_upload = max_upload_bytes + MULTIPART_OVERHEAD

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = self.max_upload if scope["path"] == "/api/datasets" else MAX_JSON_BODY
        message = f"The request is too large (limit {limit // (1024 * 1024) or limit // 1024} " \
                  f"{'MB' if limit >= 1024 * 1024 else 'KB'})."
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > limit:
            return await _error(413, "too_large", message)(scope, receive, send)

        received, too_large, started = 0, False, False

        async def limited_receive():
            nonlocal received, too_large
            msg = await receive()
            if msg["type"] == "http.request":
                received += len(msg.get("body", b""))
                if received > limit:
                    too_large = True
                    return {"type": "http.disconnect"}
            return msg

        async def guarded_send(msg):
            nonlocal started
            if too_large:
                return          # the app's own (error) response is replaced by a 413 below
            started = True
            await send(msg)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not too_large:
                raise
        if too_large and not started:
            await _error(413, "too_large", message)(scope, receive, send)


def client_ip(request: Request, trusted_hops: int) -> str:
    """The address rate limits are keyed on. Behind `trusted_hops` reverse proxies, each proxy appends the
    address it received the request from to X-Forwarded-For, so the client is `trusted_hops` entries from the
    right. Entries further left were written by the client and could be anything, so they are never used; a
    header shorter than the proxy chain means it wasn't set by the proxies, so the socket peer is used."""
    peer = request.client.host if request.client else "unknown"
    if trusted_hops <= 0:
        return peer
    chain = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",") if part.strip()]
    if len(chain) < trusted_hops:
        return peer
    return chain[-trusted_hops][:64]


def _mount_frontend(app: FastAPI, dist):
    """Serves the built React app from the API's origin: hashed assets under /assets (cacheable), the few
    top-level files (index.html, favicon), and index.html for any other non-API path."""
    index = dist / "index.html"
    top_level = {p.name for p in dist.iterdir() if p.is_file()}
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path == "api" or path.startswith("api/"):
            raise StarletteHTTPException(404, "Not Found")
        return FileResponse(dist / path if path in top_level else index)


def _quiet_library_logs():
    """HTTP clients log every request at INFO, which doesn't belong in an API server's log."""
    for name in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers", "datasets"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _load_intent_classifier():
    """The BERT intent model: onnxruntime + the int8 ONNX export by default (INTENT_RUNTIME=torch uses the
    transformers pipeline instead, if requirements-eval.txt is installed)."""
    from intent import load_intent_classifier

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # download bars don't belong in a server log
    try:
        from transformers.utils import logging as hf_logging   # only present with the torch runtime
        hf_logging.disable_progress_bar()
    except ImportError:
        pass
    return load_intent_classifier()


def _preload_retrieval_models(settings: Settings):
    """Glossary gating is plain text matching, so by default no embedding model or FAISS index is loaded.
    Only LLM_STRATEGY=example_rag needs them (sentence-transformers + faiss, from requirements-eval.txt)."""
    if settings.llm_strategy == "example_rag":
        try:
            rag_sql.get_retriever()
        except ImportError as e:
            raise RuntimeError("LLM_STRATEGY=example_rag needs sentence-transformers and faiss-cpu "
                               "(pip install -r requirements-eval.txt)") from e


def _sample_payload(session) -> dict:
    return {
        "dataset_id": session.id,
        "name": session.name,
        "description": session.description,
        "rows": len(session.dataset.df),
        "has_glossary": session.glossary is not None,
        "example_questions": session.example_questions,
        "schema": session.schema,
    }


def create_app(settings: Settings = None, intent_classifier=_DEFAULT) -> FastAPI:
    """Builds the app and loads everything once: samples, their glossaries, the intent classifier and,
    for example_rag (unless PRELOAD_MODELS is off), the embedding model."""
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    _quiet_library_logs()
    settings = settings or Settings.from_env()
    if intent_classifier is _DEFAULT:
        intent_classifier = _load_intent_classifier() if settings.preload_models else None
    if settings.preload_models:
        _preload_retrieval_models(settings)
    for spec in SAMPLES.values():
        glossary_chunks(spec["glossary"])   # parse each glossary once; a broken file fails at startup

    store = SessionStore(settings.max_sessions, settings.session_ttl_s)
    for sample_id, spec in SAMPLES.items():
        dataset = load_session_dataset(spec["path"])
        store.add_sample(make_session(
            dataset, spec["name"], sample_id, glossary=spec["glossary"], description=spec["description"],
            example_questions=[{"question": q, "needs_glossary": g} for q, g in spec["examples"]]))

    service = QueryService(settings, intent_classifier)
    upload_limiter = SlidingWindowLimiter(settings.upload_rate_limit_per_min, 60.0)

    app = FastAPI(title="NL to SQL Data Assistant", version="4.0")
    app.state.settings, app.state.store, app.state.service = settings, store, service

    app.add_middleware(BodySizeLimit, max_upload_bytes=settings.max_upload_bytes)
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.frontend_origins), allow_credentials=False,
        allow_methods=["GET", "POST"], allow_headers=["Content-Type"], max_age=600)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        # Vite's hashed asset names change with their content, so they can be cached for good.
        immutable = request.url.path.startswith("/assets/") and response.status_code == 200
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable" if immutable else "no-store"
        return response

    # --- errors: one JSON shape, never a stack trace -----------------------------------------
    @app.exception_handler(ApiError)
    async def _api_error(request, exc: ApiError):
        return _error(exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(QueryError)
    async def _query_error(request, exc: QueryError):
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return _error(exc.status, exc.code, exc.message, headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc: RequestValidationError):
        problems = []
        for err in exc.errors()[:5]:
            field = ".".join(str(p) for p in err.get("loc", []) if p != "body")
            problems.append(f"{field}: {err.get('msg', 'invalid')}" if field else err.get("msg", "invalid"))
        return _error(422, "invalid_request", "; ".join(problems) or "Invalid request.")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return _error(exc.status_code, code, message)

    @app.exception_handler(Exception)
    async def _unexpected(request, exc: Exception):
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return _error(500, "internal_error", "Something went wrong on the server.")

    # --- endpoints ---------------------------------------------------------------------------
    @app.get("/api/_debug/client", include_in_schema=False)
    def debug_client(request: Request):
        # TEMPORARY: shows the caller only their own request's forwarding chain, to set TRUSTED_PROXY_HOPS.
        chain = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        return {"trusted_proxy_hops": settings.trusted_proxy_hops, "xff": chain,
                "cf_connecting_ip": request.headers.get("cf-connecting-ip"),
                "true_client_ip": request.headers.get("true-client-ip"),
                "peer": request.client.host if request.client else None,
                "keyed_on": client_ip(request, settings.trusted_proxy_hops)}

    @app.get("/api/health")
    def health():
        return {"status": "ok", "intent_model": "bert" if service.intent_classifier else "keywords",
                "llm_configured": settings.llm_enabled}

    @app.get("/api/config")
    def config():
        return {
            "default_mode": settings.default_mode,
            "glossary_default": settings.glossary_default,
            "llm_provider": settings.llm_provider if settings.llm_enabled else None,
            "llm_model": settings.llm_model if settings.llm_enabled else None,
            "llm_fallback_models": list(settings.llm_fallback_models) if settings.llm_enabled else [],
            "llm_budget_remaining": service.budget.remaining,
            "max_upload_mb": settings.max_upload_bytes // (1024 * 1024),
            "max_rows_returned": settings.max_rows_returned,
        }

    @app.get("/api/samples")
    def samples():
        return {"samples": [_sample_payload(s) for s in store.samples()]}

    @app.post("/api/datasets", status_code=201)
    async def upload_dataset(request: Request, file: UploadFile = File(...)):
        allowed, retry_after = upload_limiter.hit(client_ip(request, settings.trusted_proxy_hops))
        if not allowed:
            raise ApiError(429, "rate_limited", f"Too many uploads. Try again in {retry_after} s.",
                           {"Retry-After": str(retry_after)})
        name = os.path.basename((file.filename or "upload.csv").replace("\\", "/"))
        name = "".join(ch for ch in name if ch.isprintable())[:100] or "upload.csv"
        if not name.lower().endswith(".csv"):
            raise ApiError(415, "unsupported_type", "Only .csv files are supported.")
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise ApiError(413, "too_large", f"The file is larger than {settings.max_upload_bytes // (1024 * 1024)} MB.")
        if not data.strip():
            raise ApiError(422, "invalid_csv", "The file is empty.")
        try:
            dataset = await run_in_threadpool(load_session_dataset, data, settings.max_columns,
                                                settings.max_rows)
            session = await run_in_threadpool(make_session, dataset, name)
        except DatasetError as e:
            raise ApiError(422, "invalid_csv", str(e))
        store.add_upload(session)
        return {**_sample_payload(session), "expires_in_s": settings.session_ttl_s}

    @app.post("/api/query")
    def query(body: QueryRequest, request: Request):
        session = store.get(body.dataset_id)
        if session is None:
            raise ApiError(404, "dataset_not_found",
                           "That dataset isn't loaded (uploads expire after inactivity). Upload it again.")
        mode = body.mode or settings.default_mode
        if body.use_glossary is None:   # the server default only applies where there is a glossary
            use_glossary = settings.glossary_default and session.glossary is not None
        else:
            use_glossary = body.use_glossary
        return service.run(session, body.question, mode, use_glossary,
                           client_ip(request, settings.trusted_proxy_hops))

    if settings.frontend_dist is not None:   # registered last, so every /api route matches first
        _mount_frontend(app, settings.frontend_dist)
    return app



_app = None


def __getattr__(name):
    """`backend.main:app` for `uvicorn backend.main:app`: created on first access (PEP 562), from the
    environment, so a plain import stays cheap."""
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
