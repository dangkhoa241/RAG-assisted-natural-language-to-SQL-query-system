"""Production behaviour: the client IP behind a reverse proxy, and serving the built frontend from the API's origin."""
import pytest
from starlette.requests import Request

from backend.main import client_ip


def request_from(peer: str, forwarded: str = None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded is not None else []
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers, "client": (peer, 1234)})


@pytest.mark.parametrize("hops, forwarded, expected", [
    (0, "203.0.113.9", "10.0.0.2"),                       # not behind a proxy: the header is ignored
    (1, None, "10.0.0.2"),                                # no header: the socket peer
    (1, "203.0.113.9", "203.0.113.9"),                    # one proxy appended the client
    (1, "6.6.6.6, 203.0.113.9", "203.0.113.9"),           # the client's own forged entry is skipped
    (2, "6.6.6.6, 203.0.113.9, 10.1.1.1", "203.0.113.9"),
    (2, "203.0.113.9", "10.0.0.2"),                       # shorter than the proxy chain: not set by the proxies
])
def test_client_ip_trusts_only_proxy_appended_entries(hops, forwarded, expected):
    assert client_ip(request_from("10.0.0.2", forwarded), hops) == expected


def test_rate_limit_is_per_forwarded_client_behind_the_proxy(make_client, llm):
    client = make_client(llm_rate_limit_per_min=1, trusted_proxy_hops=1)

    def ask(forwarded):
        body = {"dataset_id": "healthcare", "question": "how many patients by gender"}
        return client.post("/api/query", json=body, headers={"X-Forwarded-For": forwarded}).json()

    assert ask("203.0.113.9")["generator"]["used"] == "llm"
    assert ask("198.51.100.4")["generator"]["used"] == "llm"                     # another client, own bucket
    assert ask("1.2.3.4, 203.0.113.9")["generator"]["fallback_reason"] == "rate_limited"   # forging doesn't help


@pytest.fixture
def dist(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (tmp_path / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "assets" / "index-abc123.js").write_text("console.log(1)", encoding="utf-8")
    return tmp_path


def test_serves_the_built_frontend_and_keeps_api_errors_as_json(make_client, dist):
    client = make_client(frontend_dist=dist)
    root = client.get("/")
    assert root.status_code == 200 and "id=root" in root.text and root.headers["cache-control"] == "no-store"
    asset = client.get("/assets/index-abc123.js")
    assert asset.status_code == 200 and "immutable" in asset.headers["cache-control"]
    assert client.get("/favicon.svg").text == "<svg/>"
    assert "id=root" in client.get("/some/client/route").text          # unknown pages get the app
    missing = client.get("/api/does-not-exist")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
    assert client.get("/api/health").json()["status"] == "ok"           # API routes still win
    # only files inside the build are served: anything else is the app shell, never a file from disk
    assert client.get("/backend/config.py").text == root.text
    assert client.get("/assets/%2e%2e/%2e%2e/backend/config.py").status_code == 404


def test_api_only_when_no_frontend_build(client):
    assert client.get("/").status_code == 404
