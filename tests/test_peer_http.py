"""Request-builder and best-effort sender tests — stdlib urllib only.

The builders are pure (no I/O): their shape is asserted directly. The
senders are tested against a monkeypatched `urllib.request.urlopen`, so
"peer refused", "peer asleep" and "peer answered garbage" are all exercised
without a network — and none of them may raise.
"""

# isort: skip_file
#
# The quality gate runs ruff `--isolated` from the repo root, where `awpool`
# reads as THIRD-party and belongs in the same block as pytest; a dev running
# ruff from inside the package dir resolves the package root and treats
# `awpool` as FIRST-party, which wants its own block. Those two layouts are
# mutually exclusive — one block or two — so no ordering satisfies both and
# this states the conflict instead of flip-flopping the file every time
# whichever ruff ran last disagreed. Same fix as awrelay/tests/test_standalone.py.

import json
import urllib.error
import urllib.request

import pytest
from awpool.peer_http import (
    build_park_request,
    build_request,
    build_warm_request,
    fetch_peer_models,
    send_request,
)


class _FakeResponse:
    """Minimal stand-in for http.client.HTTPResponse (status + context manager)."""

    def __init__(self, status: int = 200, body: bytes = b"{}"):
        self.status = status
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def test_build_warm_request_shape():
    req = build_warm_request("desktop.local:8080", "qwen3-8b")
    assert req["method"] == "POST"
    assert req["url"] == "http://desktop.local:8080/v1/models/qwen3-8b/warm"
    assert json.loads(req["body"]) == {"model": "qwen3-8b"}  # body keeps the raw id
    assert req["headers"]["Content-Type"] == "application/json"
    assert req["timeout"] == 5.0


def test_build_park_request_shape():
    req = build_park_request("desktop.local:8080", "qwen3-8b")
    assert req["url"] == "http://desktop.local:8080/v1/models/qwen3-8b/park"
    assert req["method"] == "POST"


def test_build_request_accepts_full_url_endpoint():
    req = build_warm_request("https://desktop.local:8443", "qwen3-8b")
    assert req["url"] == "https://desktop.local:8443/v1/models/qwen3-8b/warm"


def test_build_request_escapes_model_id():
    # Model ids can contain '/' (e.g. "llama-3.2/8b"); an unescaped one
    # would rewrite the URL path and warm the wrong route.
    req = build_warm_request("desktop.local:8080", "llama-3.2/8b")
    assert "/v1/models/llama-3.2%2F8b/warm" in req["url"]
    assert "/llama-3.2/8b/" not in req["url"]


def test_build_request_rejects_unknown_action():
    with pytest.raises(ValueError):
        build_request("hibernate", "desktop.local:8080", "qwen3-8b")


def test_send_request_ok_on_2xx(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["timeout"] = timeout
        return _FakeResponse(status=200)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = send_request(build_warm_request("desktop.local:8080", "qwen3-8b"))
    assert result["ok"] is True
    assert result["status"] == 200
    assert result["error"] is None
    assert seen["method"] == "POST"
    assert seen["url"].endswith("/v1/models/qwen3-8b/warm")


def test_send_request_reports_http_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 503, "nope", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = send_request(build_warm_request("desktop.local:8080", "qwen3-8b"))
    assert result["ok"] is False
    assert result["status"] == 503
    assert result["error"]


def test_send_request_reports_network_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("peer asleep")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = send_request(build_warm_request("desktop.local:8080", "qwen3-8b"))
    assert result["ok"] is False
    assert result["status"] is None
    assert "peer asleep" in result["error"]


def test_fetch_peer_models_returns_payload(monkeypatch):
    body = json.dumps({"models": [{"id": "qwen3-8b", "state": "warm"}]}).encode("utf-8")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(200, body),
    )
    payload, err = fetch_peer_models("desktop.local:8080")
    assert err is None
    assert payload["models"][0]["id"] == "qwen3-8b"


def test_fetch_peer_models_non_json_is_an_error(monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(200, b"<html>oops</html>"),
    )
    payload, err = fetch_peer_models("desktop.local:8080")
    assert payload is None
    assert err is not None and "non-JSON" in err


def test_fetch_peer_models_http_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "missing", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    payload, err = fetch_peer_models("desktop.local:8080")
    assert payload is None
    assert err is not None and "404" in err
