"""Best-effort HTTP to a peer's own model API. A client can only REQUEST —
it never manages the fleet, and it never crashes because a peer is asleep.

Builders (`build_warm_request` / `build_park_request`) are PURE: they turn
(peer endpoint, model) into a request dict and are unit-testable with no
network. Only `send_request` / `fetch_peer_models` touch the wire, through
the stdlib (`urllib`) — no requests, no pydantic, nothing a phone cannot
install.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

_MODEL_ACTIONS = ("warm", "park")


def _models_url(endpoint: str) -> str:
    """Base URL for a peer's model API. A bare host:port gets http://."""
    base = endpoint if endpoint.startswith(("http://", "https://")) else f"http://{endpoint}"
    return base.rstrip("/") + "/v1/models"


def build_request(action: str, endpoint: str, model: str, timeout: float = 5.0) -> dict[str, Any]:
    """Build the warm/park request for a peer — pure, unit-testable.

    Returns {method, url, body, headers, timeout}. The model id is
    URL-escaped (ids may contain '/'). Warm and park are the same shape
    with a different verb path; the body carries the UNescaped id so the
    peer does not have to unquote it.
    """
    if action not in _MODEL_ACTIONS:
        raise ValueError(f"unknown model action: {action!r}")
    escaped = urllib.parse.quote(model, safe="")
    return {
        "method": "POST",
        "url": f"{_models_url(endpoint)}/{escaped}/{action}",
        "body": json.dumps({"model": model}).encode("utf-8"),
        "headers": {"Content-Type": "application/json"},
        "timeout": timeout,
    }


def build_warm_request(endpoint: str, model: str, timeout: float = 5.0) -> dict[str, Any]:
    """Request that a peer load `model` and keep it warm."""
    return build_request("warm", endpoint, model, timeout=timeout)


def build_park_request(endpoint: str, model: str, timeout: float = 5.0) -> dict[str, Any]:
    """Request that a peer park `model` (unload it, free the GPU)."""
    return build_request("park", endpoint, model, timeout=timeout)


def send_request(request: dict[str, Any]) -> dict[str, Any]:
    """Send a request built by `build_request` — best-effort, never raises.

    Returns {ok, status, error}: ok True on any 2xx; status is the HTTP
    status when one came back, None on a network failure; error explains
    what went wrong. Warm/park are REQUESTS — a refusal or a sleeping peer
    is a reported outcome, not an exception.
    """
    try:
        req = urllib.request.Request(
            request["url"],
            data=request["body"],
            headers=request["headers"],
            method=request["method"],
        )
        with urllib.request.urlopen(req, timeout=request["timeout"]) as resp:
            return {"ok": 200 <= resp.status < 300, "status": resp.status, "error": None}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "error": str(e)}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        return {"ok": False, "status": None, "error": str(e)}


def fetch_peer_models(
    endpoint: str, timeout: float = 5.0
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """GET the peer's model catalog. Returns (payload, error).

    payload is None when the peer did not answer or answered non-JSON;
    error explains why. Parsing is SEPARATE (`models.parse_peer_models`),
    so "the peer is unreachable" and "the peer said something we cannot
    read" stay distinguishable on their way into a PeerInfo.
    """
    try:
        req = urllib.request.Request(_models_url(endpoint), method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            try:
                return json.loads(body.decode("utf-8")), None
            except (ValueError, UnicodeDecodeError) as e:
                return None, f"peer answered non-JSON: {e}"
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        return None, str(e)
