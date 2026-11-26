"""Tests for declarative fleet model postures (awpool.posture & CLI)."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from awpool.cli import main as cli_main
from awpool.models import ModelState, PeerInfo
from awpool.posture import (
    BUILTIN_POSTURES,
    apply_posture,
    load_postures,
    plan_posture,
    read_active_posture,
)


def test_builtin_postures_present() -> None:
    postures = load_postures(path="/nonexistent/postures.json")
    for name in ("fleet-multi", "heavy-moe", "creator", "quiet"):
        assert name in postures
        assert postures[name].description


def test_plan_parks_before_warming_and_skips_already_warm() -> None:
    peers = [
        PeerInfo(
            name="dgx-spark",
            reachable=True,
            models={
                "deepseek-v4-flash": ModelState.WARM,
                "bonsai2.5-27b": ModelState.WARM,
                "qwen3.8-27b": ModelState.COLD,
                "inkling-2-mini": ModelState.COLD,
                "aither-code-embed": ModelState.COLD,
            },
        ),
        PeerInfo(
            name="local",
            reachable=True,
            models={
                "bonsai2.5-27b": ModelState.WARM,
                "bonsai2-4b": ModelState.COLD,
                "deepseek-v4-flash": ModelState.COLD,
            },
        ),
    ]
    plan = plan_posture(BUILTIN_POSTURES["fleet-multi"], peers)
    actions = [(a.peer, a.action, a.model) for a in plan.actions]
    # Park must come first so memory is freed before warming new models.
    assert actions[0] == ("dgx-spark", "park", "deepseek-v4-flash")
    # Already-warm models are skipped, already-cold parks are skipped.
    assert ("dgx-spark", "warm", "bonsai2.5-27b") not in actions
    assert ("local", "park", "deepseek-v4-flash") not in actions
    assert ("dgx-spark", "warm", "qwen3.8-27b") in actions
    assert ("dgx-spark", "warm", "inkling-2-mini") in actions
    assert ("local", "warm", "bonsai2-4b") in actions
    assert "coder=qwen3.8-27b" in plan.router_env["AITHER_MODEL_STANDINS"]


def test_quiet_posture_parks_all_warm_models_via_wildcard() -> None:
    peers = [
        PeerInfo(
            name="dgx-spark",
            reachable=True,
            models={"bonsai2.5-27b": ModelState.WARM, "qwen3.8-27b": ModelState.WARM},
        ),
        PeerInfo(name="offline-box", reachable=False),
    ]
    plan = plan_posture(BUILTIN_POSTURES["quiet"], peers)
    assert [(a.peer, a.action, a.model) for a in plan.actions] == [
        ("dgx-spark", "park", "bonsai2.5-27b"),
        ("dgx-spark", "park", "qwen3.8-27b"),
    ]
    assert any("offline-box" in n for n in plan.notes)


def test_apply_posture_hits_real_loopback_peer_and_persists_state(tmp_path) -> None:
    received: list[tuple[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            received.append((self.path, body["model"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, fmt: str, *args) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        peers = [
            PeerInfo(
                name="dgx-spark",
                reachable=True,
                models={
                    "bonsai2.5-27b": ModelState.WARM,
                    "qwen3.8-27b": ModelState.COLD,
                    "inkling-2-mini": ModelState.COLD,
                    "deepseek-v4-flash": ModelState.COLD,
                },
            )
        ]
        plan = plan_posture(BUILTIN_POSTURES["heavy-moe"], peers)
        state_file = tmp_path / "active-posture.json"
        env_file = tmp_path / "router-posture.env"
        outcome = apply_posture(
            plan,
            {"dgx-spark": f"http://127.0.0.1:{port}"},
            state_path=state_file,
            router_env_path=env_file,
        )
        assert outcome["ok"] is True
        # Park Bonsai first, then warm DeepSeek.
        assert received == [
            ("/v1/models/bonsai2.5-27b/park", "bonsai2.5-27b"),
            ("/v1/models/deepseek-v4-flash/warm", "deepseek-v4-flash"),
        ]
        saved = read_active_posture(state_file)
        assert saved is not None and saved["posture"] == "heavy-moe"
        env_text = env_file.read_text(encoding="utf-8")
        assert "export AWPOOL_POSTURE='heavy-moe'" in env_text
        assert "reasoning=deepseek-v4-flash" in env_text
    finally:
        server.shutdown()


def test_custom_postures_and_cli_roundtrip(tmp_path, capsys) -> None:
    custom_file = tmp_path / "postures.json"
    custom_file.write_text(
        json.dumps(
            {
                "coding-turbo": {
                    "description": "Custom dual-coder profile",
                    "primary_lane": "qwen3.8-27b",
                    "target_ram_gb": 32,
                    "peers": {"dgx-spark": {"warm": ["qwen3.8-27b"], "park_others": True}},
                    "router_standins": {"coder": "qwen3.8-27b"},
                }
            }
        ),
        encoding="utf-8",
    )
    state_file = tmp_path / "state.json"
    env_file = tmp_path / "router.env"

    rc = cli_main(["posture", "list", "--postures", str(custom_file), "--state-file", str(state_file), "--json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert any(r["name"] == "coding-turbo" for r in rows)

    rc = cli_main(
        [
            "posture",
            "apply",
            "coding-turbo",
            "--local-only",
            "--postures",
            str(custom_file),
            "--state-file",
            str(state_file),
            "--router-env",
            str(env_file),
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["posture"] == "coding-turbo" and out["ok"] is True

    rc = cli_main(["posture", "current", "--state-file", str(state_file), "--json"])
    assert rc == 0
    cur = json.loads(capsys.readouterr().out)
    assert cur["posture"] == "coding-turbo"
