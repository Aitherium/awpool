"""Declarative model postures across a personal mesh pool.

A posture is a named fleet-wide model profile — e.g. ``fleet-multi`` (run
several specialist models simultaneously on high-RAM nodes while keeping the
hot interactive model on the fastest GPU), ``heavy-moe`` (park specialists and
warm a single large MoE), ``creator`` (media/audio headroom), or ``quiet``
(park background models for low-power workstation mode).

Planning is pure and deterministic: ``plan_posture`` inspects the three-valued
``PeerInfo`` snapshot (WARM / COLD / UNKNOWN) and emits ``park`` actions BEFORE
``warm`` actions so memory is freed before new weights load. Execution sends
best-effort HTTP requests via ``peer_http`` and persists the active posture
state + ``awrouter`` stand-in environment file so switching postures is one
command, one button, or one prompt.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .models import ModelState, PeerInfo
from .peer_http import build_park_request, build_warm_request, send_request

DEFAULT_POSTURES_PATH = Path.home() / ".config" / "aither" / "postures.json"
DEFAULT_STATE_PATH = Path.home() / ".config" / "aither" / "active-posture.json"
DEFAULT_ROUTER_ENV_PATH = Path.home() / ".config" / "aither" / "router-posture.env"


@dataclass(frozen=True)
class PeerPosture:
    """Desired warm/park state for one peer (or ``"*"`` for all reachable peers)."""

    warm: tuple[str, ...] = ()
    park: tuple[str, ...] = ()
    park_others: bool = False


@dataclass(frozen=True)
class Posture:
    """A named fleet model posture."""

    name: str
    description: str
    primary_lane: str = "bonsai2.5-27b"
    target_ram_gb: int = 0
    local_model: Optional[str] = None
    peers: dict[str, PeerPosture] = field(default_factory=dict)
    router_standins: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "primary_lane": self.primary_lane,
            "target_ram_gb": self.target_ram_gb,
            "local_model": self.local_model,
            "peers": {
                peer: {
                    "warm": list(pp.warm),
                    "park": list(pp.park),
                    "park_others": pp.park_others,
                }
                for peer, pp in self.peers.items()
            },
            "router_standins": dict(self.router_standins),
        }


@dataclass(frozen=True)
class PostureAction:
    """One concrete warm/park step emitted by ``plan_posture``."""

    peer: str
    action: str  # "park" | "warm"
    model: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "peer": self.peer,
            "action": self.action,
            "model": self.model,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PosturePlan:
    """Result of planning a posture against live peer state."""

    posture: Posture
    actions: tuple[PostureAction, ...]
    skipped: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    router_env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "posture": self.posture.name,
            "description": self.posture.description,
            "primary_lane": self.posture.primary_lane,
            "local_model": self.posture.local_model,
            "actions": [a.to_dict() for a in self.actions],
            "skipped": list(self.skipped),
            "notes": list(self.notes),
            "router_env": dict(self.router_env),
        }


BUILTIN_POSTURES: dict[str, Posture] = {
    "fleet-multi": Posture(
        name="fleet-multi",
        description=(
            "Daily multi-specialist fleet: fast GPU runs hot Bonsai 2.5 27B; "
            "high-memory unified nodes hold Qwen 3.8 27B, Bonsai 2.5 overflow, "
            "Inkling 2 Mini, and embeddings simultaneously."
        ),
        primary_lane="bonsai2.5-27b",
        target_ram_gb=64,
        local_model="bonsai2.5-27b",
        peers={
            "dgx-spark": PeerPosture(
                warm=("bonsai2.5-27b", "qwen3.8-27b", "inkling-2-mini", "aither-code-embed"),
                park=("deepseek-v4-flash",),
                park_others=False,
            ),
            "local": PeerPosture(
                warm=("bonsai2.5-27b", "bonsai2-4b"),
                park=("deepseek-v4-flash",),
                park_others=False,
            ),
        },
        router_standins={
            "orchestrator": "bonsai2.5-27b",
            "coder": "qwen3.8-27b",
            "fast": "bonsai2-4b",
            "offload": "inkling-2-mini",
        },
    ),
    "heavy-moe": Posture(
        name="heavy-moe",
        description=(
            "Single giant MoE reasoning posture: parks specialist models on the "
            "high-memory node and warms DeepSeek-V4-Flash across clean unified memory."
        ),
        primary_lane="deepseek-v4-flash",
        target_ram_gb=110,
        local_model="deepseek-v4-flash",
        peers={
            "dgx-spark": PeerPosture(
                warm=("deepseek-v4-flash",),
                park=("bonsai2.5-27b", "qwen3.8-27b", "inkling-2-mini"),
                park_others=True,
            ),
            "local": PeerPosture(
                warm=("bonsai2.5-27b",),
                park=(),
                park_others=False,
            ),
        },
        router_standins={
            "orchestrator": "bonsai2.5-27b",
            "reasoning": "deepseek-v4-flash",
            "coder": "deepseek-v4-flash",
        },
    ),
    "creator": Posture(
        name="creator",
        description=(
            "Media & creation headroom: keeps lightweight Bonsai 2.5 27B warm while "
            "freeing GPU memory for image, video, and voice workloads."
        ),
        primary_lane="bonsai2.5-27b",
        target_ram_gb=24,
        local_model="bonsai2.5-27b",
        peers={
            "dgx-spark": PeerPosture(
                warm=("bonsai2.5-27b", "aither-code-embed"),
                park=("deepseek-v4-flash", "qwen3.8-27b", "inkling-2-mini"),
                park_others=True,
            ),
            "local": PeerPosture(
                warm=("bonsai2-4b",),
                park=("deepseek-v4-flash",),
                park_others=False,
            ),
        },
        router_standins={
            "orchestrator": "bonsai2.5-27b",
            "fast": "bonsai2-4b",
        },
    ),
    "quiet": Posture(
        name="quiet",
        description=(
            "Low-power workstation posture: parks background fleet models to leave "
            "maximum clean RAM (~119 GiB on Spark) and ~4W GPU idle."
        ),
        primary_lane="none",
        target_ram_gb=2,
        local_model=None,
        peers={
            "*": PeerPosture(
                warm=(),
                park=(),
                park_others=True,
            ),
        },
        router_standins={},
    ),
}


def parse_posture(name: str, raw: dict[str, Any]) -> Posture:
    """Parse one posture dictionary into a validated Posture."""
    if not isinstance(raw, dict):
        raise ValueError(f"posture {name!r} must be a JSON object")
    raw_peers = raw.get("peers") or {}
    if not isinstance(raw_peers, dict):
        raise ValueError(f"posture {name!r}: 'peers' must be an object")
    peers: dict[str, PeerPosture] = {}
    for pname, praw in raw_peers.items():
        if not isinstance(praw, dict):
            raise ValueError(f"posture {name!r}: peer {pname!r} must be an object")
        warm = tuple(str(m) for m in (praw.get("warm") or []))
        park = tuple(str(m) for m in (praw.get("park") or []))
        park_others = bool(praw.get("park_others", False))
        peers[str(pname)] = PeerPosture(warm=warm, park=park, park_others=park_others)
    standins_raw = raw.get("router_standins") or {}
    if not isinstance(standins_raw, dict):
        raise ValueError(f"posture {name!r}: 'router_standins' must be an object")
    standins = {str(k): str(v) for k, v in standins_raw.items()}
    local_model = raw.get("local_model")
    return Posture(
        name=str(name),
        description=str(raw.get("description") or ""),
        primary_lane=str(raw.get("primary_lane") or "bonsai2.5-27b"),
        target_ram_gb=int(raw.get("target_ram_gb") or 0),
        local_model=str(local_model) if local_model else None,
        peers=peers,
        router_standins=standins,
    )


def load_postures(path: Optional[Path | str] = None) -> dict[str, Posture]:
    """Return built-in postures merged with any user postures JSON file."""
    postures = dict(BUILTIN_POSTURES)
    env_path = os.environ.get("AWPOOL_POSTURES_FILE")
    cfg_path = Path(path) if path is not None else (Path(env_path) if env_path else DEFAULT_POSTURES_PATH)
    if not cfg_path.exists():
        return postures
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("postures"), dict):
        entries = data["postures"]
    elif isinstance(data, dict):
        entries = data
    else:
        raise ValueError(f"{cfg_path}: expected a JSON object of posture definitions")
    for name, raw in entries.items():
        postures[str(name)] = parse_posture(str(name), raw)
    return postures


def format_router_env(posture: Posture) -> dict[str, str]:
    """Build environment variables honoured by ``awrouter`` for this posture."""
    env: dict[str, str] = {"AWPOOL_POSTURE": posture.name}
    if posture.router_standins:
        pairs = [f"{k}={v}" for k, v in sorted(posture.router_standins.items())]
        env["AITHER_MODEL_STANDINS"] = ",".join(pairs)
    else:
        env["AITHER_MODEL_STANDINS"] = ""
    return env


def _match_peer_rule(posture: Posture, peer_name: str) -> Optional[PeerPosture]:
    if peer_name in posture.peers:
        return posture.peers[peer_name]
    lower = peer_name.lower()
    for key, rule in posture.peers.items():
        if key != "*" and key.lower() == lower:
            return rule
    return posture.peers.get("*")


def plan_posture(posture: Posture, peer_infos: list[PeerInfo]) -> PosturePlan:
    """Compute ordered park-then-warm actions for ``posture`` over ``peer_infos``.

    Pure function — no network or filesystem I/O.
    """
    park_actions: list[PostureAction] = []
    warm_actions: list[PostureAction] = []
    skipped: list[str] = []
    notes: list[str] = []

    by_name = {p.name: p for p in peer_infos}
    matched_peers: set[str] = set()

    for info in peer_infos:
        rule = _match_peer_rule(posture, info.name)
        if rule is None:
            continue
        matched_peers.add(info.name)
        if info.reachable is not True:
            state = "unreachable" if info.reachable is False else "reachability unknown"
            notes.append(f"{info.name}: skipped ({state})")
            continue

        desired_warm = set(rule.warm)
        explicit_park = list(rule.park)
        seen_park: set[str] = set()

        # 1. Explicit parks
        for mid in explicit_park:
            if mid in desired_warm or mid in seen_park:
                continue
            cur = info.models.get(mid)
            if cur is ModelState.COLD:
                skipped.append(f"{info.name}: {mid} already cold")
            else:
                reason = "currently warm" if cur is ModelState.WARM else "requested park (state unknown)"
                park_actions.append(PostureAction(peer=info.name, action="park", model=mid, reason=reason))
            seen_park.add(mid)

        # 2. Park any other WARM models when park_others=True
        if rule.park_others:
            if not info.models:
                notes.append(f"{info.name}: park_others requested, but peer reported no model catalog")
            else:
                for mid, cur in sorted(info.models.items()):
                    if cur is ModelState.WARM and mid not in desired_warm and mid not in seen_park:
                        park_actions.append(
                            PostureAction(
                                peer=info.name,
                                action="park",
                                model=mid,
                                reason="park_others (freeing memory)",
                            )
                        )
                        seen_park.add(mid)

        # 3. Desired warms
        for mid in rule.warm:
            cur = info.models.get(mid)
            if cur is ModelState.WARM:
                skipped.append(f"{info.name}: {mid} already warm")
            else:
                reason = "currently cold" if cur is ModelState.COLD else "requested warm (state unknown)"
                warm_actions.append(PostureAction(peer=info.name, action="warm", model=mid, reason=reason))

    # Note any named peers in the posture that were absent from the registry
    for pname in posture.peers:
        if pname != "*" and pname not in by_name and not any(p.lower() == pname.lower() for p in by_name):
            notes.append(f"{pname}: not present in peer registry")

    return PosturePlan(
        posture=posture,
        actions=tuple(park_actions + warm_actions),
        skipped=tuple(skipped),
        notes=tuple(notes),
        router_env=format_router_env(posture),
    )


def write_router_env(router_env: dict[str, str], path: Path = DEFAULT_ROUTER_ENV_PATH) -> Path:
    """Write shell-sourceable router posture env file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by awpool posture apply — sourced by awrouter / shell sessions",
    ]
    for key, val in sorted(router_env.items()):
        safe = val.replace("'", "'\"'\"'")
        lines.append(f"export {key}='{safe}'")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def save_active_posture(
    posture: Posture,
    results: list[dict[str, Any]],
    *,
    state_path: Path = DEFAULT_STATE_PATH,
    router_env_path: Optional[Path] = DEFAULT_ROUTER_ENV_PATH,
) -> dict[str, Any]:
    """Persist active posture state to JSON (and optionally router env)."""
    router_env = format_router_env(posture)
    env_written: Optional[str] = None
    if router_env_path is not None:
        env_written = str(write_router_env(router_env, router_env_path))
    record = {
        "posture": posture.name,
        "description": posture.description,
        "primary_lane": posture.primary_lane,
        "local_model": posture.local_model,
        "applied_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "router_env": router_env,
        "router_env_path": env_written,
        "results": results,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(state_path)
    return record


def read_active_posture(state_path: Path = DEFAULT_STATE_PATH) -> Optional[dict[str, Any]]:
    """Read the last applied posture record, or None if never set."""
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def apply_posture(
    plan: PosturePlan,
    endpoints: dict[str, str],
    *,
    dry_run: bool = False,
    timeout: float = 10.0,
    sender: Callable[[Any], dict[str, Any]] = send_request,
    state_path: Optional[Path] = DEFAULT_STATE_PATH,
    router_env_path: Optional[Path] = DEFAULT_ROUTER_ENV_PATH,
) -> dict[str, Any]:
    """Execute a ``PosturePlan`` (park first, then warm) against peer HTTP endpoints."""
    results: list[dict[str, Any]] = []
    all_ok = True

    for step in plan.actions:
        ep = endpoints.get(step.peer)
        if not ep:
            all_ok = False
            results.append(
                {
                    "peer": step.peer,
                    "action": step.action,
                    "model": step.model,
                    "ok": False,
                    "status": 0,
                    "error": "no endpoint for peer",
                }
            )
            continue
        if dry_run:
            results.append(
                {
                    "peer": step.peer,
                    "action": step.action,
                    "model": step.model,
                    "ok": True,
                    "dry_run": True,
                    "endpoint": ep,
                }
            )
            continue
        builder = build_warm_request if step.action == "warm" else build_park_request
        req = builder(ep, step.model, timeout=timeout)
        res = sender(req)
        ok = bool(res.get("ok"))
        if not ok:
            all_ok = False
        results.append(
            {
                "peer": step.peer,
                "action": step.action,
                "model": step.model,
                "ok": ok,
                "status": res.get("status", 0),
                "error": res.get("error"),
            }
        )

    state_record: Optional[dict[str, Any]] = None
    if not dry_run and state_path is not None:
        state_record = save_active_posture(
            plan.posture,
            results,
            state_path=state_path,
            router_env_path=router_env_path,
        )

    return {
        "posture": plan.posture.name,
        "ok": all_ok,
        "dry_run": dry_run,
        "actions": results,
        "skipped": list(plan.skipped),
        "notes": list(plan.notes),
        "router_env": dict(plan.router_env),
        "state": state_record,
    }
