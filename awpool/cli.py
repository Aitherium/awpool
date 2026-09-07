"""awpool CLI.

Exit codes are the contract (same as awnet):
  0: success (a decision was made, a request was accepted)
  1: a real problem (nothing reachable to pick from, a peer refused)
  2: could not judge (awnet absent, registry unreadable, a check failed)

The client never manages the fleet: warm/park are best-effort REQUESTS to
the peer's own HTTP API, and a refusal is a reported outcome, not an
exception to retry forever. awnet is imported LAZILY inside the commands —
an absent awnet is an honest "install awnet to discover peers", never a
silent feature-off.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .models import PeerInfo, is_model_catalog, parse_peer_capacity, parse_peer_models
from .peer_http import build_park_request, build_warm_request, fetch_peer_models, send_request
from .placement import pick_peer

DEFAULT_REGISTRY = str(Path.home() / ".config" / "aither" / "peers.json")

_NO_AWNET_MSG = (
    "awnet is not installed — awpool discovers peers through awnet's "
    "registry, and without it there is nothing to discover. Install it "
    "(pip install awnet) and try again; the placement logic itself is here "
    "and working."
)


def _awnet() -> Optional[tuple]:
    """Lazy awnet import. Returns (Registry, check_peer) or None."""
    try:
        from awnet.check import check_peer
        from awnet.registry import Registry

        return (Registry, check_peer)
    except ImportError:
        return None


def _load_peers(registry_path: str) -> Optional[tuple]:
    """Load awnet's Registry. Returns (registry, check_peer) or None (DEAD)."""
    mods = _awnet()
    if mods is None:
        print(_NO_AWNET_MSG, file=sys.stderr)
        return None
    registry_cls, check_peer = mods
    try:
        registry = registry_cls(registry_path)
    except Exception as e:  # noqa: BLE001 - awnet surfaces whatever json gave it
        print(f"DEAD: could not load registry {registry_path}: {e}", file=sys.stderr)
        return None
    return (registry, check_peer)


def _reachability(check_peer, endpoints: list[str], timeout: float) -> Optional[bool]:
    """Three-valued reachability across a peer's endpoints.

    Any endpoint answering -> True. None answering but at least one
    unjudgeable -> None (could not determine). All confirmed down -> False.
    """
    verdict: Optional[bool] = False
    for ep in endpoints:
        result = check_peer(ep, timeout=timeout)
        if result.ok is True:
            return True
        if result.ok is None:
            verdict = None
    return verdict


def _gather(registry, check_peer, timeout: float) -> tuple[list[PeerInfo], list[str]]:
    """Reachability + model catalogs for every peer — the data pick_peer needs.

    Returns (peer_infos, notes). A peer that is reachable but answers
    non-JSON (or not at all) becomes a PeerInfo with reachable=True and NO
    models: its model state is UNKNOWN, and the note says why.
    """
    infos: list[PeerInfo] = []
    notes: list[str] = []
    for peer in registry.list_peers():
        reachable = _reachability(check_peer, peer.endpoints, timeout)
        if reachable is True and peer.endpoints:
            payload, err = fetch_peer_models(peer.endpoints[0], timeout=timeout)
            if err or not is_model_catalog(payload):
                notes.append(f"{peer.name}: model state unknown ({err or 'no catalog'})")
                infos.append(PeerInfo(name=peer.name, reachable=True))
            else:
                infos.append(
                    PeerInfo(
                        name=peer.name,
                        reachable=True,
                        models=parse_peer_models(payload),
                        capacity=parse_peer_capacity(payload),
                    )
                )
        else:
            infos.append(PeerInfo(name=peer.name, reachable=reachable))
    return infos, notes


def _cmd_peers(args) -> int:
    loaded = _load_peers(args.registry)
    if loaded is None:
        return 2
    registry, check_peer = loaded
    peers = registry.list_peers()
    if not peers:
        print(f"no peers registered (looked in {args.registry})")
        return 0
    rows = []
    any_reachable = False
    for peer in peers:
        for ep in peer.endpoints:
            result = check_peer(ep, timeout=args.timeout)
            verdict = {True: "ok", False: "unreachable", None: "unknown"}[result.ok]
            any_reachable = any_reachable or result.ok is True
            rows.append(
                {
                    "peer": peer.name,
                    "endpoint": ep,
                    "verdict": verdict,
                    "method": result.method,
                    "error": result.error,
                }
            )
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            err = f" — {r['error']}" if r["error"] else ""
            print(f"{r['peer']:24} {r['endpoint']:32} {r['verdict']}{err}")
    return 0 if any_reachable else 1


def _cmd_status(args) -> int:
    loaded = _load_peers(args.registry)
    if loaded is None:
        return 2
    registry, check_peer = loaded
    infos, notes = _gather(registry, check_peer, args.timeout)
    if not infos:
        print(f"no peers registered (looked in {args.registry})")
        return 0
    rows = []
    for info in infos:
        if info.reachable is not True:
            state = "unreachable" if info.reachable is False else "unknown"
            rows.append({"peer": info.name, "state": state, "models": {}})
        elif not info.models:
            rows.append({"peer": info.name, "state": "unknown", "models": {}})
        else:
            models = {mid: st.value for mid, st in sorted(info.models.items())}
            rows.append({"peer": info.name, "state": "reachable", "models": models})
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            if r["models"]:
                detail = "  ".join(f"{m}={s}" for m, s in r["models"].items())
                print(f"{r['peer']:24} {detail}")
            elif r["state"] == "reachable":
                print(f"{r['peer']:24} reachable, no model state reported")
            else:
                print(f"{r['peer']:24} {r['state']}")
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    return 0


def _placement_json(model: str, placement) -> str:
    if placement is None:
        return json.dumps({"model": model, "placement": None}, indent=2)
    return json.dumps(
        {
            "model": model,
            "placement": {
                "peer": placement.peer,
                "basis": placement.basis,
                "model_state": placement.model_state.value,
                "reason": placement.reason,
            },
        },
        indent=2,
    )


def _cmd_pick(args) -> int:
    loaded = _load_peers(args.registry)
    if loaded is None:
        return 2
    registry, check_peer = loaded
    infos, notes = _gather(registry, check_peer, args.timeout)
    if not infos:
        print(f"no peers registered (looked in {args.registry})")
        return 2
    placement = pick_peer(infos, args.model)
    if args.json:
        print(_placement_json(args.model, placement))
    elif placement is None:
        print(
            f"no peer can serve {args.model}: every peer is confirmed "
            f"unreachable right now"
        )
    else:
        print(f"{args.model} -> {placement.peer}  ({placement.basis} — {placement.reason})")
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    return 0 if placement is not None else 1


def _peer_endpoint(registry, name: str) -> Optional[str]:
    peer = registry.get_peer(name)
    if peer is None or not peer.endpoints:
        return None
    return peer.endpoints[0]


def _cmd_request(args, action: str) -> int:
    loaded = _load_peers(args.registry)
    if loaded is None:
        return 2
    registry, check_peer = loaded
    infos, _notes = _gather(registry, check_peer, args.timeout)

    if args.peer:
        endpoint = _peer_endpoint(registry, args.peer)
        if endpoint is None:
            print(f"ERROR: no such peer (or no endpoint): {args.peer}", file=sys.stderr)
            return 1
        peer_name = args.peer
    else:
        placement = pick_peer(infos, args.model)
        if placement is None:
            print(
                f"no peer can {action} {args.model}: every peer is confirmed "
                f"unreachable right now",
                file=sys.stderr,
            )
            return 1
        peer_name = placement.peer
        endpoint = _peer_endpoint(registry, peer_name)
        if endpoint is None:
            print(f"ERROR: picked peer {peer_name} has no endpoint", file=sys.stderr)
            return 2

    builder = build_warm_request if action == "warm" else build_park_request
    request = builder(endpoint, args.model, timeout=args.timeout)
    result = send_request(request)
    if result["ok"]:
        print(f"{action} {args.model} -> {peer_name}: accepted (HTTP {result['status']})")
        return 0
    status = f"HTTP {result['status']}" if result["status"] else "no response"
    print(f"{action} {args.model} -> {peer_name}: refused ({status}) — {result['error']}")
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    # GENERATED doctor intercept (gen_aw_doctor.py) -- do not edit
    _dv = locals().get("argv")
    if (_dv if _dv is not None else __import__("sys").argv[1:])[:1] == ["doctor"]:
        from ._doctor import report
        return report()
    # GENERATED repo-state intercept (gen_aw_doctor.py) -- do not edit
    try:
        from awgit import state as _aw_state
    except Exception:
        _aw_state = None
    if _aw_state is not None:
        _sv = locals().get("argv")
        if _aw_state.cli_banner(_sv if _sv is not None else __import__("sys").argv[1:]):
            return 0
    ap = argparse.ArgumentParser(
        prog="awpool",
        description=(
            "Model placement and warm-keeping across your own devices — "
            "which reachable peer should serve a model right now, and asking "
            "it to warm or park. A client over awnet's peer registry."
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "registry",
            nargs="?",
            default=DEFAULT_REGISTRY,
            help="awnet registry file (default: ~/.config/aither/peers.json)",
        )
        p.add_argument(
            "--timeout", type=float, default=5.0,
            help="seconds to wait for a peer (default 5)",
        )

    peers_cmd = sub.add_parser("peers", help="list peers awnet reports reachable")
    add_common(peers_cmd)
    peers_cmd.add_argument("--json", action="store_true")

    status_cmd = sub.add_parser("status", help="per-peer model state (warm/cold/unknown)")
    add_common(status_cmd)
    status_cmd.add_argument("--json", action="store_true")

    pick_cmd = sub.add_parser("pick", help="which reachable peer should serve a model right now")
    pick_cmd.add_argument("model")
    add_common(pick_cmd)
    pick_cmd.add_argument("--json", action="store_true")

    warm_cmd = sub.add_parser("warm", help="ask the chosen peer to warm a model")
    warm_cmd.add_argument("model")
    add_common(warm_cmd)
    warm_cmd.add_argument(
        "--peer", default=None,
        help="peer name; defaults to the placement decision",
    )

    park_cmd = sub.add_parser("park", help="ask the chosen peer to park a model")
    park_cmd.add_argument("model")
    add_common(park_cmd)
    park_cmd.add_argument(
        "--peer", default=None,
        help="peer name; defaults to the placement decision",
    )

    args = ap.parse_args(argv)

    if args.cmd == "peers":
        return _cmd_peers(args)
    if args.cmd == "status":
        return _cmd_status(args)
    if args.cmd == "pick":
        return _cmd_pick(args)
    if args.cmd == "warm":
        return _cmd_request(args, "warm")
    if args.cmd == "park":
        return _cmd_request(args, "park")
    return 2


if __name__ == "__main__":
    sys.exit(main())
