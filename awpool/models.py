"""Model state across a personal mesh pool — the warm/cold/unknown vocabulary.

A model on a peer is WARM (loaded, ready to serve), COLD (not loaded), or
UNKNOWN (the peer did not say). Three states, not two: a peer that answers
but does not report a model catalog is not a peer without models — it is a
peer whose model state we cannot see, and pretending otherwise is how "the
pool is empty" starts reading as "the pool is fine". Everything here is pure
data and pure parsing; no I/O, so the placement decisions built on it are
unit-testable without a network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ModelState(Enum):
    """The three states a model can be in on a peer.

    WARM: loaded and ready to serve.
    COLD: not loaded; can be loaded on request.
    UNKNOWN: we cannot see this peer's model state at all.
    """

    WARM = "warm"
    COLD = "cold"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, raw: Any) -> "ModelState":
        """Parse a peer-supplied state string, fail-soft to UNKNOWN.

        A state we do not recognise is not a crash and not COLD — it is
        information we cannot interpret, which is UNKNOWN by definition.
        """
        if isinstance(raw, str):
            try:
                return cls(raw.lower())
            except ValueError:
                return cls.UNKNOWN
        return cls.UNKNOWN


@dataclass(frozen=True)
class PeerInfo:
    """Data-only view of a peer for placement decisions. No I/O here.

    `reachable` is THREE-valued, matching awnet's CheckResult.ok: True the
    peer answers, False it does not, None we could not judge. None is not
    False — a peer we could not check is not a peer that is down.

    `models` maps model id -> state as the peer reported it ({} when the
    peer reported nothing).

    `capacity` is how many models the peer can hold warm; 0 means unknown,
    not "no capacity".
    """

    name: str
    reachable: Optional[bool] = None
    models: dict[str, ModelState] = field(default_factory=dict)
    capacity: int = 0


def is_model_catalog(payload: Any) -> bool:
    """True when the payload is a recognizable model catalog.

    Recognized shapes: a dict with a `models` key whose value is a dict or
    list, or a bare list of rows. Anything else — a health envelope, an
    error page, garbage — is NOT a catalog: the peer's model state is
    UNKNOWN, never "empty". An empty list/dict under `models` IS a catalog
    (the peer genuinely has nothing to report).
    """
    if isinstance(payload, dict):
        rows = payload.get("models")
        return isinstance(rows, (dict, list))
    return isinstance(payload, list)


def parse_peer_models(payload: Any) -> dict[str, ModelState]:
    """Turn a peer's model payload into {model_id: ModelState}.

    Accepts the shapes a small peer HTTP API is likely to serve:
      {"models": [{"id": "qwen3-8b", "state": "warm"}, ...]}
      {"models": {"qwen3-8b": "warm", ...}}
      [{"id": ..., "state": ...}, ...]

    Anything else yields {} — callers distinguish "reported empty" from
    "reported nothing" via `is_model_catalog` and report the latter as
    UNKNOWN. Rows without an id are skipped; unparseable states fall to
    UNKNOWN.
    """
    if isinstance(payload, dict):
        rows = payload.get("models")
    else:
        rows = payload
    out: dict[str, ModelState] = {}
    if isinstance(rows, dict):
        for mid, raw in rows.items():
            out[str(mid)] = ModelState.parse(raw)
    elif isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("id"):
                out[str(row["id"])] = ModelState.parse(row.get("state"))
    return out


def parse_peer_capacity(payload: Any) -> int:
    """A peer may report how many models it can hold warm (`capacity`).

    0 means UNKNOWN — a peer that does not report capacity is not a peer
    with zero capacity, and 0 ranks last in placement tie-breaks. Negative
    values and booleans are treated as unknown.
    """
    if isinstance(payload, dict):
        cap = payload.get("capacity")
        if isinstance(cap, int) and not isinstance(cap, bool) and cap >= 0:
            return cap
    return 0
