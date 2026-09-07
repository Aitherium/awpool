"""Placement — which peer should serve a model right now. Pure logic, no I/O.

The decision takes peers as DATA (list[PeerInfo]); the CLI gathers that data
from awnet + each peer's HTTP API, and this module decides. Keeping the two
halves apart is what makes the scoring unit-testable without a network and
what makes the fallback path honest: a decision built on no evidence must
SAY it is built on no evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import ModelState, PeerInfo


@dataclass(frozen=True)
class Placement:
    """A placement decision.

    `peer` — the chosen peer's name.
    `model_state` — what we believe the model's state is there.
    `basis` — the evidence tier the decision rests on:
        "warm"       a confirmed-reachable peer holds the model warm
        "cold"       a confirmed-reachable peer can load it
        "unknown"    a confirmed-reachable peer answered but reported no
                     state for this model
        "unverified" NO peer is confirmed reachable; the pick is the
                     least-bad candidate that could not be ruled out, and
                     must be treated as a guess, not a decision
    `reason` — one human-readable line.
    """

    peer: str
    model_state: ModelState
    basis: str
    reason: str


def model_state_on(peer: PeerInfo, model: str) -> ModelState:
    """What we believe the model's state on this peer is.

    A peer that reported a catalog without the model is COLD (it can load it
    on request). A peer that reported NO catalog is UNKNOWN — we cannot see
    its model state at all, and "cannot see" must not read as "cold".
    """
    if model in peer.models:
        return peer.models[model]
    if peer.models:
        return ModelState.COLD
    return ModelState.UNKNOWN


def _rank(peer: PeerInfo, model: str) -> tuple[int, int, str]:
    """Lower is better: (warmness, -capacity, name).

    Warm-status is the primary axis, capacity the tie-break, and the name
    makes the ordering deterministic so the same pool always picks the same
    peer. Unknown capacity (0) ranks last, never first.
    """
    state = model_state_on(peer, model)
    state_rank = {
        ModelState.WARM: 0,
        ModelState.COLD: 1,
        ModelState.UNKNOWN: 2,
    }[state]
    return (state_rank, -peer.capacity, peer.name)


def pick_peer(peers: list[PeerInfo], model: str) -> Optional[Placement]:
    """Decide which peer should serve `model` right now.

    Returns None ONLY when there is genuinely no candidate: an empty pool,
    or every peer confirmed unreachable. Every other outcome is a Placement
    whose `basis` states how much evidence the decision rests on — including
    the honest-fallback case where no peer is confirmed reachable and the
    pick is a labelled guess.
    """
    if not peers:
        return None

    reachable = [p for p in peers if p.reachable is True]
    if reachable:
        best = min(reachable, key=lambda p: _rank(p, model))
        state = model_state_on(best, model)
        if state is ModelState.WARM:
            reason = f"{best.name} has {model} warm"
        elif state is ModelState.COLD:
            reason = f"{best.name} can load {model} (cold — request warm)"
        else:
            reason = f"{best.name} is reachable but reported no state for {model}"
        return Placement(peer=best.name, model_state=state, basis=state.value, reason=reason)

    unsure = [p for p in peers if p.reachable is None]
    if unsure:
        best = min(unsure, key=lambda p: _rank(p, model))
        reason = (
            f"no peer is confirmed reachable; {best.name} could not be ruled "
            f"out and is the least-bad candidate — treat this as a guess, "
            f"not a decision"
        )
        return Placement(
            peer=best.name,
            model_state=model_state_on(best, model),
            basis="unverified",
            reason=reason,
        )

    # Every peer is confirmed unreachable: there is no candidate at all.
    return None
