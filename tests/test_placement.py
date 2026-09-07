"""Placement scoring tests — pure data in, decision out, no network.

Covers the tier ladder (warm > cold > unknown), the capacity tie-break, the
determinism guarantee, and the honest-fallback path: when no peer is
confirmed reachable the decision is still made, but it is LABELLED
"unverified" rather than reading as a confident pick.
"""

from awpool.models import ModelState, PeerInfo
from awpool.placement import pick_peer


def test_pick_prefers_warm_over_cold():
    cold = PeerInfo(name="desktop", reachable=True, models={"qwen3-8b": ModelState.COLD})
    warm = PeerInfo(name="laptop", reachable=True, models={"qwen3-8b": ModelState.WARM})
    p = pick_peer([cold, warm], "qwen3-8b")
    assert p is not None
    assert p.peer == "laptop"
    assert p.basis == "warm"
    assert "warm" in p.reason


def test_pick_breaks_ties_by_capacity():
    small = PeerInfo(name="phone", reachable=True, models={"m": ModelState.WARM}, capacity=1)
    big = PeerInfo(name="desktop", reachable=True, models={"m": ModelState.WARM}, capacity=4)
    p = pick_peer([small, big], "m")
    assert p.peer == "desktop"
    assert p.basis == "warm"


def test_pick_is_deterministic_on_equal_evidence():
    a = PeerInfo(name="zulu", reachable=True, models={"m": ModelState.WARM}, capacity=2)
    b = PeerInfo(name="alpha", reachable=True, models={"m": ModelState.WARM}, capacity=2)
    assert pick_peer([a, b], "m").peer == "alpha"
    assert pick_peer([b, a], "m").peer == "alpha"  # order must not matter


def test_pick_cold_reported_beats_unknown_no_catalog():
    with_catalog = PeerInfo(name="desktop", reachable=True, models={"other": ModelState.WARM})
    no_catalog = PeerInfo(name="phone", reachable=True, models={})
    p = pick_peer([no_catalog, with_catalog], "m")
    # desktop reported a catalog without "m" -> COLD; phone reported nothing
    # -> UNKNOWN. Cold beats unknown: a peer that can load is better than
    # one we cannot see.
    assert p.peer == "desktop"
    assert p.basis == "cold"


def test_pick_never_picks_a_confirmed_unreachable_peer():
    down = PeerInfo(name="desktop", reachable=False, models={"m": ModelState.WARM})
    up = PeerInfo(name="laptop", reachable=True, models={"m": ModelState.COLD})
    p = pick_peer([down, up], "m")
    assert p is not None
    assert p.peer == "laptop"  # warm-but-unreachable is worse than cold-but-up


def test_pick_honest_fallback_when_no_peer_confirmed_reachable():
    unsure_warm = PeerInfo(name="phone", reachable=None, models={"m": ModelState.WARM})
    unsure_silent = PeerInfo(name="desktop", reachable=None)
    p = pick_peer([unsure_warm, unsure_silent], "m")
    assert p is not None
    assert p.peer == "phone"  # warm evidence wins among the unconfirmed
    assert p.basis == "unverified"  # but the pick is a labelled guess
    assert "guess" in p.reason


def test_pick_returns_none_when_every_peer_confirmed_unreachable():
    a = PeerInfo(name="desktop", reachable=False, models={"m": ModelState.WARM})
    b = PeerInfo(name="phone", reachable=False)
    assert pick_peer([a, b], "m") is None


def test_pick_returns_none_on_empty_pool():
    assert pick_peer([], "m") is None


def test_pick_basis_labels_the_evidence_tier():
    warm = pick_peer([PeerInfo(name="d", reachable=True, models={"m": ModelState.WARM})], "m")
    assert warm.basis == "warm"
    # A catalog reported WITHOUT the model -> COLD; no catalog at all ->
    # UNKNOWN. The two are different evidence tiers and must not collapse.
    cold = pick_peer([PeerInfo(name="d", reachable=True, models={"other": ModelState.WARM})], "m")
    assert cold.basis == "cold"
    silent = pick_peer([PeerInfo(name="d", reachable=True)], "m")
    assert silent.basis == "unknown"
