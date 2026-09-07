"""Pure parsing tests: model state vocabulary and peer payload parsing.

No network, no awnet — these exercise the data plane that placement
decisions are built on, including the honest-unknown paths (a peer that
reported nothing is not a peer that reported empty).
"""

from awpool.models import (
    ModelState,
    PeerInfo,
    is_model_catalog,
    parse_peer_capacity,
    parse_peer_models,
)
from awpool.placement import model_state_on


def test_parse_peer_models_list_shape():
    payload = {
        "models": [
            {"id": "qwen3-8b", "state": "warm"},
            {"id": "gemma4-12b", "state": "cold"},
        ]
    }
    out = parse_peer_models(payload)
    assert out == {"qwen3-8b": ModelState.WARM, "gemma4-12b": ModelState.COLD}


def test_parse_peer_models_dict_shape():
    payload = {"models": {"qwen3-8b": "warm", "gemma4-12b": "cold"}}
    assert parse_peer_models(payload) == {
        "qwen3-8b": ModelState.WARM,
        "gemma4-12b": ModelState.COLD,
    }


def test_parse_peer_models_bare_list_shape():
    payload = [{"id": "qwen3-8b", "state": "warm"}]
    assert parse_peer_models(payload) == {"qwen3-8b": ModelState.WARM}


def test_parse_peer_models_unknown_state_and_missing_id():
    payload = {"models": [{"id": "a", "state": "defrosting"}, {"state": "warm"}, {"id": "b"}]}
    out = parse_peer_models(payload)
    assert out["a"] is ModelState.UNKNOWN  # unparseable state -> UNKNOWN, not cold
    assert out["b"] is ModelState.UNKNOWN  # no state -> UNKNOWN
    assert len(out) == 2  # the id-less row is skipped


def test_parse_peer_models_garbage_is_empty():
    assert parse_peer_models({"status": "ok"}) == {}
    assert parse_peer_models(None) == {}
    assert parse_peer_models("warm") == {}


def test_is_model_catalog_distinguishes_reported_empty_from_unparseable():
    # An empty catalog is still a catalog (the peer genuinely has none)...
    assert is_model_catalog({"models": []}) is True
    assert is_model_catalog({"models": {}}) is True
    assert is_model_catalog([]) is True
    # ...but a health envelope or garbage is NO catalog: the peer's model
    # state is unknown, never "empty".
    assert is_model_catalog({"status": "ok"}) is False
    assert is_model_catalog(None) is False
    assert is_model_catalog("<html>oops</html>") is False


def test_parse_peer_capacity_unknown_is_zero():
    assert parse_peer_capacity({"models": []}) == 0
    assert parse_peer_capacity(None) == 0
    assert parse_peer_capacity({"capacity": -3}) == 0  # negative -> unknown
    assert parse_peer_capacity({"capacity": True}) == 0  # bool is not a count


def test_parse_peer_capacity_reads_nonnegative_int():
    assert parse_peer_capacity({"capacity": 2, "models": []}) == 2


def test_model_state_on_three_cases():
    peer = PeerInfo(name="desktop", reachable=True, models={"a": ModelState.WARM})
    assert model_state_on(peer, "a") is ModelState.WARM
    # Reported a catalog without the model -> COLD (it can load it)...
    assert model_state_on(peer, "b") is ModelState.COLD
    # Reported NO catalog -> UNKNOWN, and "cannot see" is not "cold".
    silent = PeerInfo(name="phone", reachable=True, models={})
    assert model_state_on(silent, "a") is ModelState.UNKNOWN


def test_model_state_parse_is_case_insensitive():
    assert ModelState.parse("WARM") is ModelState.WARM
    assert ModelState.parse("Cold") is ModelState.COLD
    assert ModelState.parse(42) is ModelState.UNKNOWN
