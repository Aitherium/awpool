"""awpool — model placement and warm-keeping for a personal mesh pool.

A client, not a fleet: it discovers peers through awnet's registry, asks
reachable peers about their model state, and picks where a model should be
warm — then REQUESTs warm/park from the peer's own HTTP API. It never
manages the pool; the peers do that.

Nothing here imports awnet at module load time. awnet is discovered LAZILY
in the CLI, so `import awpool` works on a machine that has never heard of
awnet, and an absent awnet is reported honestly rather than reading as a
feature that is switched off.
"""

from .models import (
    ModelState,
    PeerInfo,
    is_model_catalog,
    parse_peer_capacity,
    parse_peer_models,
)
from .placement import Placement, model_state_on, pick_peer

__all__ = [
    "ModelState",
    "PeerInfo",
    "Placement",
    "is_model_catalog",
    "model_state_on",
    "parse_peer_capacity",
    "parse_peer_models",
    "pick_peer",
]
