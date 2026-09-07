# awpool — a model stays warm across your own devices

<!-- aither-header:start GENERATED from the ecosystem registry. Edits here are overwritten; change the registry instead. -->

**[Docs](https://aitherium.github.io/awpool/)**  ·  [Source](https://github.com/Aitherium/awpool)  ·  `pip install awpool`  ·  [The Aither World](https://aitherium.github.io/)

> **The Aither World** is an operating system for agents — a Linux you can hand to one, the runtimes it works in, and the tools it works with. [awnix](https://github.com/Aitherium/awnix) is the Linux underneath it; **awpool** is one of its bricks — each installs on its own, runs offline, and needs no account.
>
> **Start here:** Have a model stay warm and servable across your own devices — pick which of your reachable peers should serve it.

<!-- aither-header:end -->

Laptop, desktop, phone: your own devices, one pool. Each runs a local
backend; a model loaded on one of them should serve whichever device is on
right now — and a model nobody is using should be parked, not sitting on a
GPU it does not need. awpool is the small client that decides and asks:
which reachable peer should serve this model, and will you warm or park it?

```bash
pip install awpool
awpool peers                       # who is reachable right now
awpool status                      # per-peer model state (warm/cold/unknown)
awpool pick qwen3-8b               # where should it serve from NOW
awpool warm qwen3-8b               # ask that peer to load it
awpool park qwen3-8b               # ask that peer to unload it
```

## What it is

A **client over awnet's peer registry** — peers are discovered the way
awnet discovers them (a registry file of names and endpoints, three-valued
reachability: answers / does not answer / could not judge). awpool layers
the two things awnet deliberately does not do: **placement** (which peer
should serve a model right now) and **warm-keeping** (asking a peer to load
or park a model through its own HTTP API).

The placement logic is pure: peers go in as data, a decision comes out, no
network. The scoring ladder:

1. **Warm beats cold beats unknown** — a reachable peer holding the model
   warm wins; a peer that can load it (reported a catalog without it) beats
   a peer whose state we cannot see.
2. **Capacity breaks ties** — a peer that can hold more models warm wins;
   unknown capacity ranks last, never first.
3. **Names break ties** — the same pool always picks the same peer.
4. **The honest fallback** — if NO peer is confirmed reachable, the pick is
   still made (the least-bad candidate that could not be ruled out), but it
   is labelled `unverified` and printed as a guess, not a decision. A pick
   built on no evidence must say so.

A model's state on a peer is **three-valued**: `warm` (loaded), `cold` (not
loaded), `unknown` (the peer did not say). A peer that answers but reports
no catalog is not a peer without models — its state is unknown, and
"cannot see" is never reported as "cold".

## What it is NOT

- **NOT the platform's shared compute market.** This pools YOUR devices —
  laptop, desktop, phone — with no settlement, no marketplace, no tenant
  plane. `aithernet-provider` is the different shape that joins the
  platform's community pool.
- **NOT the ggml-RPC tensor-split fleet stack.** That stack splits ONE model
  too large for any single box across fleet GPUs, hand-tuned per model.
  awpool is placement + warm/park lifecycle for models that fit one device
  each — a different problem, at personal scale, packaged as a client.
- **NOT a fleet manager.** warm/park are best-effort REQUESTS to the peer's
  own HTTP API. A refusal or a sleeping peer is a reported outcome, never a
  retry loop and never a crash.

## Commands

| command | what it does | exit |
|---|---|---|
| `awpool peers [registry]` | reachability per peer endpoint (ok / unreachable / unknown), via awnet | 0 any reachable · 1 none · 2 could not judge |
| `awpool status [registry]` | per-peer model state from each reachable peer's `/v1/models` | 0 |
| `awpool pick <model> [registry]` | the placement decision, with the evidence basis it rests on | 0 decided · 1 nothing reachable · 2 could not judge |
| `awpool warm <model> [--peer NAME]` | ask the chosen (or named) peer to load the model | 0 accepted · 1 refused |
| `awpool park <model> [--peer NAME]` | ask the chosen (or named) peer to unload the model | 0 accepted · 1 refused |

`registry` defaults to `~/.config/aither/peers.json` — the same registry
file awnet writes:

```bash
awnet add ~/.config/aither/peers.json laptop laptop.local:8080
awnet add ~/.config/aither/peers.json desktop desktop.local:8080
awpool pick qwen3-8b
```

The peer HTTP API awpool speaks to is deliberately small: `GET /v1/models`
returns the catalog (`{"models": [{"id": "...", "state": "warm"|"cold"}]}`,
or `{"models": {"id": "warm"}}`, or a bare list, with an optional
`"capacity": N`), and `POST /v1/models/<id>/warm` and
`POST /v1/models/<id>/park` accept the model id. `awnode` is the local
gateway that would serve that API on each device; awpool only ever calls it.

## awnet absent is a real state

awpool imports awnet **lazily**, inside the commands. On a machine without
awnet, `awpool peers` prints that awnet is missing and how to install it,
and exits 2 — an absent dependency is reported honestly, never as a feature
that is switched off. The placement logic itself (`awpool.pick_peer`,
pure, no I/O) works anywhere.

## Status

Registered in the Aither World ecosystem as `planned` (the `personal-mesh-pool`
stack: awnet + awnode + awdk). The registry entry for `id: awpool` predates
this package and describes a different concept (dispatcher onto public
mirror runners); this package implements the placement/warm-keeping client
described by the `personal-mesh-pool` stack entry. Reconcile the registry
entry with that stack when the package ships.

## Development

```bash
python -m pytest tests/ -q
```

Stdlib only — no dependencies to install to run, pytest for the tests.
Placement, parsing and request building are pure functions; the only I/O is
urllib against a peer's own API, so the whole decision plane is testable
without a network.
