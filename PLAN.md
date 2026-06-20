# Inference Demo — Implementation Plan (PLAN.md)

Execution spec for Claude Code. Pairs with CLAUDE.md (the constitution: what + why).
This file is the how + in-what-order. Read CLAUDE.md first, then follow this phase by phase.

## Operating contract (rules of engagement — follow strictly)

1. Work ONE phase at a time, in order. Do not start a later phase early.
2. TDD always: for each unit of logic, write the failing tests FIRST (red), show them
   failing, then implement to green, then refactor. No production logic without a prior test.
3. At each `### CHECKPOINT`, STOP. Summarize what was built, paste the test/verify output, and
   wait for the human to say "continue". Do not blow past checkpoints.
4. Never break the `Worker` and `RoutingStrategy` seams (see CLAUDE.md). Everything above the
   worker stays backend-agnostic.
5. Keep changes reviewable: small commits, conventional-commit messages, commit at each green
   phase. Never commit red tests.
6. Keep `CLAUDE.md` `## Commands` current the moment a new command exists.
7. Round all displayed numbers. Pure logic stays pure (no I/O in scheduler/strategies/autoscaler).
8. If a decision isn't specified here or in CLAUDE.md, pick the simplest maintainable option,
   note it in the phase summary, and keep going — don't stall.

## Tooling / dependencies (pin these in phase 0)

- Python 3.12, managed with `uv` (fallback: venv + pip). 
- Test: `pytest` + `pytest-asyncio` + `hypothesis` (property tests for the scheduler).
- Quality: `ruff` (lint+format), `mypy` (strict on `src/`).
- Runtime: `fastapi`, `uvicorn`, `prometheus-client`, `httpx`.
- Phase 6 real-model worker (host-native, lazy/optional import — NOT needed for phases 0-5):
  HuggingFace `transformers` + `torch` (MPS) recommended; `mlx-lm` optional.
- UI: React + TypeScript via Vite; charts via `recharts`; dev-proxy to the API.
- `Makefile` targets: `setup`, `test`, `lint`, `typecheck`, `dev`, `bench`, `up` (compose).

## Repo structure (create in phase 0; don't deviate)

```
inference-demo/
  CLAUDE.md  PLAN.md  pyproject.toml  Makefile  .pre-commit-config.yaml
  .github/workflows/ci.yml
  src/inference_demo/
    types.py            # shared contracts — the single source of truth
    workers/base.py     # Worker protocol
    sim/worker.py       # SimWorker
    scheduler.py        # continuous (in-flight) batching
    routing/base.py     # RoutingStrategy protocol
    routing/strategies.py
    routing/router.py
    autoscaler.py
    metrics.py          # prometheus + staged timings
    loadgen.py
    gateway/app.py      # FastAPI + SSE
    bench/static_vs_continuous.py
    workers/openai_worker.py   # phase 6
    workers/real_model_worker.py  # phase 6 (transformers+MPS recommended; or MLX)
  tests/                # mirrors src; one test module per unit
  ui/                   # React (Vite + TS) — phase 5
  docs/                 # future @imports
  scripts/  docker-compose.yml  Caddyfile   # phases 7+
```

## Shared contracts (define ONCE in `types.py`, phase 1 — prevents drift)

```python
SeqId = NewType("SeqId", str)
WorkerId = NewType("WorkerId", str)

class Priority(Enum): INTERACTIVE = "interactive"; BATCH = "batch"

@dataclass(frozen=True)
class Request:
    id: str
    prompt_tokens: int          # sim uses counts; real workers also carry the text
    max_tokens: int
    priority: Priority
    arrival_ts: float
    prefix_key: str | None      # for prefix-affinity routing
    prompt_text: str | None = None

@dataclass(frozen=True)
class TokenEvent:
    seq_id: SeqId
    is_final: bool
    ts: float

@dataclass(frozen=True)
class WorkerState:
    worker_id: WorkerId
    queue_depth: int
    pending_tokens: int         # estimated remaining work
    in_flight: int
    tok_per_s: float
    healthy: bool
    speed_profile: float        # 1.0 = baseline; <1 slow, >1 fast (hardware-aware)
    cached_prefixes: frozenset[str]
```

Every component consumes/produces these. Do not invent parallel shapes.

---

## PHASE 0 — Scaffold

Objective: a green, empty project with all tooling wired.
Tasks: init repo + git; pyproject + deps; Makefile; ruff/mypy config; pre-commit;
GitHub Actions CI running `lint + typecheck + test`; one trivial passing test.
DoD: `make test lint typecheck` all green; CI green on first push.
Verify: `make test && make lint && make typecheck`.
### CHECKPOINT 0

## PHASE 1 — Core types + SimWorker + scheduler + benchmark (the crown jewel)

Objective: the continuous-batching engine, fully tested, with the money graph — all off the sim.
Tests first:
- scheduler: admits up to max batch; evicts finished sequences each step; admits queued ones
  into freed slots same step; no sequence starves; accounting never mixes sequences. Use
  `hypothesis` to fuzz arrival/finish orders.
- SimWorker: honours latency/throughput profile; models a prefix-cache hit as a prefill speedup.
Then implement: `types.py`, `workers/base.py` (Worker protocol), `sim/worker.py`, `scheduler.py`.
Then `bench/static_vs_continuous.py`: drive identical skewed load through (a) static batching
(whole batch waits for the slowest) and (b) continuous batching; emit a throughput-vs-latency
graph (PNG/SVG) showing continuous wins at equal tail latency.
DoD: scheduler property tests green; benchmark produces the graph artifact in `bench/out/`.
Verify: `make test && make bench`.
### CHECKPOINT 1  (this alone is a shippable artifact)

## PHASE 2 — Router + all routing strategies

Objective: the full strategy spread, each pure and tested, behind one switchable router.
Tests first (per strategy, against fabricated `WorkerState` lists):
- random, round-robin (cycles), least-queue-depth, least-pending-tokens, power-of-two-choices
  (only ever picks the lighter of its two samples — seed the RNG), prefix-affinity (sends a
  known prefix_key to the worker holding it; falls back to least-loaded on miss),
  priority/SLA (interactive preferred onto least-loaded; batch backfills), hardware-aware
  (latency-sensitive -> higher speed_profile).
Then implement `routing/base.py`, `routing/strategies.py`, `routing/router.py` (hot-swap strategy
at runtime).
DoD: every strategy has tests proving its decision rule; router swaps strategy live in a test.
Verify: `make test`.
### CHECKPOINT 2

## PHASE 3 — Autoscaler (pure policy)

Objective: scale decisions as pure logic, testable in isolation.
Tests first: scale up when avg queue depth > target and below max; scale down when idle and
above min; respects min/max bounds; hysteresis/cooldown so it doesn't flap.
Then implement `autoscaler.py` as a pure `decide(pool_state) -> ScaleAction`. Wiring into a live
pool comes in phase 4.
DoD: autoscaler policy tests green incl. flap/cooldown.
Verify: `make test`.
### CHECKPOINT 3

## PHASE 4 — Runtime: gateway API + metrics + SSE + load generator

Objective: a running service that ties the pure pieces into a live system over sim workers.
Tasks: `metrics.py` (Prometheus `/metrics`; staged timings: TTFT, queue wait, prefill/decode,
p50/p99; per-worker tok/s + queue depth); `loadgen.py` (Poisson arrivals; concurrency; presets
steady/burst/spike); a `PoolManager` that runs N sim workers + the router + the autoscaler;
`gateway/app.py` (FastAPI): submit requests, switch strategy, control load-gen, get/set autoscaler
config, an SSE stream of live metrics + per-request routing decisions.
Tests: API contract tests with `httpx` (submit -> routed -> metrics update); SSE emits events;
strategy switch + autoscaler scale visible via the API.
DoD: `make dev` serves the API; `/metrics` populated; SSE streams under generated load; contract
tests green.
Verify: `make test`; manual `curl` of submit + `/metrics` + SSE (document the curls).
### CHECKPOINT 4

## PHASE 5 — React control console (the deployed UI)

Objective: the operator console from the spec in CLAUDE.md, wired to the phase-4 API/SSE.
Build: backend selector (Sim active; Endpoint/MLX present); endpoint config panel (URL + model +
test) — disabled unless local mode; load generator (presets + rate + concurrency + start/stop);
live routing-strategy switcher; autoscaler panel; worker-pool view (per-worker queue depth + tok/s);
metric cards (throughput, TTFT p50, p99, in-flight); recent-requests log; scenario buttons
("kill a worker", "switch strategy"). Consume the SSE stream for live updates.
Tests: component tests for the panels (Vitest + Testing Library); a smoke test that the dashboard
renders against a mocked SSE feed. No browser-storage APIs.
DoD: `npm run build` clean; UI renders live against the running API; scenario buttons work.
Verify: `make dev` + UI dev server; component tests green.
### CHECKPOINT 5  — END OF THE LARGELY-AUTONOMOUS RUN

>>> Everything below needs the human in the loop (real hardware / real infra). Do NOT attempt to
>>> "finish" these autonomously. Build the code, but expect to iterate WITH the human. <<<

## PHASE 6 — Real backends: OpenAIWorker + RealModelWorker (HUMAN-IN-LOOP)

Objective: plug real models in behind the same Worker interface. The framework decision lives
HERE, not earlier — phases 0-5 are fully backend-agnostic and need no ML framework.
Two ways to get a real model; pick based on how much you want to demonstrate:
  (a) Zero new framework: `OpenAIWorker` -> Ollama (`ollama run qwen2.5:0.5b`). Real model, but
      Ollama owns the batching, so this does NOT showcase ours. Fine if time is tight.
  (b) Your own batching on a real model (the showpiece): `real_model_worker.py` running a small
      model where YOU own the decode loop. RECOMMENDED: HuggingFace transformers + MPS (familiar,
      strong CV signal; drive decode manually via `past_key_values`). OPTIONAL: MLX/mlx-lm (faster
      on Apple Silicon, Apple-native, but a new framework to learn). Either way: batched decode
      over a batch dimension, per-sequence KV cache + attention masking so sequences don't
      cross-attend.
Keep the real-model import lazy so non-Mac / no-framework CI still passes.
Tests: OpenAIWorker against a mocked HTTP endpoint (runs in CI). RealModelWorker batched-decode
correctness can only be verified on the actual machine with a real model — gate behind a marker,
run manually.
HUMAN STEPS: install the chosen framework; load the small model; run the static-vs-continuous
benchmark for REAL and confirm it matches the sim's story; record the demo.
DoD: OpenAIWorker mock tests green in CI; (if doing (b)) real batched decode verified on-device.
### CHECKPOINT 6

## PHASE 7 — Docker + VPS deploy (HUMAN-IN-LOOP)

Objective: `docker compose up` for the control plane + sim/openai backends (NOT MLX); hosted demo.
Tasks: Dockerfiles + `docker-compose.yml` (control plane + UI + sim backend; openai backend via
env); Caddyfile for auto-HTTPS on a subdomain. SECURITY: public demo is sim-only by default; gate
controls (basic auth / secret link); hard caps on max workers + sim load + request rate; expose
ONLY the dashboard + its API, never Prometheus/internal ports; if custom endpoints are ever allowed
publicly, allowlist schemes + block private IP ranges (SSRF).
HUMAN STEPS: provision VPS, point DNS, deploy, verify the live URL, confirm caps hold.
DoD: `docker compose up` runs the full sim stack locally; hosted URL live and gated.
### CHECKPOINT 7

## PHASE 8 — README + demo + prior work

Objective: the artifact a reviewer reads.
Tasks: README with the 4 run modes; architecture diagram; the static-vs-continuous graph; an
embedded screen-recording of the live console; the honesty constraints (continuous batching is
non-paged; which modes prove batching vs only routing/observability); background section citing
Orca (Yu et al., OSDI 2022) and vLLM/PagedAttention (Kwon et al., SOSP 2023), and naming the
frontier (disaggregation, speculative decoding) for awareness; the hosted URL.
DoD: README reads cleanly to a reviewer with no prior context; all three artifacts (URL, recording,
repo) present so the story survives the VPS being down.
### CHECKPOINT 8 — done.

## Kickoff prompt (paste into Claude Code)

> Read CLAUDE.md, then PLAN.md. Follow the operating contract exactly. Start at PHASE 0 and stop
> at CHECKPOINT 0 with the verify output. Do not proceed past a checkpoint without my "continue".
> TDD throughout — failing tests first, shown failing, before any implementation.
EOF