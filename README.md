# LLM Inference Control Plane (toy-scale, honest)

A small but real LLM inference serving system: a **control plane** — router + autoscaler +
live observability console — sitting on top of model workers that do **continuous (in-flight)
batching**. Built to demonstrate distributed-systems competence for inference serving.

> **Status:** the autonomous spine (Phases 0–5) is **complete and CI-green** — router +
> strategies, scheduler/batching, autoscaler, observability, load generator, live console, and
> the static-vs-continuous benchmark. Phases 6–8 (real model backend, deploy, demo) are
> human-in-the-loop and in progress. See [`PLAN.md`](PLAN.md) for the phase-by-phase build and
> [`CLAUDE.md`](CLAUDE.md) for the design constitution.

## The idea

Serving one LLM to many users is slow because requests queue behind each other and a single model
handles them one at a time. Real systems fix this with (1) **batching** — running many requests
through the model together, amortizing the expensive weight-load — and (2) **multiple replicas
with intelligent routing + autoscaling**. This project is an honest, toy-scale implementation of
that control plane.

**Thesis:** given a fixed pool of workers and a mixed workload, intelligently route, schedule, and
autoscale to maximize useful throughput while protecting interactive tail latency — and prove
every decision with before/after observability. The **control plane is the centre of gravity**,
not the batching.

## The swappable Worker backend (the key seam)

Everything above the worker (router, scheduler, autoscaler, metrics, UI) is backend-agnostic,
depending only on one `Worker` protocol. This is the demo-portability story, the
hardware-agnostic story, and what makes the whole control plane deterministically testable.

| backend | what it is | status |
| --- | --- | --- |
| **SimWorker** | no real model; tunable latency/throughput + modelled prefix-cache hit. Runs anywhere, fakes hundreds of workers. | ✅ built (Phase 1) |
| **OpenAIWorker** | points at any OpenAI-compatible endpoint (Ollama/vLLM/LM Studio). Exercises routing/observability, *not* our batching. | ✅ built (Phase 6), mock-tested in CI |
| **RealModelWorker** | a real small model (Qwen2.5-0.5B) with **our** continuous batched decode loop (transformers + MPS). | ✅ built (Phase 6), verified on-device |

```python
class Worker(Protocol):
    def admit(self, req: Request) -> SeqId: ...
    def step(self) -> list[TokenEvent]: ...   # advance the running batch one decode step
    def in_flight(self) -> int: ...

class RoutingStrategy(Protocol):
    def choose(self, states: list[WorkerState], req: Request) -> WorkerId: ...
```

The scheduler and every routing strategy are **pure** (state in → decision out, no model/network),
so they're tested exhaustively against fabricated states.

## Architecture

```
                         ┌──────────────────────────────────────────┐
   load generator  ───▶  │  Gateway (FastAPI)  /api/*  ·  SSE  ·  /metrics │
   (steady/burst/spike)  └───────────────┬──────────────────────────┘
                                         │
                                  ┌──────▼───────┐      ┌───────────────┐
                                  │   Router     │◀────▶│  RoutingStrategy │ (8, hot-swappable)
                                  └──────┬───────┘      └───────────────┘
                                         │ choose(worker_states, req)
                 ┌───────────────────────┼───────────────────────┐
            ┌────▼────┐             ┌─────▼───┐              ┌─────▼───┐
            │ Worker  │   …         │ Worker  │     …        │ Worker  │   ← Autoscaler adds/removes
            │ (scheduler: continuous batching) │             └─────────┘      (pure decide() policy)
            └─────────┘             └─────────┘
                 └──────────────── Metrics (TTFT, p50/p99, throughput) ───────────┘
```

## The money graph: static vs continuous batching

`make bench` drives an identical skewed workload through one SimWorker under each scheduling
policy and writes a throughput-vs-tail-latency graph + summary to
`src/inference_demo/bench/out/`. On a ~10× output-length spread:

| metric | static batching | continuous batching |
| --- | --- | --- |
| peak throughput | 224 tok/s | **722 tok/s (3.2×)** |
| p99 latency (light load) | 72.4 s | **7.9 s** |
| makespan | 96.1 s | **30.4 s** |

A `test_bench` guard fails CI if this claim ever flips. These are **deterministic-sim**
numbers — the sim's job is to prove the *structural* win (slot reuse vs whole-batch-wait).

**Confirmed on a real model** (`make bench-real`, Qwen2.5-0.5B on MPS, 32 skewed requests,
our own continuous batched decode loop):

| metric | static | continuous |
| --- | --- | --- |
| throughput | 53 tok/s | **86 tok/s (1.62×)** |
| p99 latency | 19.8 s | **13.1 s (1.51× lower)** |

Same structural win on hardware — smaller margin than the sim because real prefill/decode costs
differ, which is exactly the point of re-running it for real.

## Routing strategies (the contrast is the demo)

Naive baselines (the villains) make the smart strategies legible:

- **random**, **round-robin** — ignore load; break under skew.
- **least-queue-depth**, **least-pending-tokens** (token-aware, not request-count),
  **power-of-two-choices** (near-optimal at O(1), dodges thundering-herd).
- **prefix-affinity** (route to the KV-cache holder — serving is stateful, cf. GORGO),
  **priority/SLA** (interactive jumps ahead; batch backfills), **hardware-aware**
  (latency-sensitive → fast workers; throughput → slow ones).

Switch any of them live from the console and watch p99 / cache-hit move.

## Quickstart

Requires Python 3.12 + [`uv`](https://github.com/astral-sh/uv) (and Node/npm for the UI).

```bash
make setup        # uv sync --extra dev (create venv + install deps)
make test         # pytest — scheduler property tests, SimWorker, routing, autoscaler, API
make lint         # ruff check + format --check
make typecheck    # mypy (strict on src/)
make bench        # produce the static-vs-continuous money graph
```

### Run the live control plane

Two terminals (the UI proxies API calls to the backend):

```bash
# terminal 1 — backend gateway on http://127.0.0.1:8000
make dev

# terminal 2 — React console on http://localhost:5273
make ui-install   # first time only
make ui-dev
```

Open **http://localhost:5273**, click **Load generator → Start** (or the **Traffic spike**
scenario), and watch the metrics, worker-pool view, and autoscaler react. Switch routing
strategies live; hit **Kill a worker** for a clean before/after moment.

Key endpoints: `POST /api/submit`, `/api/loadgen`, `/api/strategy`, `/api/autoscaler`,
`/api/workers/kill`; `GET /api/snapshot`, `/api/stream` (SSE), `/metrics` (Prometheus).

## Honesty constraints

- Our batching is **continuous (in-flight) batching, non-paged** — not PagedAttention.
- Sim and OpenAI-backend modes prove **routing/autoscaling/observability, not our batching**.
  Our batching is shown only in real-model mode + the static-vs-continuous benchmark.
- The benchmark's absolute numbers are sim-defined; they demonstrate the structural win, not a
  hardware claim.

## Roadmap

- **Phase 6** ✅ — `OpenAIWorker` (mock-tested in CI) + `RealModelWorker` with our own continuous
  batched decode on a real model, verified on-device (greedy matches HF `generate()`; batched
  decode matches per-sequence decode; mid-flight KV-cache merge correct). The real
  static-vs-continuous benchmark confirms the sim's story (1.62× throughput, 1.51× lower p99).
- **Phase 7** — Docker Compose + VPS deploy (sim-only public demo, gated controls, hard caps).
- **Phase 8** — architecture diagram, recorded demo, hosted URL, full prior-work writeup.

## Prior work

- **Orca** — iteration-level scheduling (continuous batching). Yu et al., OSDI 2022.
- **vLLM / PagedAttention** — KV-cache memory management. Kwon et al., SOSP 2023.
- **GORGO** — KV-cache-reuse-aware routing (reference for prefix-affinity).
