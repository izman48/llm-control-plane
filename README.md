# LLM Inference Control Plane (toy-scale, honest)

A small but real LLM inference serving system: a **control plane** — router + autoscaler +
live observability console — sitting on top of model workers that do **continuous (in-flight)
batching**. Built to demonstrate distributed-systems competence for inference serving.

> **Status:** Phase 0 (scaffold). See [`PLAN.md`](PLAN.md) for the phase-by-phase build and
> [`CLAUDE.md`](CLAUDE.md) for the design constitution.

## The idea

Serving one LLM to many users is slow because requests queue behind each other. Real systems fix
this with (1) **batching** — running many requests through the model together — and (2) **multiple
replicas with intelligent routing + autoscaling**. This project is an honest, toy-scale
implementation of that control plane, with a swappable `Worker` backend (sim / OpenAI-compatible /
real model) so the whole spine is testable deterministically and runs anywhere.

The contribution is the **control-plane layer** (routing, scheduling, autoscaling, observability).
Per-replica batching is ours (real-model mode) or the external server's (vLLM/Ollama mode).

## Honesty constraints

- Our batching is **continuous (in-flight) batching, non-paged** — not PagedAttention.
- Hosted/sim and OpenAI-backend modes prove routing/autoscaling/observability, **not** our
  batching. Our batching is shown only in real-model mode + the static-vs-continuous benchmark.

## Development

Requires Python 3.12 and [`uv`](https://github.com/astral-sh/uv).

```bash
make setup       # uv sync --extra dev
make test        # pytest
make lint        # ruff check + format --check
make typecheck   # mypy (strict on src/)
```

## Prior work

- **Orca** — iteration-level scheduling (continuous batching). Yu et al., OSDI 2022.
- **vLLM / PagedAttention** — KV-cache memory management. Kwon et al., SOSP 2023.
