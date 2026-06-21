"""Real-model static-vs-continuous benchmark — the sim's money graph, on hardware.

Drives an identical skewed workload through ONE loaded model under our static and
continuous batching policies, measuring real wall-clock throughput and tail
latency. Confirms the sim's structural story holds on a real model (Qwen2.5-0.5B,
transformers + MPS).

Host-native only; needs the ``realmodel`` extra. Run:

    uv sync --extra dev --extra realmodel
    uv run python -m inference_demo.bench.real_static_vs_continuous

Writes real_static_vs_continuous.{svg,md} to src/inference_demo/bench/out/.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from inference_demo.stats import percentile
from inference_demo.types import Priority, Request, SeqId, WorkerId
from inference_demo.workers.real_model_worker import DEFAULT_MODEL, RealModelWorker

OUT_DIR = Path(__file__).parent / "out"

_PROMPTS = [
    "Write one sentence about the ocean.",
    "List three fruits.",
    "Explain gravity simply.",
    "Name a famous scientist.",
    "Describe a sunny day.",
    "Give a tip for studying.",
]


@dataclass(frozen=True)
class RunStats:
    policy: str
    makespan_s: float
    throughput_tok_s: float
    p50_latency_s: float
    p99_latency_s: float
    n_requests: int
    total_tokens: int


def make_workload(n: int, seed: int) -> list[Request]:
    rng = random.Random(seed)
    reqs: list[Request] = []
    for i in range(n):
        long_tail = rng.random() < 0.3  # ~10x output-length spread
        out = rng.randint(96, 160) if long_tail else rng.randint(8, 16)
        reqs.append(
            Request(
                id=f"r{i}",
                prompt_tokens=0,
                max_tokens=out,
                priority=Priority.INTERACTIVE,
                arrival_ts=0.0,
                prefix_key=None,
                prompt_text=rng.choice(_PROMPTS),
            )
        )
    return reqs


def run_policy(worker: RealModelWorker, reqs: list[Request], *, continuous: bool) -> RunStats:
    worker.reset()
    worker.continuous = continuous
    t0 = time.perf_counter()
    for r in reqs:
        worker.admit(r)
    completion: dict[str, float] = {}
    while not worker.is_idle():
        for ev in worker.step():
            if ev.is_final:
                completion[str(ev.seq_id)] = time.perf_counter()
    total_tokens = sum(len(worker.generated_ids(SeqId(r.id))) for r in reqs)
    latencies = sorted(completion[r.id] - t0 for r in reqs)  # all admitted at t0 (burst)
    makespan = max(completion.values()) - t0
    return RunStats(
        policy="continuous" if continuous else "static",
        makespan_s=makespan,
        throughput_tok_s=total_tokens / makespan if makespan else 0.0,
        p50_latency_s=percentile(latencies, 50),
        p99_latency_s=percentile(latencies, 99),
        n_requests=len(reqs),
        total_tokens=total_tokens,
    )


def _bars(stat: RunStats, cont: RunStats) -> str:
    w, h, pad = 720, 320, 60
    groups = [
        ("throughput (tok/s)", stat.throughput_tok_s, cont.throughput_tok_s, False),
        ("p50 latency (s)", stat.p50_latency_s, cont.p50_latency_s, True),
        ("p99 latency (s)", stat.p99_latency_s, cont.p99_latency_s, True),
    ]
    gw = (w - 2 * pad) / len(groups)
    bars: list[str] = []
    for gi, (label, sval, cval, lower_better) in enumerate(groups):
        top = max(sval, cval) or 1.0
        gx = pad + gi * gw
        for bi, (val, color, name) in enumerate(
            [(sval, "#d62728", "static"), (cval, "#2ca02c", "continuous")]
        ):
            bh = (val / top) * (h - 2 * pad)
            bx = gx + 20 + bi * 55
            by = h - pad - bh
            bars.append(
                f'<rect x="{bx:.0f}" y="{by:.0f}" width="48" height="{bh:.0f}" fill="{color}"/>'
            )
            bars.append(
                f'<text x="{bx + 24:.0f}" y="{by - 6:.0f}" text-anchor="middle" '
                f'font-size="12">{val:.1f}</text>'
            )
            if gi == 0:
                ly = 30 + bi * 22
                bars.append(
                    f'<rect x="{w - 200}" y="{ly}" width="14" height="14" fill="{color}"/>'
                    f'<text x="{w - 180}" y="{ly + 12}" font-size="13">{name}</text>'
                )
        hint = "lower is better" if lower_better else "higher is better"
        bars.append(
            f'<text x="{gx + gw / 2:.0f}" y="{h - pad + 18:.0f}" text-anchor="middle" '
            f'font-size="12">{label}</text>'
        )
        bars.append(
            f'<text x="{gx + gw / 2:.0f}" y="{h - pad + 33:.0f}" text-anchor="middle" '
            f'font-size="10" fill="#888">{hint}</text>'
        )
    body = "\n  ".join(bars)
    return (
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">\n'
        f'  <rect width="{w}" height="{h}" fill="white"/>\n'
        f'  <text x="{w / 2}" y="20" text-anchor="middle" font-size="16" font-weight="bold">'
        f"Real model (Qwen2.5-0.5B, MPS): static vs continuous batching</text>\n  {body}\n</svg>\n"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reqs = make_workload(n=32, seed=7)
    print(f"loading {DEFAULT_MODEL} ...")
    worker = RealModelWorker(WorkerId("bench"), max_batch_size=8)

    print("warmup ...")
    worker.reset()
    worker.admit(make_workload(1, 0)[0])
    while not worker.is_idle():
        worker.step()

    print("running static ...")
    stat = run_policy(worker, reqs, continuous=False)
    print("running continuous ...")
    cont = run_policy(worker, reqs, continuous=True)

    (OUT_DIR / "real_static_vs_continuous.svg").write_text(_bars(stat, cont))
    lines = [
        "# Real-model static vs continuous batching\n",
        f"Model: {DEFAULT_MODEL} on MPS. {stat.n_requests} requests, skewed output lengths, "
        "burst arrival, max batch 8.\n",
        "| policy | throughput (tok/s) | p50 latency (s) | p99 latency (s) | makespan (s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in (stat, cont):
        lines.append(
            f"| {r.policy} | {r.throughput_tok_s:.1f} | {r.p50_latency_s:.2f} | "
            f"{r.p99_latency_s:.2f} | {r.makespan_s:.2f} |"
        )
    summary = "\n".join(lines) + "\n"
    (OUT_DIR / "real_static_vs_continuous.md").write_text(summary)
    print("\n" + summary)
    print(
        f"continuous vs static — throughput {cont.throughput_tok_s / stat.throughput_tok_s:.2f}x, "
        f"p99 {stat.p99_latency_s / cont.p99_latency_s:.2f}x lower"
    )
    print(f"artifacts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
