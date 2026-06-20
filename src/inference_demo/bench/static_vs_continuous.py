"""The money graph: static vs continuous batching on identical skewed load.

Drives the SAME workload through a SimWorker under each scheduling policy and
measures achieved throughput and tail (p99) latency. Sweeping offered load
produces the canonical inference plot — throughput (x) vs p99 latency (y) — on
which continuous batching sustains more throughput at equal tail latency.

This is a deterministic SIM (no real model); phase 6 re-runs the same story on a
real model to confirm the sim is honest. Run via ``make bench``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from inference_demo.scheduler import ContinuousScheduler, StaticScheduler, _BaseScheduler
from inference_demo.sim.worker import SimProfile, SimWorker
from inference_demo.types import Priority, Request, WorkerId

OUT_DIR = Path(__file__).parent / "out"


@dataclass(frozen=True)
class Arrival:
    arrival_step: int
    req: Request


@dataclass(frozen=True)
class RunStats:
    policy: str
    makespan_s: float
    throughput_tok_s: float
    p50_latency_s: float
    p99_latency_s: float
    n_requests: int
    total_tokens: int


def gen_workload(*, n: int, mean_gap_steps: float, seed: int, skew: bool) -> list[Arrival]:
    """Poisson-ish arrivals with (optionally) skewed output lengths (~10x spread)."""
    rng = random.Random(seed)
    arrivals: list[Arrival] = []
    step = 0
    for i in range(n):
        step += max(0, round(rng.expovariate(1.0 / mean_gap_steps)))
        long_tail = skew and rng.random() < 0.2  # the long tail that starves a static batch
        out = rng.randint(200, 400) if long_tail else rng.randint(8, 32)
        prompt = rng.randint(16, 512)
        req = Request(
            id=f"r{i}",
            prompt_tokens=prompt,
            max_tokens=out,
            priority=Priority.INTERACTIVE,
            arrival_ts=0.0,
            prefix_key=None,
        )
        arrivals.append(Arrival(arrival_step=step, req=req))
    return arrivals


def run_workload(
    arrivals: list[Arrival],
    *,
    max_batch_size: int,
    profile: SimProfile,
    scheduler_cls: type[_BaseScheduler],
) -> RunStats:
    worker = SimWorker(
        worker_id=WorkerId("sim-0"),
        max_batch_size=max_batch_size,
        profile=profile,
        scheduler_cls=scheduler_cls,
    )
    by_step: dict[int, list[Arrival]] = {}
    for a in arrivals:
        by_step.setdefault(a.arrival_step, []).append(a)
    arrival_clock = {a.req.id: a.arrival_step * profile.step_s for a in arrivals}

    last_arrival = max(a.arrival_step for a in arrivals)
    completion: dict[str, float] = {}

    step_index = 0
    while step_index <= last_arrival or not worker.is_idle():
        step_index += 1
        for a in by_step.get(step_index, ()):
            worker.admit(a.req)
        for ev in worker.step():
            if ev.is_final:
                completion[str(ev.seq_id)] = ev.ts

    latencies = sorted(completion[rid] - arrival_clock[rid] for rid in completion)
    makespan = max(completion.values()) - min(arrival_clock.values())
    total_tokens = sum(a.req.max_tokens for a in arrivals)
    return RunStats(
        policy=scheduler_cls.__name__.replace("Scheduler", "").lower(),
        makespan_s=makespan,
        throughput_tok_s=total_tokens / makespan if makespan else 0.0,
        p50_latency_s=_percentile(latencies, 50),
        p99_latency_s=_percentile(latencies, 99),
        n_requests=len(arrivals),
        total_tokens=total_tokens,
    )


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    rank = max(1, math.ceil(pct / 100 * len(sorted_vals)))
    return sorted_vals[rank - 1]


# ---- artifact generation ---------------------------------------------------


def _sweep() -> tuple[list[RunStats], list[RunStats]]:
    profile = SimProfile(step_s=0.01, prefill_tokens_per_step=128)
    cont, stat = [], []
    for gap in (8.0, 5.0, 3.0, 2.0, 1.5, 1.0):  # decreasing gap = increasing load
        arrivals = gen_workload(n=300, mean_gap_steps=gap, seed=42, skew=True)
        cont.append(
            run_workload(
                arrivals, max_batch_size=8, profile=profile, scheduler_cls=ContinuousScheduler
            )
        )
        stat.append(
            run_workload(arrivals, max_batch_size=8, profile=profile, scheduler_cls=StaticScheduler)
        )
    return cont, stat


def _render_svg(cont: list[RunStats], stat: list[RunStats]) -> str:
    w, h, pad = 720, 460, 70
    xs = [s.throughput_tok_s for s in cont + stat]
    ys = [s.p99_latency_s for s in cont + stat]
    x_max, y_max = max(xs) * 1.05, max(ys) * 1.05

    def px(v: float) -> float:
        return pad + (v / x_max) * (w - 2 * pad)

    def py(v: float) -> float:
        return (h - pad) - (v / y_max) * (h - 2 * pad)

    def poly(series: list[RunStats], color: str, label: str, dy: int) -> str:
        parts: list[str] = []
        pts = " ".join(f"{px(s.throughput_tok_s):.1f},{py(s.p99_latency_s):.1f}" for s in series)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for s in series:
            cx, cy = px(s.throughput_tok_s), py(s.p99_latency_s)
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{color}"/>')
        parts.append(f'<rect x="{w - 240}" y="{dy - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{w - 220}" y="{dy + 2}" font-size="14" fill="#333">{label}</text>')
        return "".join(parts)

    title = "Static vs continuous batching — throughput vs tail latency"
    cx_mid, cy_mid = w / 2, h / 2
    elements = [
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="{cx_mid}" y="30" text-anchor="middle" font-size="18"'
        f' font-weight="bold">{title}</text>',
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="#999"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="#999"/>',
        f'<text x="{cx_mid}" y="{h - 20}" text-anchor="middle" font-size="14">'
        "throughput (tokens/s)  →</text>",
        f'<text x="20" y="{cy_mid}" text-anchor="middle" font-size="14"'
        f' transform="rotate(-90 20 {cy_mid})">p99 latency (s)  →</text>',
        poly(stat, "#d62728", "static (waits for slowest)", 70),
        poly(cont, "#2ca02c", "continuous (refills slots)", 95),
        f'<text x="{w - pad}" y="{h - pad + 18}" text-anchor="end" font-size="11"'
        ' fill="#666">further right + lower = better</text>',
    ]
    body = "\n  ".join(elements)
    header = (
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">'
    )
    return f"{header}\n  {body}\n</svg>\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cont, stat = _sweep()

    (OUT_DIR / "static_vs_continuous.svg").write_text(_render_svg(cont, stat))

    lines = [
        "# Static vs continuous batching\n",
        "Identical skewed workload through one SimWorker under each policy.\n",
        "| policy | offered load pt | throughput (tok/s) | p50 (s) | p99 (s) | makespan (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i, (c, s) in enumerate(zip(cont, stat, strict=True)):
        for r in (s, c):
            lines.append(
                f"| {r.policy} | {i} | {r.throughput_tok_s:.0f} | "
                f"{r.p50_latency_s:.2f} | {r.p99_latency_s:.2f} | {r.makespan_s:.2f} |"
            )
    summary = "\n".join(lines) + "\n"
    (OUT_DIR / "static_vs_continuous.md").write_text(summary)

    peak_c = max(c.throughput_tok_s for c in cont)
    peak_s = max(s.throughput_tok_s for s in stat)
    print(summary)
    print(
        f"peak throughput — continuous: {peak_c:.0f} tok/s, static: {peak_s:.0f} tok/s "
        f"({peak_c / peak_s:.2f}x)"
    )
    print(f"artifacts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
