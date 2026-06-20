"""Regression guard for the headline claim: on a skewed workload, continuous
batching beats static batching on both throughput and tail latency.

This is the assertion behind the money graph — if it ever flips, the demo lies.
"""

from __future__ import annotations

from inference_demo.bench.static_vs_continuous import gen_workload, run_workload
from inference_demo.scheduler import ContinuousScheduler, StaticScheduler
from inference_demo.sim.worker import SimProfile


def test_continuous_beats_static_on_skewed_load() -> None:
    profile = SimProfile(step_s=0.01, prefill_tokens_per_step=128)
    # Skewed output lengths (~10x spread) are where static batching bleeds.
    arrivals = gen_workload(n=120, mean_gap_steps=2, seed=7, skew=True)

    cont = run_workload(
        arrivals, max_batch_size=8, profile=profile, scheduler_cls=ContinuousScheduler
    )
    stat = run_workload(arrivals, max_batch_size=8, profile=profile, scheduler_cls=StaticScheduler)

    assert cont.total_tokens == stat.total_tokens  # same work, fair comparison
    assert cont.makespan_s < stat.makespan_s
    assert cont.throughput_tok_s > stat.throughput_tok_s
    assert cont.p99_latency_s < stat.p99_latency_s
