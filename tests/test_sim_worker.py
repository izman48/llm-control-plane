"""SimWorker: the Worker protocol over the scheduler, plus a tunable timing model.

We test that it (a) honours its latency profile (deterministic timestamps) and
(b) models a prefix-cache hit as a prefill speedup.
"""

from __future__ import annotations

from inference_demo.sim.worker import SimProfile, SimWorker
from inference_demo.types import Priority, Request, TokenEvent, WorkerId
from inference_demo.workers.base import Worker


def _req(rid: str, *, prompt: int, out: int, prefix: str | None = None) -> Request:
    return Request(
        id=rid,
        prompt_tokens=prompt,
        max_tokens=out,
        priority=Priority.INTERACTIVE,
        arrival_ts=0.0,
        prefix_key=prefix,
    )


def _run_to_completion(worker: SimWorker, max_steps: int = 1000) -> list[TokenEvent]:
    events: list[TokenEvent] = []
    for _ in range(max_steps):
        if worker.is_idle():
            break
        events.extend(worker.step())
    assert worker.is_idle()
    return events


def test_satisfies_worker_protocol() -> None:
    worker = SimWorker(WorkerId("w0"), max_batch_size=4, profile=SimProfile())
    assert isinstance(worker, Worker)


def test_honours_timing_profile() -> None:
    # 100 prompt tokens at 100 tok/step -> exactly 1 prefill step; 3 output tokens.
    profile = SimProfile(step_s=0.01, prefill_tokens_per_step=100)
    worker = SimWorker(WorkerId("w0"), max_batch_size=1, profile=profile)
    worker.admit(_req("r1", prompt=100, out=3))

    events = _run_to_completion(worker)
    decode_events = [e for e in events if True]  # all events are decode tokens here

    # step1 admit, step2 prefill (no token), steps 3/4/5 decode -> ts 0.03/0.04/0.05
    assert [round(e.ts, 4) for e in decode_events] == [0.03, 0.04, 0.05]
    assert decode_events[-1].is_final
    assert not any(e.is_final for e in decode_events[:-1])


def test_prefix_cache_hit_speeds_up_prefill() -> None:
    profile = SimProfile(step_s=0.01, prefill_tokens_per_step=10, prefix_cache_speedup=4.0)
    worker = SimWorker(WorkerId("w0"), max_batch_size=1, profile=profile)

    # First request with prefix "sys": cold cache -> 40/10 = 4 prefill steps.
    worker.admit(_req("cold", prompt=40, out=1, prefix="sys"))
    cold_steps = _steps_to_final(worker)

    # Second request, same prefix: warm cache -> ceil(4/4) = 1 prefill step.
    worker.admit(_req("warm", prompt=40, out=1, prefix="sys"))
    warm_steps = _steps_to_final(worker)

    assert warm_steps < cold_steps
    # cold: admit + 4 prefill + 1 decode = 6; warm: admit + 1 prefill + 1 decode = 3
    assert (cold_steps, warm_steps) == (6, 3)


def test_no_speedup_without_matching_prefix() -> None:
    profile = SimProfile(step_s=0.01, prefill_tokens_per_step=10, prefix_cache_speedup=4.0)
    worker = SimWorker(WorkerId("w0"), max_batch_size=1, profile=profile)

    worker.admit(_req("a", prompt=40, out=1, prefix="alpha"))
    first = _steps_to_final(worker)
    worker.admit(_req("b", prompt=40, out=1, prefix="beta"))  # different prefix -> still cold
    second = _steps_to_final(worker)

    assert first == second == 6


def test_in_flight_tracks_running_not_queued() -> None:
    worker = SimWorker(WorkerId("w0"), max_batch_size=2, profile=SimProfile())
    for i in range(5):
        worker.admit(_req(f"r{i}", prompt=0, out=2))
    assert worker.in_flight() == 0  # nothing admitted into the batch until first step
    worker.step()
    assert worker.in_flight() == 2  # capped at max_batch_size


def _steps_to_final(worker: SimWorker, max_steps: int = 1000) -> int:
    """Count step() calls from now until a final token is emitted."""
    for n in range(1, max_steps + 1):
        events = worker.step()
        if any(e.is_final for e in events):
            return n
    raise AssertionError("no final token emitted")
