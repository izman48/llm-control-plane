"""Scheduler is the TDD crown jewel: pure continuous-batching policy + accounting.

No time, no I/O — fabricated SeqWork in, StepResult out. We test the invariants
from PLAN.md: admit up to max batch, evict finished each step, admit into freed
slots the same step, no starvation, and accounting that never mixes sequences.
"""

from __future__ import annotations

from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from inference_demo.scheduler import (
    ContinuousScheduler,
    SeqWork,
    StaticScheduler,
    StepResult,
)
from inference_demo.types import SeqId


def _work(name: str, prefill: int, decode: int) -> SeqWork:
    return SeqWork(seq_id=SeqId(name), prefill_steps=prefill, decode_steps=decode)


def _drain(
    sched: ContinuousScheduler | StaticScheduler, max_steps: int = 10_000
) -> list[StepResult]:
    """Step until idle; return every StepResult. Guards against non-termination."""
    results: list[StepResult] = []
    for _ in range(max_steps):
        if sched.is_idle():
            break
        results.append(sched.step())
    assert sched.is_idle(), "scheduler did not drain — possible starvation/livelock"
    return results


# ---- continuous: targeted behaviour ----------------------------------------


def test_admits_up_to_max_batch() -> None:
    sched = ContinuousScheduler(max_batch_size=2)
    for i in range(5):
        sched.enqueue(_work(f"s{i}", prefill=0, decode=3))
    res = sched.step()
    assert len(res.admitted) == 2
    assert sched.in_flight() == 2


def test_evicts_finished_and_admits_into_freed_slot_same_step() -> None:
    # max_batch=1: A and B each finish in one decode step.
    sched = ContinuousScheduler(max_batch_size=1)
    sched.enqueue(_work("A", prefill=0, decode=1))
    sched.enqueue(_work("B", prefill=0, decode=1))

    r1 = sched.step()  # nothing running yet -> admit A
    assert r1.admitted == [SeqId("A")]
    assert r1.finished == []

    r2 = sched.step()  # A emits its final token, is evicted, AND B fills the slot
    assert r2.tokens == [SeqId("A")]
    assert r2.finished == [SeqId("A")]
    assert r2.admitted == [SeqId("B")]  # freed slot reused the same step


def test_prefill_emits_no_token_then_decode_does() -> None:
    sched = ContinuousScheduler(max_batch_size=1)
    sched.enqueue(_work("A", prefill=2, decode=1))
    sched.step()  # admit A
    assert sched.step().tokens == []  # prefill step 1
    assert sched.step().tokens == []  # prefill step 2
    last = sched.step()  # decode -> final token
    assert last.tokens == [SeqId("A")]
    assert last.finished == [SeqId("A")]


def test_accounting_never_mixes_sequences() -> None:
    sched = ContinuousScheduler(max_batch_size=3)
    plan = {"A": 1, "B": 4, "C": 2, "D": 3}
    for name, decode in plan.items():
        sched.enqueue(_work(name, prefill=1, decode=decode))

    emitted: Counter[str] = Counter()
    finished: list[str] = []
    for res in _drain(sched):
        for sid in res.tokens:
            emitted[str(sid)] += 1
        finished.extend(str(s) for s in res.finished)

    assert emitted == Counter(plan)  # each seq emits exactly its decode budget
    assert sorted(finished) == sorted(plan)  # each finishes exactly once


# ---- static: the demo villain (whole batch waits for the slowest) -----------


def test_static_does_not_refill_until_batch_drains() -> None:
    sched = StaticScheduler(max_batch_size=2)
    sched.enqueue(_work("A", prefill=0, decode=1))  # short
    sched.enqueue(_work("B", prefill=0, decode=3))  # long
    sched.enqueue(_work("C", prefill=0, decode=1))  # waits for the whole batch

    sched.step()  # admit A, B together
    r2 = sched.step()  # A finishes; slot frees but C must NOT be admitted yet
    assert r2.finished == [SeqId("A")]
    assert r2.admitted == []
    assert sched.in_flight() == 1  # wasted capacity: B runs alone, C waits

    sched.step()  # B decodes
    r4 = sched.step()  # B finishes -> batch empty -> C finally admitted
    assert r4.finished == [SeqId("B")]
    assert r4.admitted == [SeqId("C")]


def test_continuous_refills_immediately_unlike_static() -> None:
    sched = ContinuousScheduler(max_batch_size=2)
    sched.enqueue(_work("A", prefill=0, decode=1))
    sched.enqueue(_work("B", prefill=0, decode=3))
    sched.enqueue(_work("C", prefill=0, decode=1))

    sched.step()  # admit A, B
    r2 = sched.step()  # A finishes AND C is admitted into the freed slot
    assert r2.finished == [SeqId("A")]
    assert r2.admitted == [SeqId("C")]


# ---- property tests: fuzz arrival/finish orders ----------------------------

_work_lists = st.lists(
    st.tuples(st.integers(min_value=0, max_value=3), st.integers(min_value=1, max_value=5)),
    min_size=1,
    max_size=20,
)


@settings(max_examples=200)
@given(specs=_work_lists, max_batch=st.integers(min_value=1, max_value=4))
def test_continuous_invariants(specs: list[tuple[int, int]], max_batch: int) -> None:
    sched = ContinuousScheduler(max_batch_size=max_batch)
    plan = {f"s{i}": decode for i, (_, decode) in enumerate(specs)}
    for i, (prefill, decode) in enumerate(specs):
        sched.enqueue(_work(f"s{i}", prefill=prefill, decode=decode))

    emitted: Counter[str] = Counter()
    for res in _drain(sched):
        assert sched.in_flight() <= max_batch  # never over-admit
        for sid in res.tokens:
            emitted[str(sid)] += 1

    assert emitted == Counter(plan)  # all finish, none starves, none exceeds budget


@settings(max_examples=200)
@given(specs=_work_lists, max_batch=st.integers(min_value=1, max_value=4))
def test_static_completes_everything_too(specs: list[tuple[int, int]], max_batch: int) -> None:
    sched = StaticScheduler(max_batch_size=max_batch)
    plan = {f"s{i}": decode for i, (_, decode) in enumerate(specs)}
    for i, (prefill, decode) in enumerate(specs):
        sched.enqueue(_work(f"s{i}", prefill=prefill, decode=decode))

    emitted: Counter[str] = Counter()
    for res in _drain(sched):
        assert sched.in_flight() <= max_batch
        for sid in res.tokens:
            emitted[str(sid)] += 1

    assert emitted == Counter(plan)
