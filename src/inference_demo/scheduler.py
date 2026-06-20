"""Continuous (in-flight) batching scheduler — the engine, as pure logic.

A sequence's life in the batch is ``prefill_steps`` steps (loading the prompt,
emitting no output token) followed by ``decode_steps`` steps (one output token
each). The scheduler decides, every step, which queued sequences enter the batch
and which finished ones leave it. It carries no notion of wall-clock time and
performs no I/O — timing is layered on top by the worker (see sim/worker.py).

Two policies share the advance/evict machinery and differ only in admission:

* ``ContinuousScheduler`` — refill freed slots every step (iteration-level
  scheduling, à la Orca). Short sequences leave and new ones enter immediately.
* ``StaticScheduler`` — the demo villain: admit a whole batch, then refuse to
  admit anything until the batch has fully drained (whole batch waits for the
  slowest), leaving slots idle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

from inference_demo.types import SeqId


@dataclass(frozen=True)
class SeqWork:
    """Immutable description of a sequence's workload, in scheduler steps."""

    seq_id: SeqId
    prefill_steps: int
    decode_steps: int

    def __post_init__(self) -> None:
        if self.prefill_steps < 0:
            raise ValueError("prefill_steps must be >= 0")
        if self.decode_steps < 1:
            raise ValueError("decode_steps must be >= 1 (a sequence must emit a token)")


@dataclass
class _Running:
    seq_id: SeqId
    prefill_left: int
    decode_left: int


@dataclass(frozen=True)
class StepResult:
    """What happened in a single scheduler step."""

    admitted: list[SeqId]  # sequences that entered the batch this step
    tokens: list[SeqId]  # sequences that emitted an output token this step
    finished: list[SeqId]  # sequences that completed and left the batch this step


class _BaseScheduler(ABC):
    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.max_batch_size = max_batch_size
        self._waiting: deque[SeqWork] = deque()
        self._running: dict[SeqId, _Running] = {}

    def enqueue(self, seq: SeqWork) -> None:
        self._waiting.append(seq)

    def in_flight(self) -> int:
        return len(self._running)

    def queue_depth(self) -> int:
        return len(self._waiting)

    def is_idle(self) -> bool:
        return not self._waiting and not self._running

    def pending_decode_tokens(self) -> int:
        """Estimated output tokens still owed across running + waiting work."""
        running = sum(r.decode_left for r in self._running.values())
        waiting = sum(w.decode_steps for w in self._waiting)
        return running + waiting

    def step(self) -> StepResult:
        tokens = self._advance()
        finished = self._evict()
        admitted = self._admit()
        return StepResult(admitted=admitted, tokens=tokens, finished=finished)

    def _advance(self) -> list[SeqId]:
        """Advance every running sequence one step; return those that emitted."""
        tokens: list[SeqId] = []
        for r in self._running.values():
            if r.prefill_left > 0:
                r.prefill_left -= 1  # prefill emits no output token
            elif r.decode_left > 0:
                r.decode_left -= 1
                tokens.append(r.seq_id)
        return tokens

    def _evict(self) -> list[SeqId]:
        """Remove fully-finished sequences; return their ids (order preserved)."""
        finished = [
            sid for sid, r in self._running.items() if r.prefill_left == 0 and r.decode_left == 0
        ]
        for sid in finished:
            del self._running[sid]
        return finished

    def _fill_free_slots(self) -> list[SeqId]:
        admitted: list[SeqId] = []
        while len(self._running) < self.max_batch_size and self._waiting:
            w = self._waiting.popleft()
            self._running[w.seq_id] = _Running(
                seq_id=w.seq_id, prefill_left=w.prefill_steps, decode_left=w.decode_steps
            )
            admitted.append(w.seq_id)
        return admitted

    @abstractmethod
    def _admit(self) -> list[SeqId]: ...


class ContinuousScheduler(_BaseScheduler):
    """Refill freed slots every step — the real continuous-batching policy."""

    def _admit(self) -> list[SeqId]:
        return self._fill_free_slots()


class StaticScheduler(_BaseScheduler):
    """Admit a full batch, then wait for it to drain before admitting the next."""

    def _admit(self) -> list[SeqId]:
        if self._running:
            return []  # batch still in flight — do not refill (slots sit idle)
        return self._fill_free_slots()
