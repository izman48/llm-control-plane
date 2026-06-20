"""SimWorker — a model-free Worker with a tunable latency/throughput profile.

It owns a scheduler (continuous by default) and layers a deterministic timing
model on top: every ``step()`` advances a sim clock by ``profile.step_s`` and
stamps the tokens emitted that step. Prompt length maps to prefill steps via
``prefill_tokens_per_step``; a request whose ``prefix_key`` the worker has seen
before prefills ``prefix_cache_speedup``x faster (a modelled KV-cache hit).

Being model-free, it runs anywhere, makes the whole control plane deterministically
testable, and can fake hundreds of workers for scale demos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from inference_demo.scheduler import ContinuousScheduler, SeqWork, _BaseScheduler
from inference_demo.types import Request, SeqId, TokenEvent, WorkerId, WorkerState


@dataclass(frozen=True)
class SimProfile:
    """Knobs for the simulated latency/throughput behaviour."""

    step_s: float = 0.01  # simulated seconds per batch step
    prefill_tokens_per_step: int = 256  # prompt tokens absorbed per prefill step
    prefix_cache_speedup: float = 4.0  # cached-prefix prefill is this much faster
    speed_profile: float = 1.0  # 1.0 = baseline; >1 fast, <1 slow (hardware-aware)


class SimWorker:
    """A Worker (admit / step / in_flight) backed by a scheduler + timing model."""

    def __init__(
        self,
        worker_id: WorkerId,
        *,
        max_batch_size: int,
        profile: SimProfile,
        scheduler_cls: type[_BaseScheduler] = ContinuousScheduler,
    ) -> None:
        self.worker_id = worker_id
        self.profile = profile
        self._sched = scheduler_cls(max_batch_size)
        self._clock = 0.0
        self._cached_prefixes: set[str] = set()

    # ---- Worker protocol ----------------------------------------------------

    def admit(self, req: Request) -> SeqId:
        seq_id = SeqId(req.id)
        self._sched.enqueue(
            SeqWork(
                seq_id=seq_id,
                prefill_steps=self._prefill_steps(req),
                decode_steps=max(1, req.max_tokens),
            )
        )
        if req.prefix_key is not None:
            self._cached_prefixes.add(req.prefix_key)  # warm for subsequent requests
        return seq_id

    def step(self) -> list[TokenEvent]:
        self._clock += self.profile.step_s
        res = self._sched.step()
        finished = set(res.finished)
        return [
            TokenEvent(seq_id=sid, is_final=sid in finished, ts=self._clock) for sid in res.tokens
        ]

    def in_flight(self) -> int:
        return self._sched.in_flight()

    # ---- extras (beyond the protocol) --------------------------------------

    def is_idle(self) -> bool:
        return self._sched.is_idle()

    @property
    def clock(self) -> float:
        return self._clock

    def state(self) -> WorkerState:
        """A point-in-time snapshot for routing strategies (phase 2)."""
        return WorkerState(
            worker_id=self.worker_id,
            queue_depth=self._sched.queue_depth(),
            pending_tokens=self._sched.pending_decode_tokens(),
            in_flight=self._sched.in_flight(),
            tok_per_s=self.profile.speed_profile / self.profile.step_s,
            healthy=True,
            speed_profile=self.profile.speed_profile,
            cached_prefixes=frozenset(self._cached_prefixes),
        )

    # ---- internals ----------------------------------------------------------

    def _prefill_steps(self, req: Request) -> int:
        base = math.ceil(req.prompt_tokens / self.profile.prefill_tokens_per_step)
        if req.prefix_key is not None and req.prefix_key in self._cached_prefixes:
            base = math.ceil(base / self.profile.prefix_cache_speedup)
        return base
