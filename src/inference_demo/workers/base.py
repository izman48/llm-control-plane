"""The architecture seam: the ``Worker`` protocol.

Everything above the worker (router, scheduler, autoscaler, metrics, UI) is
backend-agnostic and depends only on this interface. Three implementations exist
or are planned: SimWorker (phase 1), OpenAIWorker + RealModelWorker (phase 6).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from inference_demo.types import Request, SeqId, TokenEvent, WorkerState


@runtime_checkable
class Worker(Protocol):
    def admit(self, req: Request) -> SeqId:
        """Accept a request for execution; return its sequence id."""
        ...

    def step(self) -> list[TokenEvent]:
        """Advance the running batch one decode step; return tokens emitted."""
        ...

    def in_flight(self) -> int:
        """Number of sequences currently running (not counting those queued)."""
        ...


@runtime_checkable
class ControlWorker(Worker, Protocol):
    """A Worker the PoolManager can also observe and reclaim.

    Adds the two reads the control plane needs beyond the bare execution
    interface; every backend (SimWorker, OpenAIWorker, RealModelWorker) provides
    these so the pool, router, and metrics stay backend-agnostic.
    """

    def state(self) -> WorkerState:
        """A point-in-time snapshot for routing strategies + the dashboard."""
        ...

    def is_idle(self) -> bool:
        """True when nothing is queued or running (safe to reclaim)."""
        ...
