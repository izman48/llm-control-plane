"""The architecture seam: the ``Worker`` protocol.

Everything above the worker (router, scheduler, autoscaler, metrics, UI) is
backend-agnostic and depends only on this interface. Three implementations exist
or are planned: SimWorker (phase 1), OpenAIWorker + RealModelWorker (phase 6).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from inference_demo.types import Request, SeqId, TokenEvent


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
