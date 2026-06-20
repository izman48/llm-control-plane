"""Shared contracts — the single source of truth.

Every component consumes/produces these shapes. Do not invent parallel shapes
elsewhere; extend here. See PLAN.md "Shared contracts".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType

SeqId = NewType("SeqId", str)
WorkerId = NewType("WorkerId", str)


class Priority(Enum):
    INTERACTIVE = "interactive"
    BATCH = "batch"


@dataclass(frozen=True)
class Request:
    """A unit of work submitted to the control plane.

    Sim workers use the token *counts*; real workers also carry ``prompt_text``.
    """

    id: str
    prompt_tokens: int
    max_tokens: int
    priority: Priority
    arrival_ts: float
    prefix_key: str | None  # for prefix-affinity routing / KV-cache reuse
    prompt_text: str | None = None


@dataclass(frozen=True)
class TokenEvent:
    """One decode token emitted by a worker for a sequence."""

    seq_id: SeqId
    is_final: bool
    ts: float


@dataclass(frozen=True)
class WorkerState:
    """A point-in-time snapshot of a worker, consumed by routing strategies."""

    worker_id: WorkerId
    queue_depth: int
    pending_tokens: int  # estimated remaining work
    in_flight: int
    tok_per_s: float
    healthy: bool
    speed_profile: float  # 1.0 = baseline; <1 slow, >1 fast (hardware-aware)
    cached_prefixes: frozenset[str]
