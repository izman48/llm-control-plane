"""Autoscaler — scale decisions as pure policy, testable in isolation.

``decide(snapshot)`` maps a point-in-time view of the pool to a ``ScaleAction``
(UP / DOWN / HOLD). It is pure: same snapshot, same action, no mutation. The
caller (the PoolManager, phase 4) owns the clock and the pool; it applies the
action (±1 worker) and tracks the time since the last scale.

Two mechanisms keep it from flapping:

* **Hysteresis band** — scale up above ``target_queue_depth``, down only below
  ``scale_down_queue_depth``. In between it holds, so it doesn't oscillate around
  a single threshold.
* **Cooldown** — after any scale action, hold for ``cooldown_s`` before acting
  again (in either direction). The caller reports elapsed time in the snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScaleAction(Enum):
    UP = "up"
    DOWN = "down"
    HOLD = "hold"


@dataclass(frozen=True)
class AutoscalerConfig:
    min_workers: int
    max_workers: int
    target_queue_depth: float  # scale up when avg queue depth exceeds this
    scale_down_queue_depth: float  # scale down when avg queue depth falls below this
    cooldown_s: float  # min seconds between scale actions (either direction)

    def __post_init__(self) -> None:
        if self.min_workers < 1:
            raise ValueError("min_workers must be >= 1")
        if self.min_workers > self.max_workers:
            raise ValueError("min_workers must be <= max_workers")
        if self.scale_down_queue_depth > self.target_queue_depth:
            raise ValueError("scale_down_queue_depth must be <= target_queue_depth")
        if self.cooldown_s < 0:
            raise ValueError("cooldown_s must be >= 0")


@dataclass(frozen=True)
class PoolSnapshot:
    num_workers: int
    avg_queue_depth: float
    seconds_since_last_scale: float


class Autoscaler:
    def __init__(self, config: AutoscalerConfig) -> None:
        self.config = config

    def decide(self, snapshot: PoolSnapshot) -> ScaleAction:
        c = self.config
        # Hard bounds first, ignoring cooldown: if the pool is out of [min, max]
        # (e.g. a scenario killed every worker), recover toward the range at once.
        if snapshot.num_workers < c.min_workers:
            return ScaleAction.UP
        if snapshot.num_workers > c.max_workers:
            return ScaleAction.DOWN

        if snapshot.seconds_since_last_scale < c.cooldown_s:
            return ScaleAction.HOLD  # cooling down after a recent action

        if snapshot.avg_queue_depth > c.target_queue_depth and snapshot.num_workers < c.max_workers:
            return ScaleAction.UP
        if (
            snapshot.avg_queue_depth < c.scale_down_queue_depth
            and snapshot.num_workers > c.min_workers
        ):
            return ScaleAction.DOWN
        return ScaleAction.HOLD
