"""The autoscaler is pure policy: a snapshot in, a ScaleAction out. We test it in
isolation — no live pool — covering scale up/down, min/max bounds, the hysteresis
band, and cooldown so it cannot flap. Wiring into a real pool is phase 4.
"""

from __future__ import annotations

import pytest

from inference_demo.autoscaler import (
    Autoscaler,
    AutoscalerConfig,
    PoolSnapshot,
    ScaleAction,
)


def cfg(**kw: float | int) -> AutoscalerConfig:
    base: dict[str, float | int] = dict(
        min_workers=1,
        max_workers=8,
        target_queue_depth=4.0,
        scale_down_queue_depth=1.0,
        cooldown_s=15.0,
    )
    base.update(kw)
    return AutoscalerConfig(**base)  # type: ignore[arg-type]


def snap(*, n: int, avg: float, since: float = 999.0) -> PoolSnapshot:
    return PoolSnapshot(num_workers=n, avg_queue_depth=avg, seconds_since_last_scale=since)


# ---- scale up --------------------------------------------------------------


def test_scales_up_when_over_target_and_below_max() -> None:
    auto = Autoscaler(cfg())
    assert auto.decide(snap(n=2, avg=10.0)) == ScaleAction.UP


def test_does_not_scale_up_at_max() -> None:
    auto = Autoscaler(cfg(max_workers=4))
    assert auto.decide(snap(n=4, avg=10.0)) == ScaleAction.HOLD


# ---- scale down ------------------------------------------------------------


def test_scales_down_when_idle_and_above_min() -> None:
    auto = Autoscaler(cfg())
    assert auto.decide(snap(n=5, avg=0.0)) == ScaleAction.DOWN


def test_does_not_scale_down_at_min() -> None:
    auto = Autoscaler(cfg(min_workers=2))
    assert auto.decide(snap(n=2, avg=0.0)) == ScaleAction.HOLD


# ---- hysteresis: hold inside the band --------------------------------------


def test_holds_inside_hysteresis_band() -> None:
    # between scale_down (1.0) and target (4.0): no action either way
    auto = Autoscaler(cfg())
    assert auto.decide(snap(n=4, avg=2.5)) == ScaleAction.HOLD
    assert auto.decide(snap(n=4, avg=4.0)) == ScaleAction.HOLD  # exactly target -> hold
    assert auto.decide(snap(n=4, avg=1.0)) == ScaleAction.HOLD  # exactly down line -> hold


# ---- cooldown: anti-flap ---------------------------------------------------


def test_cooldown_blocks_action_even_when_hot() -> None:
    auto = Autoscaler(cfg(cooldown_s=15.0))
    assert auto.decide(snap(n=2, avg=99.0, since=5.0)) == ScaleAction.HOLD  # too soon
    assert auto.decide(snap(n=2, avg=99.0, since=20.0)) == ScaleAction.UP  # cooldown elapsed


def test_cooldown_blocks_scale_down_too() -> None:
    auto = Autoscaler(cfg(cooldown_s=15.0))
    assert auto.decide(snap(n=5, avg=0.0, since=3.0)) == ScaleAction.HOLD


# ---- config validation -----------------------------------------------------


def test_rejects_min_above_max() -> None:
    with pytest.raises(ValueError):
        cfg(min_workers=5, max_workers=2)


def test_rejects_down_line_above_target() -> None:
    with pytest.raises(ValueError):
        cfg(target_queue_depth=2.0, scale_down_queue_depth=5.0)


def test_decide_is_pure_no_mutation_between_calls() -> None:
    # Same snapshot must yield the same action regardless of prior calls.
    auto = Autoscaler(cfg())
    hot = snap(n=2, avg=10.0)
    assert auto.decide(hot) == ScaleAction.UP
    assert auto.decide(hot) == ScaleAction.UP
