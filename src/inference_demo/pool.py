"""PoolManager — the runtime that ties the pure pieces into a live system.

It owns the worker pool, the router, the autoscaler, and metrics, and advances
everything one global sim step at a time. The pool's own clock is authoritative
for metrics (worker-local clocks would diverge when the autoscaler adds workers
mid-run), so token timings are stamped with the pool clock, not TokenEvent.ts.

Workers are created via an injected factory, so the pool is backend-agnostic: the
same control plane runs over SimWorker, OpenAIWorker, or RealModelWorker. All
policy lives in the pure modules; this class is the wiring and the I/O-free event
loop body. The gateway drives it (background loop in prod, explicit steps in tests).
"""

from __future__ import annotations

from collections.abc import Callable

from inference_demo.autoscaler import Autoscaler, AutoscalerConfig, PoolSnapshot, ScaleAction
from inference_demo.metrics import Metrics
from inference_demo.routing.router import Router
from inference_demo.routing.strategies import make_strategy
from inference_demo.sim.worker import SimProfile, SimWorker
from inference_demo.types import Request, WorkerId, WorkerState
from inference_demo.workers.base import ControlWorker

WorkerFactory = Callable[[WorkerId], ControlWorker]


class PoolManager:
    def __init__(
        self,
        *,
        worker_factory: WorkerFactory,
        n_workers: int,
        step_s: float,
        router: Router,
        autoscaler: Autoscaler,
        metrics: Metrics,
        autoscale_enabled: bool = True,
        autoscale_every_steps: int = 10,
    ) -> None:
        self._worker_factory = worker_factory
        self.step_s = step_s
        self.router = router
        self.autoscaler = autoscaler
        self.metrics = metrics
        self.autoscale_enabled = autoscale_enabled
        self._autoscale_every = autoscale_every_steps

        self._clock = 0.0
        self._steps = 0
        self._next_id = 0
        self._last_scale_clock = 0.0
        self._workers: dict[WorkerId, ControlWorker] = {}
        for _ in range(n_workers):
            self._add_worker()

    # ---- pool composition ---------------------------------------------------

    def _add_worker(self) -> WorkerId:
        wid = WorkerId(f"w{self._next_id}")
        self._next_id += 1
        self._workers[wid] = self._worker_factory(wid)
        return wid

    def _remove_idle_worker(self) -> WorkerId | None:
        for wid, w in self._workers.items():
            if w.is_idle():
                del self._workers[wid]
                return wid
        return None

    def kill_worker(self, worker_id: WorkerId | None = None) -> WorkerId | None:
        """Scenario button: drop a worker (busiest by default) to show recovery."""
        if not self._workers:
            return None
        if worker_id is None:
            worker_id = max(self._workers, key=lambda wid: self._workers[wid].in_flight())
        return worker_id if self._workers.pop(worker_id, None) is not None else None

    # ---- properties / reads -------------------------------------------------

    @property
    def clock(self) -> float:
        return self._clock

    @property
    def num_workers(self) -> int:
        return len(self._workers)

    def worker_states(self) -> list[WorkerState]:
        return [w.state() for w in self._workers.values()]

    # ---- controls -----------------------------------------------------------

    def set_strategy(self, name: str) -> None:
        self.router.set_strategy(make_strategy(name))

    def set_autoscaler(
        self, *, config: AutoscalerConfig | None = None, enabled: bool | None = None
    ) -> None:
        if config is not None:
            self.autoscaler = Autoscaler(config)
        if enabled is not None:
            self.autoscale_enabled = enabled

    # ---- the loop body ------------------------------------------------------

    def submit(self, req: Request) -> WorkerId:
        wid = self.router.route(self.worker_states(), req)
        self._workers[wid].admit(req)
        self.metrics.on_submit(req.id, arrival_ts=self._clock)
        self.metrics.on_route(req.id, wid, self.router.strategy_name)
        return wid

    def step(self) -> None:
        self._clock += self.step_s
        self._steps += 1
        for w in self._workers.values():
            for ev in w.step():
                self.metrics.on_token(str(ev.seq_id), ts=self._clock, is_final=ev.is_final)
        self.metrics.set_in_flight(sum(w.in_flight() for w in self._workers.values()))
        if self.autoscale_enabled and self._steps % self._autoscale_every == 0:
            self._maybe_scale()

    def _maybe_scale(self) -> None:
        states = self.worker_states()
        avg_load = sum(s.queue_depth + s.in_flight for s in states) / len(states) if states else 0.0
        snapshot = PoolSnapshot(
            num_workers=self.num_workers,
            avg_queue_depth=avg_load,
            seconds_since_last_scale=self._clock - self._last_scale_clock,
        )
        action = self.autoscaler.decide(snapshot)
        if action is ScaleAction.UP:
            self._add_worker()
            self._last_scale_clock = self._clock
        elif action is ScaleAction.DOWN and self._remove_idle_worker() is not None:
            self._last_scale_clock = self._clock

    # ---- snapshots for the API / SSE ---------------------------------------

    def pool_snapshot(self) -> dict[str, object]:
        c = self.autoscaler.config
        return {
            "num_workers": self.num_workers,
            "strategy": self.router.strategy_name,
            "clock_s": round(self._clock, 2),
            "autoscaler": {
                "enabled": self.autoscale_enabled,
                "min_workers": c.min_workers,
                "max_workers": c.max_workers,
                "target_queue_depth": c.target_queue_depth,
            },
            "workers": [
                {
                    "worker_id": str(s.worker_id),
                    "queue_depth": s.queue_depth,
                    "in_flight": s.in_flight,
                    "tok_per_s": round(s.tok_per_s, 1),
                    "cached_prefixes": len(s.cached_prefixes),
                    "healthy": s.healthy,
                }
                for s in self.worker_states()
            ],
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "metrics": self.metrics.snapshot(now=self._clock).to_dict(),
            "pool": self.pool_snapshot(),
            "recent": self.metrics.recent_requests(15),
        }


def _sim_factory(max_batch_size: int, profile: SimProfile) -> WorkerFactory:
    def make(wid: WorkerId) -> ControlWorker:
        return SimWorker(wid, max_batch_size=max_batch_size, profile=profile)

    return make


def _openai_factory(base_url: str, model: str) -> WorkerFactory:
    from inference_demo.workers.openai_worker import OpenAIWorker

    def make(wid: WorkerId) -> ControlWorker:
        return OpenAIWorker(wid, base_url=base_url, model=model)

    return make


def _realmodel_factory(model: str, max_batch_size: int) -> WorkerFactory:
    from inference_demo.workers.real_model_worker import DEFAULT_MODEL, RealModelWorker

    name = model if model != "qwen2.5:0.5b" else DEFAULT_MODEL  # default OpenAI tag -> HF id

    def make(wid: WorkerId) -> ControlWorker:
        return RealModelWorker(wid, model_name=name, max_batch_size=max_batch_size)

    return make


def build_pool(
    *,
    backend: str = "sim",
    n_workers: int = 2,
    max_batch_size: int = 8,
    strategy: str = "least-pending-tokens",
    autoscale: bool = True,
    min_workers: int = 1,
    max_workers: int = 8,
    seed: int | None = None,
    base_url: str = "http://localhost:11434",
    model: str = "qwen2.5:0.5b",
) -> PoolManager:
    """Construct a ready-to-run pool with sensible defaults (used by the gateway)."""
    if backend == "sim":
        profile = SimProfile(step_s=0.01, prefill_tokens_per_step=128)
        factory: WorkerFactory = _sim_factory(max_batch_size, profile)
        step_s = profile.step_s
    elif backend == "openai":
        factory = _openai_factory(base_url, model)
        step_s = 0.05  # pool tick granularity for metrics (external server decodes)
    elif backend == "realmodel":
        factory = _realmodel_factory(model, max_batch_size)
        step_s = 0.05  # one decode iteration per pool step
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    router = Router(make_strategy(strategy, seed=seed))
    autoscaler = Autoscaler(
        AutoscalerConfig(
            min_workers=min_workers,
            max_workers=max_workers,
            target_queue_depth=4.0,
            scale_down_queue_depth=1.0,
            cooldown_s=0.5,
        )
    )
    return PoolManager(
        worker_factory=factory,
        n_workers=n_workers,
        step_s=step_s,
        router=router,
        autoscaler=autoscaler,
        metrics=Metrics(),
        autoscale_enabled=autoscale,
    )
