"""OpenAIWorker against a mocked OpenAI-compatible endpoint (httpx.MockTransport),
so it runs in CI with no real model. Verifies streaming -> TokenEvents, in-flight
accounting, and that an endpoint error degrades health but still completes the seq.
"""

from __future__ import annotations

import json
import time

import httpx

from inference_demo.types import Priority, Request, WorkerId
from inference_demo.workers.base import ControlWorker
from inference_demo.workers.openai_worker import OpenAIWorker, _parse_sse_delta


def _sse(tokens: list[str]) -> bytes:
    lines = []
    for tok in tokens:
        chunk = {"choices": [{"delta": {"content": tok}}]}
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _req(rid: str, out: int = 4) -> Request:
    return Request(
        id=rid,
        prompt_tokens=10,
        max_tokens=out,
        priority=Priority.INTERACTIVE,
        arrival_ts=0.0,
        prefix_key=None,
        prompt_text="hi",
    )


def _drain_until_final(worker: OpenAIWorker, timeout_s: float = 5.0) -> list:
    events = []
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        events.extend(worker.step())
        if any(e.is_final for e in events):
            return events
        time.sleep(0.01)
    raise AssertionError("no final event before timeout")


def test_parse_sse_delta() -> None:
    assert _parse_sse_delta('data: {"choices":[{"delta":{"content":"hi"}}]}') == "hi"
    assert _parse_sse_delta("data: [DONE]") is None
    assert _parse_sse_delta(": comment") is None
    assert _parse_sse_delta("") is None


def test_streams_tokens_into_events() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(["He", "llo", "!"]))

    client = httpx.Client(base_url="http://mock", transport=httpx.MockTransport(handler))
    worker = OpenAIWorker(WorkerId("oai0"), client=client)
    assert isinstance(worker, ControlWorker)

    worker.admit(_req("r1"))
    events = _drain_until_final(worker)

    content_tokens = [e for e in events if not e.is_final]
    assert len(content_tokens) == 3  # three streamed deltas
    assert events[-1].is_final
    assert worker.in_flight() == 0
    assert worker.is_idle()


def test_unhealthy_on_endpoint_error_but_still_completes() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    client = httpx.Client(base_url="http://mock", transport=httpx.MockTransport(handler))
    worker = OpenAIWorker(WorkerId("oai0"), client=client)

    worker.admit(_req("r1"))
    events = _drain_until_final(worker)

    assert events[-1].is_final  # the sequence still terminates
    assert worker.in_flight() == 0
    assert worker.state().healthy is False  # error degraded health
