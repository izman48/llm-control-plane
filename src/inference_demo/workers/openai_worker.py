"""OpenAIWorker — a Worker backed by any OpenAI-compatible endpoint (Ollama,
vLLM, LM Studio).

The external server owns the decode loop here, so this backend exercises
routing / autoscaling / observability but NOT our continuous batching (see
CLAUDE.md honesty constraints). Each ``admit`` streams a chat completion on a
background thread, pushing TokenEvents into a queue that ``step`` drains — fitting
the synchronous Worker interface the PoolManager drives.

SECURITY: taking an arbitrary URL server-side is SSRF. This worker is for
self-hosted / local mode; the public demo stays sim-only (see CLAUDE.md).
"""

from __future__ import annotations

import json
import queue
import threading

import httpx

from inference_demo.types import Request, SeqId, TokenEvent, WorkerId, WorkerState


class OpenAIWorker:
    def __init__(
        self,
        worker_id: WorkerId,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:0.5b",
        client: httpx.Client | None = None,
        nominal_tok_per_s: float = 50.0,
    ) -> None:
        self.worker_id = worker_id
        self.model = model
        self._client = client or httpx.Client(base_url=base_url, timeout=60.0)
        self._nominal_tok_per_s = nominal_tok_per_s
        self._events: queue.Queue[TokenEvent] = queue.Queue()
        self._lock = threading.Lock()
        self._in_flight = 0
        self._pending_tokens = 0
        self._threads: list[threading.Thread] = []
        self._healthy = True

    # ---- Worker protocol ----------------------------------------------------

    def admit(self, req: Request) -> SeqId:
        seq_id = SeqId(req.id)
        with self._lock:
            self._in_flight += 1
            self._pending_tokens += max(1, req.max_tokens)
        t = threading.Thread(target=self._stream, args=(seq_id, req), daemon=True)
        t.start()
        self._threads.append(t)
        return seq_id

    def step(self) -> list[TokenEvent]:
        out: list[TokenEvent] = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                break
        return out

    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    # ---- ControlWorker extras ----------------------------------------------

    def is_idle(self) -> bool:
        return self.in_flight() == 0 and self._events.empty()

    def state(self) -> WorkerState:
        with self._lock:
            in_flight, pending = self._in_flight, self._pending_tokens
            healthy = self._healthy
        return WorkerState(
            worker_id=self.worker_id,
            queue_depth=0,  # the external server owns its queue; not visible to us
            pending_tokens=pending,
            in_flight=in_flight,
            tok_per_s=self._nominal_tok_per_s,
            healthy=healthy,
            speed_profile=1.0,
            cached_prefixes=frozenset(),
        )

    def close(self) -> None:
        self._client.close()

    # ---- internals ----------------------------------------------------------

    def _stream(self, seq_id: SeqId, req: Request) -> None:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": req.prompt_text or "Hello"}],
            "max_tokens": req.max_tokens,
            "stream": True,
        }
        try:
            with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    delta = _parse_sse_delta(line)
                    if delta:
                        self._emit(seq_id, is_final=False)
        except Exception:
            with self._lock:
                self._healthy = False
        finally:
            self._emit(seq_id, is_final=True)
            with self._lock:
                self._in_flight -= 1

    def _emit(self, seq_id: SeqId, *, is_final: bool) -> None:
        # ts is stamped by the PoolManager's clock; 0.0 here is a placeholder.
        self._events.put(TokenEvent(seq_id=seq_id, is_final=is_final, ts=0.0))
        if not is_final:
            with self._lock:
                self._pending_tokens = max(0, self._pending_tokens - 1)


def _parse_sse_delta(line: str) -> str | None:
    """Extract the content delta from one OpenAI streaming SSE line, if any."""
    if not line or not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return None
    try:
        chunk = json.loads(data)
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content")
        return content if isinstance(content, str) and content else None
    except (json.JSONDecodeError, KeyError, IndexError):
        return None
