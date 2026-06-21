"""RealModelWorker — a real small model with OUR continuous batching.

This is the backend that actually showcases the project's batching claim: we own
the decode loop (no ``model.generate``), keep a KV cache across steps, batch
multiple sequences through one forward pass, and admit/evict sequences every step
(continuous, non-paged).

Host-native only (MPS isn't available in Docker on macOS). Heavy deps (torch,
transformers) are imported lazily so the rest of the package — and CI — never
needs them. Default model: Qwen2.5-0.5B-Instruct.

Batching strategy (honest framing): we keep one batched KV cache for the running
set. When the set changes (a sequence finishes or a new one is admitted) we
**rebuild** the cache with a batched prefill over the current sequences; between
membership changes we decode incrementally against the cache. The rebuild-on-
membership-change recompute is the inefficiency that PagedAttention removes — we
do NOT implement paged attention. Decoding is greedy (argmax) for determinism.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from inference_demo.types import Request, SeqId, TokenEvent, WorkerId, WorkerState

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class _Seq:
    seq_id: SeqId
    token_ids: list[int]  # full real sequence so far (prompt + generated)
    max_new: int
    generated: int = 0
    finished: bool = False


class RealModelWorker:
    def __init__(
        self,
        worker_id: WorkerId,
        *,
        model_name: str = DEFAULT_MODEL,
        max_batch_size: int = 8,
        device: str | None = None,
        dtype: Any = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.worker_id = worker_id
        self.max_batch_size = max_batch_size
        self._torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._device = device

        # Handles are external/dynamically-typed; treat as Any (stubs vary by version).
        self._tok: Any = AutoTokenizer.from_pretrained(model_name)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        kwargs: dict[str, Any] = {} if dtype is None else {"dtype": dtype}
        model: Any = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self._model: Any = model.to(device).eval()
        self._eos = self._tok.eos_token_id

        self._waiting: deque[_Seq] = deque()
        self._running: list[_Seq] = []
        self._generated: dict[SeqId, list[int]] = {}
        self._cache: Any = None
        self._attn: Any = None
        self._dirty = True

    # ---- Worker protocol ----------------------------------------------------

    def admit(self, req: Request) -> SeqId:
        seq_id = SeqId(req.id)
        messages = [{"role": "user", "content": req.prompt_text or "Hello"}]
        enc = self._tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)
        ids = [int(t) for t in enc["input_ids"]]
        self._waiting.append(_Seq(seq_id=seq_id, token_ids=ids, max_new=max(1, req.max_tokens)))
        return seq_id

    def step(self) -> list[TokenEvent]:
        torch = self._torch
        before = len(self._running)
        self._running = [s for s in self._running if not s.finished]  # evict finished
        while len(self._running) < self.max_batch_size and self._waiting:  # admit into free slots
            self._running.append(self._waiting.popleft())
        if len(self._running) != before or self._cache is None:
            self._dirty = True
        if not self._running:
            return []

        with torch.no_grad():
            logits = self._forward()  # [B, vocab] next-token logits per running seq
        next_ids = torch.argmax(logits, dim=-1).tolist()

        events: list[TokenEvent] = []
        for s, tid in zip(self._running, next_ids, strict=True):
            s.token_ids.append(int(tid))
            s.generated += 1
            self._generated.setdefault(s.seq_id, []).append(int(tid))
            is_final = tid == self._eos or s.generated >= s.max_new
            s.finished = is_final
            events.append(TokenEvent(seq_id=s.seq_id, is_final=is_final, ts=0.0))
        return events

    def in_flight(self) -> int:
        return len(self._running)

    # ---- ControlWorker extras ----------------------------------------------

    def is_idle(self) -> bool:
        return not self._waiting and not self._running

    def generated_ids(self, seq_id: SeqId) -> list[int]:
        """Token ids generated for a sequence (for verification / detokenizing)."""
        return self._generated.get(seq_id, [])

    def state(self) -> WorkerState:
        pending = sum(max(0, s.max_new - s.generated) for s in self._running)
        pending += sum(s.max_new for s in self._waiting)
        return WorkerState(
            worker_id=self.worker_id,
            queue_depth=len(self._waiting),
            pending_tokens=pending,
            in_flight=len(self._running),
            tok_per_s=0.0,  # measured by Metrics, not self-reported here
            healthy=True,
            speed_profile=1.0,
            cached_prefixes=frozenset(),
        )

    # ---- the decode loop (ours) --------------------------------------------

    def _forward(self) -> Any:
        torch = self._torch
        if self._dirty:
            input_ids, attn = self._left_pad([s.token_ids for s in self._running])
            pos = self._position_ids(attn)
            out = self._model(
                input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=True
            )
            self._cache = out.past_key_values
            self._attn = attn
            self._dirty = False
            return out.logits[:, -1, :]

        # incremental: one new token per running seq against the running cache
        last = torch.tensor([[s.token_ids[-1]] for s in self._running], device=self._device)
        pos = self._attn.sum(dim=-1, keepdim=True)  # next position per row (0-indexed)
        self._attn = torch.cat(
            [
                self._attn,
                torch.ones((len(self._running), 1), dtype=self._attn.dtype, device=self._device),
            ],
            dim=1,
        )
        out = self._model(
            input_ids=last,
            attention_mask=self._attn,
            position_ids=pos,
            past_key_values=self._cache,
            use_cache=True,
        )
        self._cache = out.past_key_values
        return out.logits[:, -1, :]

    def _left_pad(self, batch: list[list[int]]) -> tuple[Any, Any]:
        torch = self._torch
        maxlen = max(len(b) for b in batch)
        pad = self._tok.pad_token_id
        ids = [[pad] * (maxlen - len(b)) + b for b in batch]
        mask = [[0] * (maxlen - len(b)) + [1] * len(b) for b in batch]
        return (
            torch.tensor(ids, device=self._device),
            torch.tensor(mask, device=self._device),
        )

    def _position_ids(self, attn: Any) -> Any:
        # left-padding-aware: real tokens get 0,1,2,...; pad positions clamped to 0
        pos = attn.long().cumsum(dim=-1) - 1
        pos.masked_fill_(attn == 0, 0)
        return pos
