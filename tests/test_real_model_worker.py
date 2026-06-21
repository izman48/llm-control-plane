"""On-device tests for RealModelWorker — our own continuous batched decode loop.

Gated behind the ``realmodel`` marker and skipped unless torch is installed, so
CI (no torch) never runs them. Run manually after `uv sync --extra realmodel`:

    uv run pytest -m realmodel

First run downloads the model (~1 GB). These verify that (a) our greedy decode
matches HuggingFace's reference generate() for a single sequence, and (b) batched
continuous decode keeps per-sequence outputs identical to decoding each alone
(no cross-attention leakage).
"""

from __future__ import annotations

import pytest

from inference_demo.types import Priority, Request, SeqId, WorkerId

pytestmark = pytest.mark.realmodel

torch = pytest.importorskip("torch")

from inference_demo.workers.real_model_worker import RealModelWorker  # noqa: E402


def _req(rid: str, text: str, out: int) -> Request:
    return Request(
        id=rid,
        prompt_tokens=0,
        max_tokens=out,
        priority=Priority.INTERACTIVE,
        arrival_ts=0.0,
        prefix_key=None,
        prompt_text=text,
    )


def _run(worker: RealModelWorker, max_steps: int = 200) -> None:
    for _ in range(max_steps):
        if worker.is_idle():
            return
        worker.step()
    raise AssertionError("worker did not drain")


def test_greedy_matches_reference_single_sequence() -> None:
    worker = RealModelWorker(WorkerId("rm0"), max_batch_size=4)
    prompt, n = "Reply with exactly: hello", 10
    worker.admit(_req("r1", prompt, out=n))
    _run(worker)
    ours = worker.generated_ids(SeqId("r1"))

    # reference: HF greedy generate over the same chat-templated prompt
    tok, model = worker._tok, worker._model
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(worker._device)
    gen = model.generate(input_ids, max_new_tokens=n, do_sample=False)
    ref = gen[0, input_ids.shape[1] :].tolist()

    assert ours == ref


def test_batched_decode_isolates_sequences() -> None:
    # Two different prompts decoded together must match each decoded alone.
    prompts = [("a", "Count: one two"), ("b", "Name a color")]
    n = 8

    alone: dict[str, list[int]] = {}
    for rid, text in prompts:
        w = RealModelWorker(WorkerId("solo"), max_batch_size=1)
        w.admit(_req(rid, text, out=n))
        _run(w)
        alone[rid] = w.generated_ids(SeqId(rid))

    batched = RealModelWorker(WorkerId("batch"), max_batch_size=4)
    for rid, text in prompts:
        batched.admit(_req(rid, text, out=n))
    _run(batched)

    for rid, _ in prompts:
        assert batched.generated_ids(SeqId(rid)) == alone[rid], f"seq {rid} differs when batched"
