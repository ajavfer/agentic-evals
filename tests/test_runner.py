"""Unit tests for evalkit.runner token accounting, error-trace saving,
and run_batch repeats — using a fake ADK runner (no LLM calls)."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.adk.sessions import InMemorySessionService

from evalkit.runner import ask_adk, run_batch
from evalkit.tracer import Tracer


async def _events():
    yield SimpleNamespace(is_final_response=lambda: False, content=None)
    yield SimpleNamespace(
        is_final_response=lambda: True,
        content=SimpleNamespace(parts=[SimpleNamespace(text="done")]),
    )


def _fake_runner(tracer, *, llm_calls=((10, 4), (7, None)), error=None):
    async def run_async(**kwargs):
        for tin, tout in llm_calls:
            with tracer.span("generate_content", kind="llm") as s:
                s.output = {"usage": {"input_tokens": tin,
                                      "output_tokens": tout}}
        if error is not None:
            raise error
        async for ev in _events():
            yield ev
    return SimpleNamespace(run_async=run_async)


def test_ask_once_aggregates_tokens_and_saves_trace(tmp_path):
    tracer = Tracer(tmp_path)
    svc = InMemorySessionService()
    row = asyncio.run(ask_adk(_fake_runner(tracer), svc, "q?", tracer,
                              app_name="t", user_id="u"))
    assert row["output"] == "done"
    assert row["input_tokens"] == 17      # 10 + 7
    assert row["output_tokens"] == 4      # None ignored
    saved = list(tmp_path.glob("*.json"))
    assert len(saved) == 1                # trace persisted


def test_ask_once_saves_error_trace(tmp_path):
    tracer = Tracer(tmp_path)
    svc = InMemorySessionService()
    runner = _fake_runner(tracer, error=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        asyncio.run(ask_adk(runner, svc, "q?", tracer,
                            app_name="t", user_id="u"))
    saved = list(tmp_path.glob("*.json"))
    assert len(saved) == 1
    data = json.loads(saved[0].read_text(encoding="utf-8"))
    assert data["status"] == "error"


def test_run_batch_repeats(tmp_path):
    tracer = Tracer(tmp_path)
    svc = InMemorySessionService()
    runner = _fake_runner(tracer)
    rows = asyncio.run(run_batch(
        ["q1", "q2"], lambda: (runner, svc, tracer),
        app_name="t", user_id="u", repeats=2,
    ))
    assert len(rows) == 4
    assert all(r["input_tokens"] == 17 for r in rows)
    assert len(list(tmp_path.glob("*.json"))) == 4
