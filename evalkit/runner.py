"""Trajectory capture: run questions against an ADK agent, collect the
ordered tool calls (name, args, result) each run made.

ask_adk() returns one row per query:
    {input, output, tools_called, trajectory, trajectory_text,
     n_tool_calls, latency_s}

run_batch() adds 429/503 retry + pacing-friendly sequential execution.
The caller builds the Runner (agent, instruction version, tracer output
dir) — this module only knows how to drive it and read the spans.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from evalkit.tracer import Tracer, walk_spans


def format_trajectory(traj: list) -> str:
    """Render tool calls as <call>/<arguments>/<result> XML-ish text for
    judge prompt templates ({trajectory_text} placeholder)."""
    if not traj:
        return "  (no tools were called)"
    lines = []
    for i, call in enumerate(traj, 1):
        lines.append(f'  <call step="{i}" tool="{call["tool"]}">')
        lines.append(f'    <arguments>{json.dumps(call["args"], default=str)[:2000]}</arguments>')
        lines.append(f'    <result>{json.dumps(call["result"], default=str)[:2000]}</result>')
        lines.append("  </call>")
    return "\n".join(lines)


def _is_transient(err: str) -> bool:
    return "RESOURCE_EXHAUSTED" in err or "UNAVAILABLE" in err


async def ask_adk(
    runner: Runner,
    session_service: InMemorySessionService,
    question: str,
    tracer: Tracer,
    *,
    app_name: str = "app",
    user_id: str = "eval",
    max_retries: int = 3,
) -> dict[str, Any]:
    """One question = one chain span; tool/llm spans nest via callbacks.

    Retries transient 429/503s. Returns a trajectory row dict."""
    for attempt in range(max_retries):
        try:
            return await _ask_once(runner, session_service, question, tracer,
                                   app_name=app_name, user_id=user_id)
        except Exception as exc:
            err = str(exc)
            if not _is_transient(err) or attempt == max_retries - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"rate limited/overloaded — retrying in {wait}s ({question[:50]})")
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def _ask_once(runner, session_service, question, tracer, *,
                    app_name, user_id) -> dict[str, Any]:
    session = await session_service.create_session(app_name=app_name, user_id=user_id)
    t0 = time.monotonic()
    with tracer.span("run", kind="chain", input={"user_query": question}) as span:
        final_text = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=question)]),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(
                    p.text for p in event.content.parts if getattr(p, "text", None)
                )
        span.output = {"final_answer": final_text}
        trajectory = [
            {"tool": s.name, "args": s.input, "result": s.output}
            for s in walk_spans(span)
            if s.kind == "tool"
        ]
    return {
        "input": question,
        "output": final_text,
        "tools_called": [c["tool"] for c in trajectory],
        "trajectory": trajectory,
        "trajectory_text": format_trajectory(trajectory),
        "n_tool_calls": len(trajectory),
        "latency_s": round(time.monotonic() - t0, 2),
    }


async def run_batch(
    questions: list[str],
    make_runner: Callable[[], tuple[Runner, InMemorySessionService, Tracer]],
    *,
    app_name: str = "app",
    user_id: str = "eval",
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Run every question sequentially through one runner.

    make_runner() -> (runner, session_service, tracer) — build once per
    batch (e.g. per prompt version) so pacing state is shared.
    progress=True prints one line per question — useful since each query
    can take tens of seconds under LLM pacing."""
    runner, session_service, tracer = make_runner()
    rows = []
    for i, question in enumerate(questions, 1):
        if progress:
            print(f"[{i}/{len(questions)}] {question[:70]}", flush=True)
        row = await ask_adk(runner, session_service, question, tracer,
                            app_name=app_name, user_id=user_id)
        rows.append(row)
    return rows
