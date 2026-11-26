"""ADK instrumentation helpers: LLM pacing + traced callbacks.

Any ADK agent can be instrumented by passing a Tracer 
(from evalkit.tracer) — no agent-code changes required.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import ToolContext
from google.adk.tools.base_tool import BaseTool

from evalkit.tracer import Span, Tracer


def _usage(resp: LlmResponse) -> dict[str, int]:
    usage = getattr(resp, "usage_metadata", None) or getattr(resp, "usageMetadata", None) or {}
    get = usage.get if isinstance(usage, dict) else lambda k, d=None: getattr(usage, k, d)
    return {
        "input_tokens": get("prompt_token_count", get("input_token_count")),
        "output_tokens": get("candidates_token_count", get("output_token_count")),
    }


def _llm_input(req: LlmRequest) -> dict[str, Any]:
    contents = getattr(req, "contents", None) or []
    texts = [
        part.text
        for content in contents
        for part in (getattr(content, "parts", None) or [])
        if getattr(part, "text", None)
    ]
    return {"model": getattr(req, "model", None), "turns": len(contents),
            "last_message": (texts[-1][:500] if texts else None)}


def _llm_output(resp: LlmResponse) -> dict[str, Any]:
    parts = getattr(getattr(resp, "content", None), "parts", None) or []
    text = "".join(p.text for p in parts if getattr(p, "text", None))
    calls = [
        p.function_call.name for p in parts if getattr(p, "function_call", None)
    ]
    out = {"text": text or None, "tool_calls": calls or None, "usage": _usage(resp)}
    if getattr(resp, "error_code", None) or getattr(resp, "error_message", None):
        out["error"] = getattr(resp, "error_message", None) or resp.error_code
    return {k: v for k, v in out.items() if v is not None}


def llm_pacer(min_interval_s: float | None = None):
    """Async before_model callback: space model calls under a free-tier RPM cap."""
    min_interval = float(
        min_interval_s if min_interval_s is not None
        else os.getenv("LLM_MIN_INTERVAL_S", "4.5")
    )
    last = 0.0

    async def before_model(callback_context, llm_request):
        nonlocal last
        wait = min_interval - (time.monotonic() - last)
        if wait > 0:
            await asyncio.sleep(wait)
        last = time.monotonic()
        return None

    return before_model


def traced_callbacks(tracer: Tracer) -> dict[str, Any]:
    """ADK callbacks that open/close llm and tool spans around each call."""
    pending: dict[str, list[tuple[Any, Span]]] = {"llm": [], "tool": []}

    def enter(kind: str, name: str, input_: Any):
        cm = tracer.span(name, kind=kind, input=input_)
        span = cm.__enter__()
        pending[kind].append((cm, span))

    def close(kind: str, output: Any = None, error: BaseException | None = None) -> None:
        # Drain stale entries (e.g. a failed call ADK never closed).
        while len(pending[kind]) > 1:
            cm, stale = pending[kind].pop(0)
            stale.status, stale.error = "error", {"type": "UnclosedSpan", "message": kind}
            cm.__exit__(None, None, None)
        if not pending[kind]:
            return
        cm, span = pending[kind].pop()
        if error is not None:
            span.status = "error"
            span.error = {"type": type(error).__name__, "message": str(error)}
        elif output is not None:
            span.output = output
        cm.__exit__(None, None, None)

    paced = llm_pacer()

    async def before_model(callback_context: CallbackContext, llm_request: LlmRequest):
        await paced(callback_context, llm_request)
        enter("llm", "generate_content", _llm_input(llm_request))
        return None

    def after_model(callback_context: CallbackContext, llm_response: LlmResponse):
        close("llm", output=_llm_output(llm_response))
        return None

    def before_tool(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext):
        enter("tool", tool.name, args)
        return None

    def after_tool(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
                   tool_response: dict[str, Any]):
        close("tool", output=tool_response)
        return None

    return {
        "before_model_callback": before_model,
        "after_model_callback": after_model,
        "before_tool_callback": before_tool,
        "after_tool_callback": after_tool,
    }
