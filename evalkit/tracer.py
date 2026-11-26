"""Minimal local tracer: chain / llm / tool spans saved as sanitized JSON.

One class, pydantic models, context-manager spans only.
"""

from __future__ import annotations

import inspect
import json
import re
import time
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

REDACTED_KEYS = frozenset(
    {"api_key", "apikey", "authorization", "cookie", "email", "password", "secret", "token"}
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+"),
    re.compile(r"\b(?:sk|AIza)[a-zA-Z0-9_-]{16,}\b"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_value(
    value: Any,
    *,
    max_string_chars: int = 4_000,
    max_collection_items: int = 100,
    max_depth: int = 12,
) -> Any:
    """Convert a value to bounded JSON-safe data, redacting secrets and PII."""

    def sensitive(key: str) -> bool:
        k = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        return k in REDACTED_KEYS or any(k.endswith(f"_{s}") for s in REDACTED_KEYS)

    def convert(item: Any, depth: int, seen: set[int]) -> Any:
        if item is None or isinstance(item, (bool, int, float)):
            return item
        if isinstance(item, str):
            for pattern in _SECRET_PATTERNS:
                item = pattern.sub("[REDACTED]", item)
            return item if len(item) <= max_string_chars else item[: max_string_chars - 3] + "..."
        if depth >= max_depth:
            return "[MAX_DEPTH]"
        item_id = id(item)
        if item_id in seen:
            return "[CYCLE]"
        seen.add(item_id)
        try:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif isinstance(item, dict):
                pass
            elif isinstance(item, (list, tuple, set, frozenset)):
                values = list(item)
                out = [convert(v, depth + 1, seen) for v in values[:max_collection_items]]
                if len(values) > max_collection_items:
                    out.append(f"[{len(values) - max_collection_items} ITEMS OMITTED]")
                return out
            elif hasattr(item, "__dict__"):
                item = vars(item)
            else:
                return convert(repr(item), depth + 1, seen)
            items = list(item.items())[:max_collection_items]
            out = {
                str(k): "[REDACTED]" if sensitive(str(k)) else convert(v, depth + 1, seen)
                for k, v in items
            }
            if len(item) > max_collection_items:
                out["_omitted"] = len(item) - max_collection_items
            return out
        finally:
            seen.discard(item_id)

    return convert(value, 0, set())


class Span(BaseModel):
    """One recorded operation. `children` + `parent_span_id` give tree and flat views."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    kind: str = "span"  # chain | llm | tool | span
    input: Any = None
    output: Any = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: str = "unset"  # unset | ok | error
    error: dict[str, str] | None = None
    start_time: str = Field(default_factory=utc_now)
    end_time: str | None = None
    duration_ms: float | None = None
    children: list["Span"] = Field(default_factory=list)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    span_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    parent_span_id: str | None = None
    _started_at: float = PrivateAttr(default_factory=time.perf_counter)

    def to_dict(self, sanitizer: Callable[[Any], Any] = sanitize_value) -> dict[str, Any]:
        data = self.model_dump(exclude={"_started_at"})
        return {k: sanitizer(v) for k, v in data.items()}


Span.model_rebuild()


def walk_spans(span: Span | dict) -> Iterator[Span | dict]:
    yield span
    children = span.get("children", []) if isinstance(span, dict) else span.children
    for child in children:
        yield from walk_spans(child)


def format_trace_tree(span: Span | dict, *, max_chars: int = 240, depth: int = 0) -> str:
    """Render a span and its children as an indented plain-text tree."""
    node = span if isinstance(span, dict) else span.to_dict()
    marker = "x" if node.get("status") == "error" else "-"
    duration = node.get("duration_ms")
    timing = f" ({duration:.1f} ms)" if isinstance(duration, (int, float)) else ""
    lines = [f"{'  ' * depth}{marker} {node.get('name', '?')} [{node.get('kind', 'span')}]{timing}"]
    for key in ("input", "output", "error"):
        value = node.get(key)
        if value is not None:
            text = json.dumps(value, ensure_ascii=False, default=str)
            if len(text) > max_chars:
                text = text[: max_chars - 3] + "..."
            lines.append(f"{'  ' * (depth + 1)}{key}: {text}")
    for child in node.get("children", []) or []:
        lines.append(format_trace_tree(child, max_chars=max_chars, depth=depth + 1))
    return "\n".join(lines)


class Tracer:
    """Records spans into a tree and exports each finished trace as JSON."""

    def __init__(
        self,
        output_dir: str | Path = "traces",
        *,
        capture_traceback: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.capture_traceback = capture_traceback
        self.traces: list[Span] = []
        self.saved_paths: list[Path] = []
        self.run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        self._counter = 0
        self._current: ContextVar[Span | None] = ContextVar(
            f"current_span_{self.run_id[-8:]}", default=None
        )

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = "span",
        input: Any = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """Open a span; children attach to whatever span is current."""
        parent = self._current.get()
        span = Span(
            name=name,
            kind=kind,
            input=input,
            attributes=attributes or {},
            trace_id=parent.trace_id if parent else uuid4().hex,
            parent_span_id=parent.span_id if parent else None,
        )
        if parent is None:
            self.traces.append(span)
        else:
            parent.children.append(span)
        token = self._current.set(span)
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.error = {"type": type(exc).__name__, "message": str(exc)}
            if self.capture_traceback:
                span.error["traceback"] = traceback.format_exc()
            raise
        else:
            span.status = "ok"
        finally:
            span.end_time = utc_now()
            span.duration_ms = round((time.perf_counter() - span._started_at) * 1_000, 3)
            try:
                self._current.reset(token)
            except ValueError:
                # ADK runs nodes in copied contexts; the token may not exist here.
                pass

    def decorator(self, *, kind: str, name: str | None = None):
        """Wrap a function so each call becomes a span (sync and async)."""

        def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
            span_name = name or function.__name__

            def call_input(args: tuple, kwargs: dict) -> dict:
                try:
                    bound = inspect.signature(function).bind(*args, **kwargs)
                    return {
                        k: v for k, v in bound.arguments.items()
                        if k not in {"self", "cls"}
                    }
                except (TypeError, ValueError):
                    return {"args": args, "kwargs": kwargs}

            if inspect.iscoroutinefunction(function):
                @wraps(function)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.span(span_name, kind=kind, input=call_input(args, kwargs)) as s:
                        s.output = await function(*args, **kwargs)
                        return s.output

                return async_wrapper

            @wraps(function)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.span(span_name, kind=kind, input=call_input(args, kwargs)) as s:
                    s.output = function(*args, **kwargs)
                    return s.output

            return wrapper

        return decorate

    def tool(self, function: Callable | None = None, *, name: str | None = None):
        decorate = self.decorator(kind="tool", name=name)
        return decorate(function) if function is not None else decorate

    def llm(self, function: Callable | None = None, *, name: str | None = None):
        decorate = self.decorator(kind="llm", name=name)
        return decorate(function) if function is not None else decorate

    def chain(self, function: Callable | None = None, *, name: str | None = None):
        decorate = self.decorator(kind="chain", name=name)
        return decorate(function) if function is not None else decorate

    @property
    def last_trace(self) -> Span | None:
        return self.traces[-1] if self.traces else None

    def save_trace(self, span: Span | None = None, path: str | Path | None = None) -> Path:
        root = span or self.last_trace
        if root is None:
            raise ValueError("No trace available to save.")
        if path is None:
            self._counter += 1
            label = str(root.input)[:40] if root.input is not None else root.name
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.lower()).strip("_") or "trace"
            path = self.output_dir / (
                f"{self.run_id}_{self._counter:04d}_{slug[:40]}_{root.span_id[:6]}.json"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
        tmp.write_text(
            json.dumps(root.to_dict(sanitizer=sanitize_value), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
        self.saved_paths.append(path)
        return path

    def save_last(self, path: str | Path | None = None) -> Path:
        return self.save_trace(self.last_trace, path=path)

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
