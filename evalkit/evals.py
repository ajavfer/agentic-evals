"""Evaluator construction: rubric loading, LLM-judge factory, and
spec-driven code evaluators (tool_selection / tool_invocation).

Rubric convention: a markdown file with ONE fenced ``` block containing
the judge prompt; {placeholder} names must match dataframe column names.

Spec contract for code evaluators — spec_lookup(question) returns either
None (unscored -> 1.0) or::

    {"expected_tools": {"tool_a", ...} | None,
     "expected_args": {"tool_a": {"arg": "substring"}, ...} | None}

expected_tools: must be a subset of tools_called (extras allowed).
expected_args:  substring match, case-insensitive; one call to the tool
                must contain all of its expected substrings.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from openinference.instrumentation import suppress_tracing
from phoenix.evals import (
    LLM, ClassificationEvaluator, create_evaluator, evaluate_dataframe,
)

SpecLookup = Callable[[str], dict | None]


def load_template(rubric_path: str | Path) -> str:
    """Extract the single fenced ``` block from a rubric markdown file."""
    md = Path(rubric_path).read_text(encoding="utf-8")
    blocks = re.findall(r"```\n(.*?)```", md, re.DOTALL)
    if len(blocks) != 1:
        raise ValueError(f"{rubric_path}: expected exactly 1 fenced block, found {len(blocks)}")
    return blocks[0].strip()


def judge_llm(model: str | None = None, provider: str | None = None) -> LLM:
    """Gemini judge. JUDGE_MODEL/JUDGE_PROVIDER env override; flash-lite
    default because gemini-3.5-flash resolves to a 5-RPM free-tier model
    that 503s."""
    return LLM(
        provider=provider or os.getenv("JUDGE_PROVIDER", "google"),
        model=model or os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite"),
    )


def make_judge(
    name: str,
    *,
    rubric_path: str | Path | None = None,
    template: str | None = None,
    choices: dict[str, float],
    model: str | None = None,
    provider: str | None = None,
) -> ClassificationEvaluator:
    """Build a ClassificationEvaluator from a rubric file or raw template."""
    if template is None:
        if rubric_path is None:
            raise ValueError("make_judge needs rubric_path or template")
        template = load_template(rubric_path)
    return ClassificationEvaluator(
        name=name,
        llm=judge_llm(model=model, provider=provider),
        prompt_template=template,
        choices=choices,
    )


def tool_selection_evaluator(spec_lookup: SpecLookup, name: str = "tool_selection"):
    """1.0 when expected_tools ⊆ tools_called; 1.0 when no spec."""

    @create_evaluator(name=name, kind="code")
    def _eval(input: str, tools_called: list) -> float:
        spec = spec_lookup(input)
        if spec is None or not spec.get("expected_tools"):
            return 1.0
        return 1.0 if spec["expected_tools"].issubset(set(tools_called)) else 0.0

    return _eval


def tool_invocation_evaluator(spec_lookup: SpecLookup, name: str = "tool_invocation"):
    """1.0 when some single call to each expected tool contains all of that
    tool's expected arg substrings (case-insensitive).
    Skips tools that were never called (that's selection's failure)."""

    @create_evaluator(name=name, kind="code")
    def _eval(input: str, trajectory: list) -> float:
        spec = spec_lookup(input)
        if spec is None:
            return 1.0
        for tool_name, expected_args in (spec.get("expected_args") or {}).items():
            calls = [c for c in trajectory if c["tool"] == tool_name]
            if not calls:
                continue
            if not any(
                all(expected.lower() in str(call["args"].get(key, "")).lower()
                    for key, expected in expected_args.items())
                for call in calls
            ):
                return 0.0
        return 1.0

    return _eval


def run_evaluators(
    df: pd.DataFrame,
    evaluators: list,
    *,
    required: list[str] | None = None,
    hide_tqdm_bar: bool = True,
) -> pd.DataFrame:
    """evaluate_dataframe under suppress_tracing.

    `required`: evaluator names whose score column must exist afterwards —
    pass your LLM judges here; if every judge call failed, phoenix returns
    no column and we exit cleanly instead of a downstream KeyError."""
    with suppress_tracing():
        results = evaluate_dataframe(
            dataframe=df, evaluators=evaluators, hide_tqdm_bar=hide_tqdm_bar
        )
    for name in required or []:
        if f"{name}_score" not in results.columns:
            raise RuntimeError(
                f"Judge '{name}' produced no scores — all calls failed. "
                "Check API errors above."
            )
    return results
