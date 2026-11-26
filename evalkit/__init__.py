"""evalkit — reusable agent-evaluation toolkit.

Tracing:      evalkit.tracer (Span/Tracer), evalkit.instrumentation (ADK)
Running:      evalkit.runner (trajectory capture, batch + retry)
Evaluating:   evalkit.evals (rubrics, judges, code evaluators)
Scoring:      evalkit.scoring (extract + compare versions)
Meta:         evalkit.meta (judge vs human labels)
Phoenix:      evalkit.annotations (span annotation logging)

Top-level names are lazy re-exports: `from evalkit import Tracer, make_judge`
works without paying the import cost of runner/evals until used.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    # tracer
    "Span": "evalkit.tracer",
    "Tracer": "evalkit.tracer",
    "walk_spans": "evalkit.tracer",
    "format_trace_tree": "evalkit.tracer",
    "sanitize_value": "evalkit.tracer",
    # instrumentation
    "llm_pacer": "evalkit.instrumentation",
    "traced_callbacks": "evalkit.instrumentation",
    # runner
    "ask_adk": "evalkit.runner",
    "run_batch": "evalkit.runner",
    "format_trajectory": "evalkit.runner",
    # evals
    "load_template": "evalkit.evals",
    "judge_llm": "evalkit.evals",
    "make_judge": "evalkit.evals",
    "tool_selection_evaluator": "evalkit.evals",
    "tool_invocation_evaluator": "evalkit.evals",
    "run_evaluators": "evalkit.evals",
    # scoring
    "score_value": "evalkit.scoring",
    "score_label": "evalkit.scoring",
    "label_counts": "evalkit.scoring",
    "compare": "evalkit.scoring",
    "three_way": "evalkit.scoring",
    "cost_table": "evalkit.scoring",
    # meta
    "judge_vs_human": "evalkit.meta",
    # annotations
    "to_annotation_dataframe": "evalkit.annotations",
    "log_annotations": "evalkit.annotations",
    # agentic evaluator
    "AgenticEvaluator": "evalkit.agentic_evaluator",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'evalkit' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # cache after first access
    return value
