"""End-to-end eval demo: run examples/agent_demo.py against a question set,
score trajectories with evalkit code evaluators + an LLM judge, save the
scored dataframe (CSV for reading + a sibling .pkl for AgenticEvaluator).

Requires GOOGLE_API_KEY in .env — the agent itself calls Gemini, and the
judge adds one LLM call per row. Free-tier pacing is handled inside
build_agent via llm_pacer().

Run from the repo root (with .venv activated):

 python examples/run_eval_demo.py --limit 2                     # test (2 traces only)
 python examples/run_eval_demo.py --limit 2 --agentic-evaluator # test (2 traces only) with meta-review
 python examples/run_eval_demo.py --no-judge                    # code (deterministic) evaluators only
 python examples/run_eval_demo.py                               # full run + Agent-as-Judge
 python examples/run_eval_demo.py --agentic-evaluator           # + meta-review report

Or feed the scored pickle into the meta-reviewer separately:

    python -m evalkit.agentic_evaluator examples/demo_scored.pkl --dry-run
    # (the .pkl, not the .csv — CSV loses the nested score/trajectory dicts)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> evalkit
sys.path.insert(0, str(Path(__file__).resolve().parent))        # examples/ -> agent_demo

import pandas as pd
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from agent_demo import build_agent
from evalkit.evals import make_judge, run_evaluators, tool_invocation_evaluator, tool_selection_evaluator
from evalkit.runner import run_batch
from evalkit.scoring import score_value
from evalkit.tracer import Tracer

APP_NAME = "shop_eval"
ROOT = Path(__file__).resolve().parents[1]
RUBRIC = Path(__file__).resolve().parent / "rubrics" / "correctness.md"
# same resolution as agent_demo.build_agent — recorded on every row so the
# scored df/report can distinguish agent-under-eval from analyst/judge models
AGENT_MODEL = (os.getenv("AGENT_MODEL")
               or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))

QUESTIONS = [
    "Where is order ORD-1002?",
    "Do you have the merino sweater in stock in size L?",
    "What's your return policy?",
    "How much is the wool beanie and what colors does it come in?",
    "Is the olive hoodie available in medium?",
    "What's the difference between the Classic Tee and the Heavyweight Tee?",
]


def spec_lookup(question: str) -> dict | None:
    """Expected tools/args per question. None -> row is unscored (1.0).

    expected_tools must be a SUBSET of tools_called; expected_args substrings
    are checked only for tools that were actually called.
    """
    q = question.lower()
    if "ord-1002" in q:
        return {
            "expected_tools": {"get_order_status"},
            "expected_args": {"get_order_status": {"order_id": "ORD-1002"}},
        }
    if "return policy" in q:
        return {"expected_tools": {"return_policy"}}
    if "beanie" in q:
        return {
            "expected_tools": {"search_catalog"},
            "expected_args": {"search_catalog": {"query": "beanie"}},
        }
    if "hoodie" in q:
        return {"expected_tools": {"search_catalog", "check_inventory"}}
    if "sweater" in q or "tee" in q:
        return {"expected_tools": {"search_catalog"}}
    return None


def make_runner():
    """One runner+tracer per batch so llm_pacer state is shared."""
    tracer = Tracer("traces")
    agent = build_agent(tracer)
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    return runner, session_service, tracer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of questions (cheap smoke test)")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the LLM correctness judge (code evals only)")
    parser.add_argument("--out", default="examples/demo_scored.csv",
                        help="where to save the scored dataframe "
                             "(.csv for reading, .pkl also written for "
                             "AgenticEvaluator — it needs real dict cells)")
    parser.add_argument("--agentic-evaluator", action="store_true",
                        help="after scoring, run the AgenticEvaluator "
                             "meta-review (LLM calls) and write a markdown "
                             "report next to --out")
    args = parser.parse_args()

    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS

    print(f"running {len(questions)} queries against the demo agent — "
          "real LLM calls, paced ~4.5s apart; this takes a while...\n",
          flush=True)
    rows = asyncio.run(run_batch(questions, make_runner, app_name=APP_NAME,
                                 progress=True))
    df = pd.DataFrame(rows)
    df["agent_model"] = AGENT_MODEL
    print(f"ran {len(df)} queries -> {df['n_tool_calls'].sum()} tool calls, "
          f"traces in ./traces/")

    evaluators = [
        tool_selection_evaluator(spec_lookup),
        tool_invocation_evaluator(spec_lookup),
    ]
    required = []
    if not args.no_judge:
        evaluators.append(make_judge(
            "correctness",
            rubric_path=RUBRIC,
            choices={"correct": 1.0, "incorrect": 0.0},
        ))
        required = ["correctness"]

    print("scoring trajectories (LLM judge calls, if enabled)...", flush=True)
    scored = run_evaluators(df, evaluators, required=required)
    out = Path(args.out)
    if out.suffix == ".csv":
        scored.to_csv(out, index=False)
        # AgenticEvaluator needs dict cells — CSV stringifies them, so the
        # pickle is what the meta-review CLI should consume.
        scored.to_pickle(out.with_suffix(".pkl"))
    else:
        scored.to_pickle(out)
    means = "  ".join(
        f"{name}={score_value(scored, name).mean():.2f}"
        for name in [c[: -len("_score")] for c in scored.columns
                     if c.endswith("_score")]
    )
    print(f"mean scores: {means}")
    scored_path = out if out.suffix != ".csv" else out.with_suffix(".pkl")
    print(f"\nscored dataframe -> {out}"
          + (f" (+ {scored_path} for AgenticEvaluator)" if scored_path != out else ""))
    print("meta-review: python -m evalkit.agentic_evaluator "
          f"{scored_path} --dry-run")

    if args.agentic_evaluator:
        from evalkit.agentic_evaluator import AgenticEvaluator
        print("\nrunning AgenticEvaluator meta-review (LLM calls)...",
              flush=True)
        ev = AgenticEvaluator(repo_root=ROOT)
        ev.add_trajectories(scored_path)
        ev.analyze()
        report_path = out.with_name(out.stem + "_agentic_report.md")
        ev.to_markdown(report_path)
        print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
