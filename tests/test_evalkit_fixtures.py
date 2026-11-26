"""Fixture-based tests: tests/mocked_traces + tests/mocked_datasets.

These fixtures are derived from the labs of the
DeepLearning.AI 'Evaluating and Improving Agentic Applications' course —
they are sample data only, not a dependency on any lab code.

The tests below exercise AgenticEvaluator's real intake paths
(add_traces / add_trajectories / add_scores) against these artifacts.
Everything runs offline — no LLM calls, no network access, and nothing
outside tests/ is required, keeping evalkit self-contained.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.agentic_evaluator import AgenticEvaluator

ROOT = Path(__file__).resolve().parents[1]
MOCKED_TRACES = ROOT / "tests" / "mocked_traces"
MOCKED_DATASETS = ROOT / "tests" / "mocked_datasets"

TRACES_DIR = MOCKED_TRACES / "traces"
V1_RAW = MOCKED_TRACES / "v1_trajectories.pkl"
V1_SCORED = MOCKED_TRACES / "v1_scored.pkl"
V3_SCORED = MOCKED_TRACES / "v3_scored.pkl"
META_EVAL = MOCKED_DATASETS / "meta_eval_labeled.csv"


def test_fixtures_exist():
    for p in [TRACES_DIR, V1_SCORED, V3_SCORED, META_EVAL]:
        assert p.exists(), f"missing fixture: {p}"
    assert len(list(TRACES_DIR.glob("*.json"))) >= 2


def test_add_traces_loads_real_spans():
    ev = AgenticEvaluator()
    n = ev.add_traces(TRACES_DIR)
    assert n == 3
    ep = ev.episodes[0]
    assert ep["query"]
    assert ep["tools"], "expected tool spans in trace"
    assert ep["trajectory_text"]
    # episodes keep full text; get_full_episode serves on-demand detail
    assert ev.get_full_episode(ep["id"])["trajectory_text"] == ep["trajectory_text"]


def test_add_trajectories_attaches_inline_scores():
    """*_score columns in the pkl attach during add_trajectories."""
    ev = AgenticEvaluator()
    assert ev.add_trajectories(V1_SCORED) == 3
    ep = ev.episodes[0]
    assert ep["scores"], "score columns should attach to episodes"
    labels = {k: v.get("label") for k, v in ep["scores"].items()}
    assert "ungrounded" in labels.values()  # sweater care fails under v1
    # single unclipped store — the prompt overview clips a copy instead
    assert ev.get_full_episode(ep["id"])["trajectory_text"] == ep["trajectory_text"]


def test_add_scores_merges_onto_traj_only_episodes():
    """add_scores merges a separate scored file by matching query text."""
    ev = AgenticEvaluator()
    assert ev.add_trajectories(V1_RAW) == 3
    assert not ev.episodes[0].get("scores")
    assert ev.add_scores(V1_SCORED) == 3
    labels = {k: v.get("label") for k, v in ev.episodes[0]["scores"].items()}
    assert "ungrounded" in labels.values()


def test_full_pipeline_dry_run():
    """traces + two scored pkls -> prioritized sample -> evidence payload."""
    ev = AgenticEvaluator(repo_root=ROOT)
    ev.add_traces(TRACES_DIR)
    ev.add_trajectories(V1_SCORED)
    ev.add_trajectories(V3_SCORED)
    assert len(ev.episodes) == 9

    result = ev.analyze(dry_run=True)
    assert result["n_episodes"] == 9
    assert result["n_selected"] == 9
    evidence = result["evidence"]
    assert "v1_scored" in evidence and "trace-" in evidence
    # the v1 ungrounded label must be visible in the evidence payload
    assert "ungrounded" in evidence


def test_labeled_dataset_loads_for_meta_eval():
    """The labeled csv feeds judge-vs-human episodes (meta-eval pattern)."""
    df = pd.read_csv(META_EVAL)
    assert {"input", "human", "judge", "explanation"} <= set(df.columns)
    assert len(df) >= 10
    ev = AgenticEvaluator()
    for i, row in df.iterrows():
        ev.episodes.append({
            "id": f"meta-{i}", "source": "meta_eval",
            "query": row["input"],
            "scores": {
                "policy_compliance": {"label": row["judge"],
                                      "explanation": str(row["explanation"])[:400]},
                "human_reference": {"label": row["human"]},
            },
        })
    # every episode carries both judge and human labels
    assert all("policy_compliance" in e["scores"] and
               "human_reference" in e["scores"] for e in ev.episodes)
    result = ev.analyze(dry_run=True)
    assert result["n_episodes"] == len(df)
