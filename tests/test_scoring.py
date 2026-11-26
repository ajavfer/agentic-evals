"""Unit tests for evalkit.scoring aggregation over repeated case rows."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.scoring import _verdict, compare, cost_table, three_way


def _scored(rows):
    """rows: list of (case, score) — duplicates = repeated runs."""
    return pd.DataFrame({
        "case": [c for c, _ in rows],
        "correctness_score": [{"score": s} for _, s in rows],
    })


def test_verdict_fractional_and_binary():
    assert _verdict(0.5, 1.0) == "IMPROVED"
    assert _verdict(1.0, 0.5) == "REGRESSED"
    assert _verdict(0.5, 0.5) == "same"
    assert _verdict(0.0, 1.0) == "FIXED"
    assert _verdict(1.0, 0.0) == "REGRESSED"
    assert _verdict(1.0, 1.0) == "pass"
    assert _verdict(None, 1.0) == "?"


def test_compare_aggregates_repeated_cases(capsys):
    base = _scored([("c1", 0.0), ("c1", 1.0)])   # mean 0.5
    new = _scored([("c1", 1.0), ("c1", 1.0)])    # mean 1.0
    compare(base, new, "correctness", "t")
    out = capsys.readouterr().out
    assert "0.5" in out and "1.0" in out
    assert "IMPROVED" in out


def test_compare_unscored_rows(capsys):
    compare(pd.DataFrame({"case": ["c1"]}),
            pd.DataFrame({"case": ["c1"]}), "correctness", "t")
    out = capsys.readouterr().out
    assert "-" in out and "?" in out


def test_three_way_aggregates_repeated_cases(capsys):
    base = _scored([("c1", 0.0), ("c1", 1.0), ("c2", 1.0)])
    new = _scored([("c1", 1.0), ("c1", 1.0), ("c2", 0.0)])
    three_way([("v1", base), ("v2", new)], "correctness", "t")
    out = capsys.readouterr().out
    # c1 printed once despite duplicate rows; means shown
    assert out.count("c1") == 1
    assert "0.5" in out and "1.0" in out and "0.0" in out


def test_cost_table_includes_tokens(capsys):
    df = pd.DataFrame({
        "n_tool_calls": [1, 3],
        "latency_s": [1.0, 3.0],
        "input_tokens": [100, 300],
        "output_tokens": [40, 60],
    })
    cost_table({"v1": df, "v2": pd.DataFrame({"n_tool_calls": [1]})})
    out = capsys.readouterr().out
    assert "in/out=400/100" in out          # totals
    assert "mean 200/50" in out             # means
    assert "in/out" not in out.split("v2")[1]  # absent without token cols
