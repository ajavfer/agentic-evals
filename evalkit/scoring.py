"""Score extraction and cross-version comparison for experiment dataframes.

Score columns produced by phoenix evaluate_dataframe look like
{"name": ..., "score": 0.0|1.0, "label": ..., "explanation": ...}.
These helpers flatten them and diff two versions per case.
"""

from __future__ import annotations

import pandas as pd


def score_value(df: pd.DataFrame, evaluator: str):
    """Numeric score (0.0/1.0) per row for an evaluator column."""
    col = f"{evaluator}_score"
    if col not in df.columns:
        return pd.Series([None] * len(df))
    return df[col].apply(lambda c: c.get("score") if isinstance(c, dict) else None)


def score_label(df: pd.DataFrame, evaluator: str):
    """Label string (e.g. 'grounded'/'ungrounded') per row."""
    col = f"{evaluator}_score"
    if col not in df.columns:
        return pd.Series(["unscored"] * len(df))
    return df[col].apply(lambda c: c.get("label") if isinstance(c, dict) else "unscored")


def label_counts(df: pd.DataFrame, evaluator: str) -> pd.Series:
    return score_label(df, evaluator).value_counts()


def _verdict(before, after) -> str:
    if before is None or after is None:
        return "?"
    if before == 0.0 and after == 1.0:
        return "FIXED"
    if before == 1.0 and after == 0.0:
        return "REGRESSED"
    return "pass" if after == 1.0 else "fail"


def compare(base: pd.DataFrame, new: pd.DataFrame, evaluator: str,
            title: str, *, key: str = "case") -> None:
    """Per-case before/after on one evaluator score."""
    b = base.set_index(key)
    n = new.set_index(key)
    print(f"\n{title}")
    print(f"{key:<22} {'before':>7} {'after':>7}  change")
    for case in b.index.union(n.index):
        bv = score_value(b.loc[[case]], evaluator).iloc[0] if case in b.index else None
        nv = score_value(n.loc[[case]], evaluator).iloc[0] if case in n.index else None
        bs = f"{bv:.1f}" if bv is not None else "-"
        ns = f"{nv:.1f}" if nv is not None else "-"
        print(f"{case:<22} {bs:>7} {ns:>7}  {_verdict(bv, nv)}")


def three_way(runs: list[tuple[str, pd.DataFrame]], evaluator: str,
              title: str, *, key: str = "case") -> None:
    """Side-by-side scores for 2+ versions, plus tool-call counts."""
    print(f"\n{title}")
    headers = [label for label, _ in runs]
    print(f"{key:<22} " + " ".join(f"{h:>12}" for h in headers)
          + "  " + " ".join(f"calls_{h}:>9" for h in headers))
    cases = runs[0][1][key].tolist()
    for case in cases:
        cells, call_cells = [], []
        for _, df in runs:
            row = df[df[key] == case]
            v = score_value(row, evaluator).iloc[0] if len(row) else None
            cells.append(f"{v:.1f}" if v is not None else "-")
            call_cells.append(
                str(row["n_tool_calls"].iloc[0]) if "n_tool_calls" in row and len(row) else "-"
            )
        print(f"{case:<22} " + " ".join(f"{c:>12}" for c in cells)
              + "  " + " ".join(f"{c:>9}" for c in call_cells))


def cost_table(dfs: dict[str, pd.DataFrame]) -> None:
    """Mean tool calls and latency per version (needs n_tool_calls/latency_s)."""
    print("\ncost per version:")
    for label, df in dfs.items():
        calls = df["n_tool_calls"].mean() if "n_tool_calls" in df else float("nan")
        lat = df["latency_s"].mean() if "latency_s" in df else float("nan")
        total = int(df["n_tool_calls"].sum()) if "n_tool_calls" in df else "-"
        print(f"  {label}: mean calls={calls:.2f} mean latency={lat:.1f}s total calls={total}")
