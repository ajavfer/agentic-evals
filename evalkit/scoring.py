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


def _mean_score(df: pd.DataFrame, evaluator: str):
    """Mean evaluator score across (possibly repeated) case rows.
    None when nothing was scored."""
    vals = pd.to_numeric(score_value(df, evaluator), errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else None


def _verdict(before, after) -> str:
    if before is None or after is None:
        return "?"
    if before == 0.0 and after == 1.0:
        return "FIXED"
    if before == 1.0 and after == 0.0:
        return "REGRESSED"
    if after > before:
        return "IMPROVED"
    if after < before:
        return "REGRESSED"
    return "pass" if after == 1.0 else "fail" if after == 0.0 else "same"


def compare(base: pd.DataFrame, new: pd.DataFrame, evaluator: str,
            title: str, *, key: str = "case") -> None:
    """Per-case before/after on one evaluator score (means over repeats)."""
    b = base.set_index(key)
    n = new.set_index(key)
    print(f"\n{title}")
    print(f"{key:<22} {'before':>7} {'after':>7}  change")
    for case in b.index.union(n.index):
        bv = _mean_score(b.loc[[case]], evaluator) if case in b.index else None
        nv = _mean_score(n.loc[[case]], evaluator) if case in n.index else None
        bs = f"{bv:.1f}" if bv is not None else "-"
        ns = f"{nv:.1f}" if nv is not None else "-"
        print(f"{case:<22} {bs:>7} {ns:>7}  {_verdict(bv, nv)}")


def three_way(runs: list[tuple[str, pd.DataFrame]], evaluator: str,
              title: str, *, key: str = "case") -> None:
    """Side-by-side scores for 2+ versions, plus tool-call counts."""
    print(f"\n{title}")
    headers = [label for label, _ in runs]
    print(f"{key:<22} " + " ".join(f"{h:>12}" for h in headers)
          + "  " + " ".join(f"{f'calls_{h}':>9}" for h in headers))
    cases = list(dict.fromkeys(runs[0][1][key].tolist()))
    for case in cases:
        cells, call_cells = [], []
        for _, df in runs:
            row = df[df[key] == case]
            v = _mean_score(row, evaluator)
            cells.append(f"{v:.1f}" if v is not None else "-")
            call_cells.append(
                f"{row['n_tool_calls'].mean():g}"
                if "n_tool_calls" in row and len(row) else "-"
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
        line = (f"  {label}: mean calls={calls:.2f} "
                f"mean latency={lat:.1f}s total calls={total}")
        if "input_tokens" in df and "output_tokens" in df:
            line += (f" tokens in/out={int(df['input_tokens'].sum())}/"
                     f"{int(df['output_tokens'].sum())} "
                     f"(mean {df['input_tokens'].mean():.0f}/"
                     f"{df['output_tokens'].mean():.0f})")
        print(line)
