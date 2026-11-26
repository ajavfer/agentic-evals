"""Phoenix annotation logging: flatten evaluator score cells into the
span-annotation dataframe Phoenix expects, keyed by context.span_id."""

from __future__ import annotations

import pandas as pd


def to_annotation_dataframe(
    results: pd.DataFrame,
    score_column: str,
    span_id_column: str = "context.span_id",
) -> pd.DataFrame:
    rows = []
    for _, row in results.iterrows():
        cell = row[score_column]
        cell = cell if isinstance(cell, dict) else {}
        score = cell.get("score", 0.0)
        label = cell.get("label") or ("pass" if score >= 0.5 else "fail")
        rows.append({
            "span_id": row[span_id_column],
            "label": label,
            "score": score,
            "explanation": cell.get("explanation", ""),
        })
    return pd.DataFrame(rows)


def log_annotations(
    results: pd.DataFrame,
    *,
    score_column: str,
    annotation_name: str,
    annotator_kind: str = "LLM",
    span_id_column: str = "context.span_id",
) -> None:
    """Log evaluator results as Phoenix span annotations. No-op with a
    message if Phoenix isn't reachable."""
    try:
        import phoenix.client

        client = phoenix.client.Client()
        ann_df = to_annotation_dataframe(results, score_column, span_id_column)
        client.spans.log_span_annotations_dataframe(
            dataframe=ann_df,
            annotator_kind=annotator_kind,
            annotation_name=annotation_name,
        )
        print(f"logged {len(ann_df)} {annotation_name} annotations to Phoenix")
    except Exception as exc:
        print(f"Phoenix annotation logging skipped: {exc}")
