"""Meta-evaluation: score an LLM judge against human labels.

Run the judge blind on held-out rows, then measure agreement:
classification report, confusion matrix, Cohen's kappa, and every
disagreement with the judge's explanation.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix


def judge_vs_human(
    results: pd.DataFrame,
    *,
    score_col: str,
    human: pd.Series,
    inputs: pd.Series | None = None,
    label_order: list[str] | None = None,
) -> pd.DataFrame:
    """Compare judge labels (from a *_score column) to human labels.

    results: dataframe that contains score_col (dict cells with 'label').
    human:   human label series aligned to results' index.
    inputs:  optional question/input series for the disagreement listing.
    Returns the comparison dataframe; prints metrics + disagreements.
    """
    judge_labels = results[score_col].apply(
        lambda c: c["label"] if isinstance(c, dict) else "unscored"
    )
    explanations = results[score_col].apply(
        lambda c: c.get("explanation", "") if isinstance(c, dict) else ""
    )

    comparison = pd.DataFrame(
        {
            "input": inputs.values if inputs is not None else results.index,
            "human": human.values,
            "judge": judge_labels.values,
            "explanation": explanations.values,
        }
    )

    scored = judge_labels != "unscored"
    n_unscored = int((~scored).sum())
    if n_unscored:
        print(f"{n_unscored} of {len(results)} rows could not be scored by the judge; "
              "they are left out of the metrics below.")

    h, j = human[scored], judge_labels[scored]
    labels = label_order or sorted(set(h) | set(j))

    print("\nclassification report (rows=human):")
    print(classification_report(h, j, labels=labels, digits=3, zero_division=0))

    print("confusion matrix (rows=human, cols=judge; order:", labels, "):")
    print(confusion_matrix(h, j, labels=labels))

    kappa = cohen_kappa_score(h, j)
    print(f"\nCohen's kappa: {kappa:.3f}")
    print("Rule of thumb: 0.4-0.6 good, >0.7 excellent. "
          "Two human experts often land at just 0.2-0.3.")

    disagreements = comparison[scored & (comparison["human"] != comparison["judge"])]
    print(f"\n{len(disagreements)} disagreements:")
    for _, row in disagreements.iterrows():
        print(f"\nQ: {row['input']}\n  human={row['human']}  judge={row['judge']}")
        print(f"  judge says: {row['explanation'][:300]}")

    return comparison
