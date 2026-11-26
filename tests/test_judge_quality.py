"""Judge-quality gate: score synthetic trajectories with known labels
through the real configured judge (JUDGE_PROVIDER/JUDGE_MODEL).

Verifies three things for an unfamiliar judge model:
1. Schema reliability — every row returns a valid label, score, and explanation
   (no silent None/unscored fallbacks).
2. Discrimination — passes both positive and negative golden cases, including
   honest refusals and subtle faithfulness drift.
3. Diagnostics — prints the judge's explanation on any mismatch.

Live test (one judge call per case); run explicitly:

$env:EVALKIT_LIVE=1; python -m pytest tests/test_judge_quality.py -v -s
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("phoenix")

pytestmark = pytest.mark.skipif(
    os.getenv("EVALKIT_LIVE") != "1",
    reason="live judge-quality test — set EVALKIT_LIVE=1 to run",
)

from evalkit.evals import make_judge, run_evaluators  # noqa: E402
from evalkit.runner import format_trajectory  # noqa: E402
from evalkit.scoring import score_label, score_value  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report, cohen_kappa_score,
)

RUBRIC = Path(__file__).resolve().parents[1] / "examples" / "rubrics" / "correctness.md"
VALID_LABELS = {"correct", "incorrect"}

ORD_1002 = {"status": "in_transit", "items": ["SWE-001"],
            "tracking": ["Label created", "In transit — arriving Thursday"]}
ORD_1001 = {"status": "delivered", "items": ["TEE-001", "SOX-001"],
            "tracking": ["Label created", "In transit", "Out for delivery", "Delivered"]}
POLICY = ("Items may be returned within 30 days of delivery with tags attached. "
          "Exchanges for a different size or color are free; refunds go to the "
          "original payment method within 5 business days.")


def _case(name, question, trajectory, output, expected):
    return {"name": name, "input": question, "trajectory": trajectory,
            "trajectory_text": format_trajectory(trajectory),
            "output": output, "expected": expected}


CASES = [
    # --- clearly correct ---
    _case(
        "order_status_correct",
        "Where is order ORD-1002?",
        [{"tool": "get_order_status", "args": {"order_id": "ORD-1002"},
          "result": ORD_1002}],
        "Your order ORD-1002 is in transit and should arrive Thursday.",
        "correct",
    ),
    _case(
        "return_policy_correct",
        "What's your return policy?",
        [{"tool": "return_policy", "args": {}, "result": POLICY}],
        "Items can be returned within 30 days of delivery with tags attached. "
        "Exchanges for a different size or color are free, and refunds go to "
        "the original payment method within 5 business days.",
        "correct",
    ),
    _case(
        "catalog_answer_correct",
        "How much is the wool beanie and what colors does it come in?",
        [{"tool": "search_catalog", "args": {"query": "beanie"},
          "result": [{"id": "BEA-001", "name": "Wool Beanie",
                      "category": "accessories", "price": 24.0,
                      "colors": ["black", "rust", "cream"], "sizes": ["One size"],
                      "care": "Hand wash cold, lay flat to dry"}]}],
        "The Wool Beanie is $24.00 and comes in black, rust, and cream.",
        "correct",
    ),
    _case(
        "honest_not_found",
        "Where is order ORD-9999?",
        [{"tool": "get_order_status", "args": {"order_id": "ORD-9999"},
          "result": {"order_id": "ORD-9999", "found": False,
                     "reason": "order_not_found"}}],
        "I couldn't find an order with ID ORD-9999. Please double-check the order number.",
        "correct",
    ),
    # --- clearly incorrect ---
    _case(
        "stock_hallucination",
        "Is the merino sweater in stock in size XL?",
        [{"tool": "check_inventory",
          "args": {"product_id": "SWE-001", "size": "XL"},
          "result": {"product_id": "SWE-001", "size": "XL",
                     "available": False, "quantity": 0}}],
        "Yes, the Merino Wool Sweater is in stock in size XL.",
        "incorrect",
    ),
    _case(
        "wrong_order_status",
        "Where is order ORD-1001?",
        [{"tool": "get_order_status", "args": {"order_id": "ORD-1001"},
          "result": ORD_1001}],
        "Your order ORD-1001 is still in transit.",
        "incorrect",
    ),
    _case(
        "wrong_price",
        "How much is the quilted jacket?",
        [{"tool": "search_catalog", "args": {"query": "jacket"},
          "result": [{"id": "JKT-001", "name": "Quilted Jacket",
                      "category": "outerwear", "price": 129.0,
                      "colors": ["black", "olive"], "sizes": ["S", "M", "L", "XL"],
                      "care": "Machine wash cold, gentle cycle"}]}],
        "The Quilted Jacket is $89.00.",
        "incorrect",
    ),
    _case(
        "ungrounded_no_tool",
        "Is the Classic Tee in stock in size M and how much is it?",
        [],
        "Yes, the Classic Tee is in stock in size M for $19.99.",
        "incorrect",
    ),
]


def test_judge_labels_golden_set():
    df = pd.DataFrame([
        {"input": c["input"], "output": c["output"],
         "trajectory_text": c["trajectory_text"]}
        for c in CASES
    ])
    judge = make_judge("correctness", rubric_path=RUBRIC,
                       choices={"correct": 1.0, "incorrect": 0.0})
    scored = run_evaluators(df, [judge], required=["correctness"])

    scores = score_value(scored, "correctness")
    labels = score_label(scored, "correctness")

    unscored = []
    missing_explanations = []
    mismatches = []

    for i, case in enumerate(CASES):
        cell = scored["correctness_score"].iloc[i]
        score = scores.iloc[i]
        label = labels.iloc[i]
        expl = (cell.get("explanation", "") if isinstance(cell, dict) else "").strip()

        if label not in VALID_LABELS or pd.isna(score):
            unscored.append(f"{case['name']} (label={label!r}, score={score!r})")
            continue
        if not expl:
            missing_explanations.append(case["name"])

        mark = "ok" if label == case["expected"] else "MISS"
        print(f"{mark} {case['name']}: expected={case['expected']} "
              f"judge={label} (score={score})")
        if label != case["expected"]:
            print(f"     explanation: {expl[:240]}")
            mismatches.append(case["name"])

    # schema gates first — an all-failed judge must hit these clean asserts,
    # not a ValueError from sklearn on empty series
    assert not unscored, f"judge returned invalid/unscored outputs: {unscored}"
    assert not missing_explanations, (
        f"judge returned empty explanations on: {missing_explanations}"
    )

    # --- KPI: same sklearn metrics evalkit.meta.judge_vs_human reports ---
    human = pd.Series([c["expected"] for c in CASES], name="human")
    judge_labels = labels.rename("judge")
    report = classification_report(
        human, judge_labels, labels=["correct", "incorrect"],
        output_dict=True, zero_division=0,
    )
    kappa = cohen_kappa_score(human, judge_labels)
    judge_model = f"{os.getenv('JUDGE_PROVIDER', 'google')}/{os.getenv('JUDGE_MODEL', '?')}"
    print(f"\njudge KPI ({judge_model}):")
    print(f"  agreement:  {report['accuracy']:.0%}  ({int(report['accuracy'] * len(human))}/{len(human)})")
    print(f"  kappa:       {kappa:.3f}")
    for cls in ("correct", "incorrect"):
        print(f"  {cls:>9}:  precision={report[cls]['precision']:.2f}  "
              f"recall={report[cls]['recall']:.2f}")
    if mismatches:
        print(f"  misses:     {', '.join(mismatches)}")

    # Allow at most 1 miss for LLM nondeterminism.
    assert len(mismatches) <= 1, (
        f"judge disagreed on {mismatches} ({len(mismatches)}/{len(CASES)} misses)"
    )