"""Tests for evalkit.tracer.save_trace fix and evalkit.agentic_evaluator."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.tracer import Tracer
from evalkit.agentic_evaluator import AgenticEvaluator


# ------------------------------------------------------------------ tracer

def test_save_trace_no_args_uses_last_trace(tmp_path):
    tracer = Tracer(tmp_path)
    with tracer.span("root", kind="chain", input={"q": "hi"}):
        pass
    path = tracer.save_trace()  # no args — must use the last_trace property
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["name"] == "root"


# ------------------------------------------------------- select_episodes

def _mk_ep(i, *, errors=None, scores=None, latency=1.0):
    return {
        "id": f"e{i}", "source": "test", "query": f"q{i}", "answer": "a",
        "tools": [], "trajectory_text": "", "n_tool_calls": 0,
        "latency_s": latency, "errors": errors, "scores": scores,
    }


def test_select_episodes_priority_order():
    ev = AgenticEvaluator()
    eps = []
    # e0-e2: errors
    for i in range(3):
        eps.append(_mk_ep(i, errors=[{"span": "t", "error": "boom"}]))
    # e3-e7: low scores
    for i in range(3, 8):
        eps.append(_mk_ep(i, scores={"grounding": {"score": 0.0, "label": "ungrounded"}}))
    # e8-e13: latency outliers (top 10% of 60 = 6)
    for i in range(8, 14):
        eps.append(_mk_ep(i, latency=50.0))
    # e14-e59: normal
    for i in range(14, 60):
        eps.append(_mk_ep(i, latency=1.0))
    ev.episodes = eps

    sel = ev.select_episodes(max_episodes=20)
    ids = [e["id"] for e in sel]
    # tiers 0-2 fill first, in order
    assert ids[:3] == ["e0", "e1", "e2"]
    assert ids[3:8] == ["e3", "e4", "e5", "e6", "e7"]
    assert ids[8:14] == ["e8", "e9", "e10", "e11", "e12", "e13"]
    # remaining 6 come from the normal pool only
    assert len(sel) == 20
    assert all(i.startswith("e") and int(i[1:]) >= 14 for i in ids[14:])


def test_select_episodes_small_set_returns_all():
    ev = AgenticEvaluator()
    ev.episodes = [_mk_ep(0), _mk_ep(1)]
    assert len(ev.select_episodes()) == 2


# ------------------------------------------------------------------ tools

def _ev_with_episodes(repo_root=None):
    ev = AgenticEvaluator(repo_root=repo_root)
    ev.episodes = [
        _mk_ep(0, errors=[{"span": "t", "error": "rate limited"}], latency=9.0),
        _mk_ep(1, scores={"g": {"score": 1.0, "label": "grounded"}}),
        _mk_ep(2, latency=30.0),
    ]
    ev.episodes[0]["tools"] = ["search_catalog"]
    ev.episodes[1]["tools"] = ["check_inventory"]
    ev.episodes[2]["tools"] = ["get_order_status"]
    ev.episodes[0]["query"] = "where is my sweater order " + "x" * 5000
    ev.episodes[0]["trajectory"] = [
        {"name": "search_catalog", "args": {"q": "sweater"}}
    ]
    return ev


def test_get_full_episode_returns_unclipped():
    ev = _ev_with_episodes()
    full = ev.get_full_episode("e0")
    assert len(full["query"]) > 5000  # unclipped, unlike the episode summary
    assert full["trajectory"][0]["name"] == "search_catalog"
    assert ev.get_full_episode("nope")["error"].startswith("unknown episode_id")


def test_filter_episodes():
    ev = _ev_with_episodes()
    assert [e["id"] for e in ev.filter_episodes(has_error=True)] == ["e0"]
    assert [e["id"] for e in ev.filter_episodes(tool_name="get_order_status")] == ["e2"]
    assert [e["id"] for e in ev.filter_episodes(min_latency_s=10.0)] == ["e2"]
    assert len(ev.filter_episodes()) == 3


def test_search_episodes():
    ev = _ev_with_episodes()
    hits = ev.search_episodes("SWEATER")
    assert [h["id"] for h in hits] == ["e0"]
    assert ev.search_episodes("rate limited")[0]["id"] == "e0"
    assert ev.search_episodes("") == []


def test_read_repo_file_traversal_protection(tmp_path):
    (tmp_path / "prompt.md").write_text("hello rubric", encoding="utf-8")
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        ev = AgenticEvaluator(repo_root=tmp_path)
        ok = ev.read_repo_file("prompt.md")
        assert "hello rubric" in ok["content"]
        bad = ev.read_repo_file("../outside_secret.txt")
        assert "error" in bad and "escapes" in bad["error"]
        missing = ev.read_repo_file("nope.txt")
        assert "error" in missing
        no_root = AgenticEvaluator().read_repo_file("x")
        assert "error" in no_root
    finally:
        outside.unlink(missing_ok=True)


# ------------------------------------------------------------- analysis

def test_analyze_dry_run():
    ev = _ev_with_episodes()
    result = ev.analyze(dry_run=True)
    assert result["dry_run"] is True
    assert result["n_episodes"] == 3
    assert "evidence" in result and "e0" in result["evidence"]


# -- fake genai plumbing for the agentic loop -------------------------------

def _part_fc(name, args):
    return SimpleNamespace(function_call=SimpleNamespace(name=name, args=args), text=None)


def _part_text(t):
    return SimpleNamespace(function_call=None, text=t)


def _resp(parts, text=None):
    content = SimpleNamespace(parts=parts)
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)], text=text)


def test_agentic_loop_tool_calling(monkeypatch):
    """Model calls get_full_episode, then emits final JSON — 2 steps."""
    import google.genai as genai_mod
    monkeypatch.setattr(genai_mod, "Client", lambda **kw: object())

    ev = _ev_with_episodes()
    ev.episodes[0]["scores"] = {"g": {"score": 0.0, "label": "ungrounded"}}
    queue = [
        _resp([_part_fc("get_full_episode", {"episode_id": "e0"})]),
        _resp([_part_text('{"summary":"s","findings":[],"human_checklist":[]}')]),
    ]
    seen_contents = []

    async def fake_generate(client, contents, config):
        seen_contents.append(list(contents))
        return queue.pop(0)

    monkeypatch.setattr(ev, "_generate_with_retry", fake_generate)
    out = asyncio.run(ev._agentic_loop("evidence"))
    assert "summary" in out
    assert ev.tool_calls_made == ["get_full_episode"]
    # 3 contents appended across 2 steps: user prompt, model fc, tool response
    assert len(seen_contents) == 2
    assert len(seen_contents[1]) == 3
    # tool response carried the unclipped episode back to the model
    fr = seen_contents[1][-1].parts[0].function_response
    assert fr.name == "get_full_episode"
    assert len(fr.response["result"]["query"]) > 5000


def test_analyze_with_mocked_loop(monkeypatch):
    ev = _ev_with_episodes()

    async def fake_loop(evidence):
        return json.dumps({
            "summary": "ok", "findings": [{
                "role": "trace_analyst", "severity": "minor",
                "pattern": "p", "episodes": ["e0"],
                "observation": "o", "root_cause": "c",
                "suggested_fix": None, "confidence": 0.5,
            }],
            "human_checklist": ["verify"],
        })

    monkeypatch.setattr(ev, "_agentic_loop", fake_loop)
    report = ev.analyze()
    assert report["summary"] == "ok"
    assert report["findings"][0]["episodes"] == ["e0"]
    assert report["tool_calls_made"] == []
    assert "## Findings" in ev.to_markdown()


def test_add_scores_preserves_full_explanation(tmp_path):
    long_expl = "because " * 200  # > 400 chars
    p = tmp_path / "scored.pkl"
    pd.DataFrame({
        "input": ["q1"],
        "correctness_score": [
            {"score": 0.0, "label": "bad", "explanation": long_expl}
        ],
    }).to_pickle(p)
    ev = AgenticEvaluator()
    ev.episodes.append(_mk_ep(0))
    ev.episodes[0]["query"] = "q1"
    assert ev.add_scores(p) == 1
    # stored unclipped — only the prompt-overview copy is clipped
    assert (ev.get_full_episode("e0")["scores"]["correctness"]["explanation"]
            == long_expl)
    sel = ev.select_episodes()
    assert sel[0]["scores"]["correctness"]["explanation"].endswith(
        "...[truncated]")


def test_verify_finding_before_after():
    ev = _ev_with_episodes()
    ev.report = {"findings": [{
        "pattern": "bad answers", "episodes": ["e0", "e1"],
    }]}

    def runner(payload):
        return {"answer": f"fixed: {payload['query'][:10]}", "errors": None,
                "tools_called": ["search_catalog"]}

    out = ev.verify_finding(0, runner)
    assert out["n_episodes"] == 2
    c0 = out["comparisons"][0]
    assert c0["before"]["errors"] == ev.episodes[0]["errors"]
    assert c0["after"]["answer"].startswith("fixed:")
    assert c0["after"]["tools"] == ["search_catalog"]
    with pytest.raises(IndexError):
        ev.verify_finding(9, runner)
