"""Unit tests for evalkit.evals judge provider/model resolution.

The real phoenix LLM client is never constructed (it would need an API
key), so LLM/ClassificationEvaluator are monkeypatched. phoenix itself is
still required by evalkit.evals at import time — skip where it is absent.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("phoenix.evals")

import evalkit.evals as evals


def test_judge_llm_provider_param_env_default(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(evals, "LLM", FakeLLM)
    monkeypatch.delenv("JUDGE_PROVIDER", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)

    evals.judge_llm(provider="openai")
    assert captured["provider"] == "openai"

    monkeypatch.setenv("JUDGE_PROVIDER", "anthropic")
    evals.judge_llm()
    assert captured["provider"] == "anthropic"   # env fallback

    evals.judge_llm(provider="google")
    assert captured["provider"] == "google"      # param beats env

    monkeypatch.delenv("JUDGE_PROVIDER")
    evals.judge_llm()
    assert captured["provider"] == "google"      # default


def test_judge_llm_model_param_env_default(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(evals, "LLM", FakeLLM)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)

    evals.judge_llm(model="gpt-4o")
    assert captured["model"] == "gpt-4o"              # param wins

    monkeypatch.setenv("JUDGE_MODEL", "IFM/K2-Horizon")
    evals.judge_llm()
    assert captured["model"] == "IFM/K2-Horizon"      # env fallback

    evals.judge_llm(model="override")
    assert captured["model"] == "override"           # param beats env

    monkeypatch.delenv("JUDGE_MODEL")
    evals.judge_llm()
    assert captured["model"] == "gemini-3.5-flash-lite"  # default


def test_make_judge_passes_provider(monkeypatch):
    seen = {}

    class FakeLLM:
        def __init__(self, **kw):
            self.kwargs = kw

    class FakeEvaluator:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(evals, "LLM", FakeLLM)
    monkeypatch.setattr(evals, "ClassificationEvaluator", FakeEvaluator)
    evals.make_judge("j", template="t {input}", choices={"a": 1.0},
                     provider="openai", model="m1")
    assert seen["llm"].kwargs["provider"] == "openai"
    assert seen["llm"].kwargs["model"] == "m1"
