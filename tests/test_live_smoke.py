"""Live smoke tests: one minimal API call per configured model.

Verifies that the provider credentials and model names behind
AGENT_MODEL / GEMINI_MODEL / JUDGE_MODEL(+JUDGE_PROVIDER) /
AGENTIC_MODEL actually answer — catches bad keys, wrong model names,
and provider mismatches before a full eval run wastes time.

Off by default (real API calls). Run explicitly:

    EVALKIT_LIVE=1 python -m pytest tests/test_live_smoke.py -v
"""

import os
import sys
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(
    os.getenv("EVALKIT_LIVE") != "1",
    reason="live API smoke test — set EVALKIT_LIVE=1 to run",
)

_GEMINI_DEFAULT = "gemini-3.5-flash-lite"


def _targets():
    """(labels, provider, model) per env var, resolved like the consumers:
    agent_demo/agentic_evaluator fall back AGENT*/AGENTIC* -> GEMINI_MODEL
    -> default; judge_llm uses JUDGE_PROVIDER + JUDGE_MODEL."""
    gem = os.getenv("GEMINI_MODEL")
    jprovider = os.getenv("JUDGE_PROVIDER", "google")
    pairs = [
        ("GEMINI_MODEL", "google", gem),
        ("AGENT_MODEL", "google",
         os.getenv("AGENT_MODEL") or os.getenv("GEMINI_MODEL", _GEMINI_DEFAULT)),
        ("AGENTIC_MODEL", "google",
         os.getenv("AGENTIC_MODEL") or os.getenv("GEMINI_MODEL", _GEMINI_DEFAULT)),
        ("JUDGE", jprovider, os.getenv("JUDGE_MODEL", _GEMINI_DEFAULT)),
    ]
    # dedup identical provider+model pairs, keep all labels
    uniq = {}
    for label, provider, model in pairs:
        if model:
            uniq.setdefault((provider, model), []).append(label)
    return [("+".join(labels), provider, model)
            for (provider, model), labels in uniq.items()]


_TARGETS = _targets()


def _ping_google(model: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(http_options=types.HttpOptions(timeout=30_000))
    resp = client.models.generate_content(
        model=model,
        contents="ok",
        config=types.GenerateContentConfig(max_output_tokens=16),
    )
    return (resp.text or "").strip()


def _ping_openai(model: str) -> str:
    from openai import OpenAI
    client = OpenAI(timeout=30, max_retries=0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "ok"}],
        max_tokens=16,
    )
    return (resp.choices[0].message.content or "").strip()


def _ping(provider: str, model: str) -> str:
    if provider == "google":
        return _ping_google(model)
    if provider == "openai":
        return _ping_openai(model)
    pytest.fail(f"smoke test has no ping for provider {provider!r}")


@pytest.mark.parametrize("labels,provider,model", _TARGETS)
def test_live_model_answers(labels, provider, model):
    t0 = time.monotonic()
    text = _ping(provider, model)
    dt = time.monotonic() - t0
    # empty text is fine (reasoning models may return thoughts only) —
    # a completed call already proves key + model + provider are valid
    print(f"{labels}: {provider}/{model} answered in {dt:.1f}s -> {text[:80]!r}")
