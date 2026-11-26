# Evalkit: Local Tracing & Agentic Evaluation

**Evalkit** is a lightweight, vendor-neutral Python toolkit for tracing AI agent
executions and running bounded, human-in-the-loop agentic evaluations.

Built as an open-source technical companion to the
[HUMXUS Executive Briefings](https://www.humxus.com), Evalkit demonstrates how
engineering teams can move beyond static prompt testing and establish
independent quality oversight without locking into a proprietary observability
platform.

---

## Why Evalkit?

* **Local, portable telemetry:** Captures nested `chain`, `llm`, and `tool`
  spans as sanitized JSON files with redaction of sensitive values on export.
* **True agentic evaluation:** Instead of dumping truncated logs into a single
  prompt, `AgenticEvaluator` runs a bounded tool-calling loop. It reviews a
  prioritized overview of episodes and uses read-only tools to inspect full
  trajectories, search across all loaded runs, and check repository files
  before diagnosing root causes.
* **Propose only, human in the loop:** The evaluator never modifies code or
  applies fixes automatically. It outputs a structured Markdown report with
  evidence, confidence scores, and a human review checklist.
* **Non-destructive verification:** Includes a replay hook (`verify_finding`)
  to test candidate fixes against failing episodes and compare before/after
  behavior without touching disk.

---

## Architecture

### Modules

| module | what it gives you |
|---|---|
| `evalkit.tracer` | `Span`/`Tracer`: nested `chain`/`llm`/`tool` spans via `tracer.span()`, sanitize, JSON export |
| `evalkit.instrumentation` | `traced_callbacks(tracer)` + `llm_pacer()` — drop-in ADK callbacks |
| `evalkit.runner` | `ask_adk()` / `run_batch()` — query → trajectory row (tools, args, results, latency, token counts) with 429/503 retry |
| `evalkit.evals` | `make_judge()` (rubric → `ClassificationEvaluator`), spec-driven `tool_selection`/`tool_invocation` code evals, `run_evaluators()` |
| `evalkit.scoring` | `score_value/label`, `compare()`, `three_way()`, `cost_table()` |
| `evalkit.meta` | `judge_vs_human()` — kappa, confusion matrix, disagreements |
| `evalkit.annotations` | `log_annotations()` — results → Phoenix span annotations |
| `evalkit.agentic_evaluator` | `AgenticEvaluator` — LLM orchestrator: trace analyst, agent-as-judge, improvement engine, human-augmentation report |

### 1. Local Tracer (`evalkit/tracer.py`)

Records execution trees with the `tracer.span(name, kind=...)` context
manager — nest spans to build `chain` → `llm`/`tool` hierarchies — and exports
each finished trace as sanitized JSON via `tracer.save_trace()`. Pure Python,
no framework imports.

### 2. Agentic Evaluator (`evalkit/agentic_evaluator.py`)

Orchestrates four evaluation roles over your traces, trajectories, and
evaluator scores:

1. **Trace Analyst:** Identifies recurring errors, patterns, and anomalies
   across runs.
2. **Agent-as-Judge:** Evaluates the full decision trajectory — not just the
   final answer — to surface *unknown unknowns*.
3. **Improvement Engine:** Groups related failures, infers root causes, and
   proposes concrete fixes (prompt rules, rubric edits, or data fixes).
4. **Human Augmentation:** Produces a review-ready report and verification
   checklist, stamped with a UTC `generated` timestamp.

**Prioritized Sampling:** `select_episodes()` automatically prioritizes
episodes in four tiers: (1) recorded errors, (2) low evaluator scores,
(3) top-10% latency outliers, and (4) a seeded random sample of normal runs.

**Read-Only Investigation Tools:** During its loop, the evaluator can call:

* `get_full_episode(episode_id)` — inspects unclipped queries, answers, and
  tool inputs/outputs.
* `filter_episodes(...)` — queries across all loaded episodes by error status,
  tool name, or latency.
* `search_episodes(keyword)` — case-insensitive search across queries,
  answers, tools, and errors.
* `read_repo_file(rel_path)` — enabled only when `repo_root` is set; strictly
  bounded inside the repository root to prevent path traversal.

---

## Installation

```bash
git clone https://github.com/ajavfer/agentic-evals.git
cd agentic-evals
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key in a `.env` file (see .env-example and never commit `.env` to version control):

```bash
GOOGLE_API_KEY=your_api_key_here
AGENTIC_MODEL=gemini-3.5-flash-lite   # meta-review model
AGENTIC_MAX_EPISODES=50               # prioritized sample cap per run
```

See `.env_example` for all options — including `JUDGE_PROVIDER`/`JUDGE_MODEL`/
`OPENAI_*` for pointing the LLM judge at a non-Google endpoint.

---

## Quick Start

### 1. Trace an Agent Run

```python
from evalkit.tracer import Tracer

tracer = Tracer(output_dir="traces")

with tracer.span("customer_support_agent", kind="chain",
                 input={"query": "sweater price"}):
    with tracer.span("search_catalog", kind="tool",
                     input={"query": "sweater"}):
        result = "Wool sweater - $89"

trace_path = tracer.save_trace()
```

### 2. Run the Agentic Evaluator (Python)

```python
from evalkit.agentic_evaluator import AgenticEvaluator

ev = AgenticEvaluator(repo_root=".", max_steps=8)
ev.add_traces("traces/")          # trace .json files (or a single file)
# Optionally load trajectory DataFrames (.pkl or .csv) and merge scores:
# ev.add_trajectories("trajectories.pkl")
# ev.add_scores("scored_results.pkl")

report = ev.analyze()
ev.to_markdown("agentic_report.md")
```

### 3. Run via CLI

```bash
# Dry run: inspect the prioritized evidence payload without calling the LLM
python -m evalkit.agentic_evaluator traces/ --dry-run

# Full analysis with repository context and custom output path
python -m evalkit.agentic_evaluator traces/ --repo-root . --max-steps 8 --out report.md
```

### 4. Verify a Proposed Fix

```python
def candidate_runner(payload: dict) -> dict:
    # Run your updated prompt or agent against payload["query"]
    return {"answer": "updated response", "errors": None,
            "tools_called": ["search_catalog"]}

comparison = ev.verify_finding(finding_index=0, candidate_runner=candidate_runner)
```

---

## Running Tests

The offline suite mocks all LLM calls and verifies tracing, prioritized
sampling, read-only tools (including path-traversal protection), the
multi-step tool loop, and `verify_finding`:

```bash
pytest tests/ -v
```

Two live tests (`test_live_smoke.py`, `test_judge_quality.py`) hit real LLM
endpoints and are skipped unless enabled:

```powershell
$env:EVALKIT_LIVE="1"; python -m pytest tests/ -v
```

---

## About HUMXUS

This repository is maintained by HUMXUS (www.humxus.com), an independent AI advisory practice
focused on enterprise AI governance, evaluation architectures, and
vendor-neutral due diligence.
