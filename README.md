# evalkit

**A generic, reusable evaluation toolkit for any agentic application** —
not tied to any one agent, framework demo, or dataset.

Local observability and evaluation toolkit for [Google ADK](https://google.github.io/adk-docs/)
agents, refactored from the DeepLearning.AI's 'Evaluating and Improving Agentic Applications' labs (M1–M5).

`evalkit` is a library, not an app: point it at *your* agent — any ADK agent,
any tools, any domain — and feed it *your* traces, trajectories, and eval
results. Nothing in the package assumes a specific agent exists.

## Layout

```
evalkit/          the toolkit — see below
examples/         agent_demo.py — reference ADK agent (Northwind Apparel
                  shop) showing the evalkit integration pattern;
                  run_eval_demo.py runs it end-to-end (run → code evals +
                  LLM judge → scored pickle)
tests/            pytest suite + fixtures (mocked_traces/, mocked_datasets/)
```

## Modules

Every module is agent-agnostic: they operate on trajectories, tool calls,
spans, and score dataframes — never on any specific agent's code or data.

| module | what it gives you |
|---|---|
| `evalkit.tracer` | `Span`/`Tracer`: chain/llm/tool spans, sanitize, JSON export, tree render |
| `evalkit.instrumentation` | `traced_callbacks(tracer)` + `llm_pacer()` — drop-in ADK callbacks |
| `evalkit.runner` | `ask_adk()` / `run_batch()` — one query → trajectory row (tools, args, results, latency, `input_tokens`/`output_tokens`) with 429/503 retry; `save_trace` in `finally` so failed attempts still persist |
| `evalkit.evals` | `load_template()`, `make_judge()` (rubric→`ClassificationEvaluator`, `JUDGE_PROVIDER`/`JUDGE_MODEL` + `OPENAI_*` envs select provider/model), `tool_selection_evaluator()`/`tool_invocation_evaluator()` (spec-driven code evals), `run_evaluators()` |
| `evalkit.scoring` | `score_value/label`, `compare()`, `three_way()`, `cost_table()` |
| `evalkit.meta` | `judge_vs_human()` — kappa, confusion matrix, disagreements |
| `evalkit.annotations` | `log_annotations()` — results → Phoenix span annotations |
| `evalkit.agentic_evaluator` | `AgenticEvaluator` — LLM orchestrator: trace analyst, agent-as-judge, improvement engine, human-augmentation report |

Top-level names are lazy re-exports: `from evalkit import Tracer, make_judge`
works without importing every submodule.

**Contracts:**

- *Trajectory row*: `{input, output, tools_called, trajectory, trajectory_text, n_tool_calls, latency_s, input_tokens, output_tokens}` — produced by `evalkit.runner.ask_adk`, consumed by every evaluator/judge.
- *Spec lookup*: `fn(question) -> {"expected_tools": set|None, "expected_args": {tool: {arg: substring}}|None}` for the code evaluators.
- *Rubric*: markdown file with one fenced ` ``` ` block; `{placeholder}` names match dataframe columns.

### Agentic_Evaluator (`evalkit.agentic_evaluator`)

A meta-layer that reviews evalkit's own output. Feed it trace JSONs,
trajectory pickles, and/or scored result pickles; it normalizes everything
into "episodes" (query, answer, tool calls, evaluator scores + explanations)
and asks an LLM to act in four roles — trace analyst, agent-as-judge
(judges the *trajectory*, surfaces unknown-unknowns), improvement engine
(proposes fixes, never applies them), and human-augmentation (emits a
review-ready markdown report with a decision checklist).

```powershell
python -m evalkit.agentic_evaluator traces_dir/ scored.pkl --dry-run   # inspect evidence, no API call
python -m evalkit.agentic_evaluator run_v1_scored.pkl run_v3_scored.pkl --out report.md
```

The report is stamped with a UTC `generated` timestamp in its header (and in
the report dict as `generated_at`) so stale reports are distinguishable from
fresh ones.

The test fixtures under `tests/mocked_*` double as a smoke test for this
CLI — e.g. `python -m evalkit.agentic_evaluator tests/mocked_traces/v1_scored.pkl --dry-run`.

## Portability

`evalkit` is layered: the evaluation core is framework-agnostic, and only the
execution/instrumentation edge is ADK-specific today.

**Works with any agent framework as-is:**

| module | why it's portable |
|---|---|
| `evalkit.evals` | builds on `phoenix.evals` + pandas; consumes any results dataframe |
| `evalkit.scoring` | pandas only — diffs score columns across versions |
| `evalkit.meta` | pandas + sklearn — judge-vs-human kappa, no agent imports |
| `evalkit.annotations` | pandas + `phoenix.client` |
| `evalkit.tracer` | pure Python span recording (`Span`/`Tracer`, sanitize, JSON export, tree render); no framework imports |
| `evalkit.agentic_evaluator` | inputs (episodes: query, answer, tool calls, scores) are generic; only its internal LLM calls use `google.genai` |

**ADK-specific today:**

- `evalkit.runner` — `ask_adk()`/`run_batch()` import `google.adk` directly.
- `evalkit.instrumentation` — `traced_callbacks()` is built on ADK's
  `CallbackContext`/`LlmRequest`/`ToolContext` types.

**Using other frameworks (LangChain, LangGraph, etc.):** the *trajectory row*
contract is the seam. Any runner that emits
`{input, output, tools_called, trajectory, trajectory_text, n_tool_calls,
latency_s, input_tokens, output_tokens}` per query slots into every
downstream module unchanged — write a thin `ask_*()` for your framework
(LangChain agents can reuse the same callback-handler pattern; LangGraph has
native tracing). For observability,
Phoenix ingests OpenInference/OTEL traces from LangChain/LangGraph natively,
so `evalkit.annotations` works there too.

`examples/agent_demo.py` shows the ADK reference pattern end-to-end.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env_example .env   # then add your GOOGLE_API_KEY
```

`.env` options (see `.env_example` for commented details):

| variable | default | purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini API key |
| `AGENT_MODEL` | `GEMINI_MODEL` fallback | model for the agent under eval |
| `AGENTIC_MODEL` | `GEMINI_MODEL` fallback | AgenticEvaluator meta-review |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | generic model fallback |
| `LLM_MIN_INTERVAL_S` | `4.5` | pace model calls (~13 RPM) |
| `JUDGE_PROVIDER` | `google` | judge backend (`google` or `openai`) |
| `JUDGE_MODEL` | `gemini-3.5-flash-lite` | LLM-judge model (must match provider) |
| `OPENAI_API_KEY` | — | key for non-Google judge endpoints |
| `OPENAI_BASE_URL` | — | OpenAI-compatible endpoint, e.g. `https://api.ifm.ai/v1` |
| `EVALKIT_LIVE` | unset | set to `1` to enable live (real-LLM) tests |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006/v1/traces` | Phoenix OTLP endpoint |
| `PHOENIX_PROJECT_NAME` | `agent-tracer-adk` | Phoenix project |

## Phoenix UI

```powershell
.\.venv\Scripts\phoenix.exe serve      # http://localhost:6006
```

`evalkit.instrumentation.traced_callbacks(tracer)` records spans into the
local JSON `Tracer` (files under your chosen trace dir) — Phoenix does not
consume them directly. To see eval results in Phoenix, run the server and
push annotations with `evalkit.annotations.log_annotations()` after
`run_evaluators()`.

## Tests

```powershell
pip install pytest   # not in requirements.txt — dev-only dependency
python -m pytest tests/ -v
```

Offline suite — no LLM calls, no external data:

| file | what it covers |
|---|---|
| `test_agentic_evaluator.py` | unit tests for `evalkit.agentic_evaluator` (episode selection tiers, `get_full_episode`, filtering, repo-file traversal guard, tool-calling loop, `verify_finding`) and a `Tracer.save_trace` regression |
| `test_evalkit_fixtures.py` | fixture-based tests against `tests/mocked_traces/` and `tests/mocked_datasets/` — real `add_traces`/`add_trajectories`/`add_scores` intake paths plus a full `analyze(dry_run=True)` pipeline |
| `test_runner.py` | `ask_adk`/`run_batch` behavior: retry/pacing, token fields, trace persistence on error |
| `test_scoring.py` | `score_value/label`, `compare()`, `three_way()`, `cost_table()` |
| `test_evals_provider.py` | `make_judge` provider/env wiring (`JUDGE_PROVIDER`, `JUDGE_MODEL`, `OPENAI_*`) |

Live suite — real LLM calls, gated behind `EVALKIT_LIVE=1` (skipped by
default; `$env:EVALKIT_LIVE="1"; python -m pytest tests/` runs everything):

| file | what it covers |
|---|---|
| `test_live_smoke.py` | connectivity ping for agent/judge/meta-reviewer models |
| `test_judge_quality.py` | judge golden set — 8 known-correct/incorrect synthetic trajectories scored by the real judge; reports agreement %, Cohen's kappa, and per-class precision/recall (currently 8/8, κ=1.0 on `IFM/K2-Horizon`) |

## Notes

- Traces are sanitized on export only — functions still see raw objects.
- `evalkit.tracer` is framework-agnostic span recording; for non-ADK code
  instantiate `Tracer`/`Span` directly and emit `llm`/`tool`/`chain` spans
  around your own calls (Phoenix accepts the same JSON shape).
- The fixtures under `tests/mocked_*` are derived from the corresponding
  labs of the DeepLearning.AI course — sample data only, not a dependency
  on any lab code.

### Gemini free tier

- **Rate limits**: the Gemini free tier allows ~15 requests/min. `llm_pacer()`
  (in `evalkit.instrumentation`) spaces model calls ~4.5s apart (tune
  `LLM_MIN_INTERVAL_S`), and runners retry on `429`/`503`. Pacing time shows
  up in `chain` span duration, not in `llm` spans. Free-tier capacity for
  individual models varies by demand — `503 UNAVAILABLE` spikes are
  server-side and transient; retry later or switch `AGENT_MODEL`.