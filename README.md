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
| `evalkit.runner` | `ask_adk()` / `run_batch()` — one query → trajectory row (tools, args, results, latency) with 429/503 retry |
| `evalkit.evals` | `load_template()`, `make_judge()` (rubric→ClassificationEvaluator), `tool_selection_evaluator()`/`tool_invocation_evaluator()` (spec-driven code evals), `run_evaluators()` |
| `evalkit.scoring` | `score_value/label`, `compare()`, `three_way()`, `cost_table()` |
| `evalkit.meta` | `judge_vs_human()` — kappa, confusion matrix, disagreements |
| `evalkit.annotations` | `log_annotations()` — results → Phoenix span annotations |
| `evalkit.agentic_evaluator` | `AgenticEvaluator` — LLM orchestrator: trace analyst, agent-as-judge, improvement engine, human-augmentation report |

Top-level names are lazy re-exports: `from evalkit import Tracer, make_judge`
works without importing every submodule.

**Contracts:**

- *Trajectory row*: `{input, output, tools_called, trajectory, trajectory_text, n_tool_calls, latency_s}` — produced by `evalkit.runner.ask_adk`, consumed by every evaluator/judge.
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

The test fixtures under `tests/mocked_*` double as a smoke test for this
CLI — e.g. `python -m evalkit.agentic_evaluator tests/mocked_traces --dry-run`.

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
| `evalkit.tracer` | pure Python span recording; `@tracer.tool`/`@tracer.chain` decorators work on any code |
| `evalkit.agentic_evaluator` | inputs (episodes: query, answer, tool calls, scores) are generic; only its internal LLM calls use `google.genai` |

**ADK-specific today:**

- `evalkit.runner` — `ask_adk()`/`run_batch()` import `google.adk` directly.
- `evalkit.instrumentation` — `traced_callbacks()` is built on ADK's
  `CallbackContext`/`LlmRequest`/`ToolContext` types.

**Using other frameworks (LangChain, LangGraph, etc.):** the *trajectory row*
contract is the seam. Any runner that emits
`{input, output, tools_called, trajectory, trajectory_text, n_tool_calls,
latency_s}` per query slots into every downstream module unchanged — write a
thin `ask_*()` for your framework (LangChain agents can reuse the same
callback-handler pattern; LangGraph has native tracing). For observability,
Phoenix ingests OpenInference/OTEL traces from LangChain/LangGraph natively,
so `evalkit.annotations` works there too.

`examples/agent_demo.py` shows the ADK reference pattern end-to-end.

## Setup to create the virtual environment with suitable dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env_example .env   # then add your GOOGLE_API_KEY
```

`.env` options:

| variable                | default                              | purpose                        |
|-------------------------|--------------------------------------|--------------------------------|
| `GOOGLE_API_KEY`        | —                                    | Gemini API key                 |
| `AGENT_MODEL`           | `GEMINI_MODEL` fallback              | model for the agent under eval |
| `AGENTIC_MODEL`         | `GEMINI_MODEL` fallback              | AgenticEvaluator meta-review   |
| `GEMINI_MODEL`          | `gemini-3.5-flash-lite`              | generic model fallback         |
| `LLM_MIN_INTERVAL_S`    | `4.5`                                | pace model calls (~13 RPM)     |
| `JUDGE_MODEL`           | `gemini-3.5-flash-lite`              | LLM-judge model override       |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006/v1/traces` | Phoenix OTLP endpoint     |
| `PHOENIX_PROJECT_NAME`  | `agent-tracer-adk`                   | Phoenix project                |

## Phoenix UI

```powershell
.\.venv\Scripts\phoenix.exe serve      # http://localhost:6006
```

Point your ADK agent's tracer at `PHOENIX_COLLECTOR_ENDPOINT` via
`evalkit.instrumentation.traced_callbacks(tracer)` to see spans in Phoenix.

## Tests

```powershell
pip install pytest   # not in requirements.txt — dev-only dependency
python -m pytest tests/ -v
```

Two suites, both self-contained — no LLM calls, no external data:

| file | what it covers |
|---|---|
| `test_agentic_evaluator.py` | unit tests for `evalkit.agentic_evaluator` (episode selection tiers, `get_full_episode`, filtering, repo-file traversal guard, tool-calling loop, `verify_finding`) and a `Tracer.save_trace` regression |
| `test_evalkit_fixtures.py` | fixture-based tests against `tests/mocked_traces/` and `tests/mocked_datasets/` — real `add_traces`/`add_trajectories`/`add_scores` intake paths plus a full `analyze(dry_run=True)` pipeline |

## Notes

- Traces are sanitized on export only — functions still see raw objects.
- `evalkit.tracer` also works standalone (the `@tracer.tool`/`@tracer.chain`
  decorators) for non-ADK code.
- The fixtures under `tests/mocked_*` are derived from the corresponding
  labs of the DeepLearning.AI course — sample data only, not a dependency
  on any lab code.

## Note if using free tier of Gemini

- **Rate limits**: the Gemini free tier allows ~15 requests/min. `llm_pacer()`
  (in `evalkit.instrumentation`) spaces model calls ~4.5s apart (tune
  `LLM_MIN_INTERVAL_S`), and runners retry on `429`/`503`. Pacing time shows
  up in `chain` span duration, not in `llm` spans.