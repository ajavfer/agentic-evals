"""evalkit/agentic_evaluator.py — an agent that evaluates the evaluators' output.

Orchestrates the four Agentic_Evaluator roles on top of evalkit's raw material
(traces, trajectories, evaluator scores):

1. Trace Analyst      — first-pass read of traces/trajectories: recurring
                        errors, patterns, unexpected behavior, concerns.
2. Agent-as-Judge     — judges the *trajectory*, not just the answer:
                        was the path reasonable, what are the root causes,
                        which "unknown unknowns" appear.
3. Improvement Engine — groups similar failures, infers root cause,
                        proposes a concrete fix (prompt rule, spec change,
                        rubric tweak, data fix). Proposes only — never
                        applies changes itself.
4. Human Augmentation — emits a review-ready report: findings with
                        evidence, suggested fixes, and a decision checklist
                        for the human to approve/reject.

The analysis is a bounded agentic loop: the model gets a prioritized episode
overview, then investigates on demand via read-only tools
(get_full_episode / filter_episodes / search_episodes / read_repo_file) before
emitting the final JSON or MD report. It cannot write files or apply fixes.

Usage:
    from evalkit.agentic_evaluator import AgenticEvaluator
    ev = AgenticEvaluator(repo_root=".")
    ev.add_trajectories("trajectories.pkl")
    ev.add_scores("run_v1_scored.pkl")
    report = ev.analyze()          # -> dict with findings + summary
    ev.to_markdown("report.md")

CLI:
    python -m evalkit.agentic_evaluator traces_dir/ scored.pkl --dry-run
    python -m evalkit.agentic_evaluator scored.pkl --repo-root . --out report.md
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import inspect
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from evalkit.tracer import walk_spans

MAX_FIELD = 1500          # chars kept per trajectory/answer field
MAX_EPISODES = int(os.getenv("AGENTIC_MAX_EPISODES", "50"))
MAX_TOOL_OUTPUT = 12000   # chars returned to the model per tool call
_FAIL_LABELS = {
    "fail", "failed", "failure", "ungrounded", "non_compliant",
    "non-compliant", "incorrect", "false", "no", "violated",
}


# ---------------------------------------------------------------- evidence

def _clip(value: Any, limit: int = MAX_FIELD) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...[truncated]"
    return value


def _c(value: Any, clip: bool) -> Any:
    return _clip(value) if clip else value


def _trace_to_episode(path: Path, idx: int, *, clip: bool = True) -> dict[str, Any] | None:
    """One tracer JSON file -> episode summary."""
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    spans = list(walk_spans(root))
    tools = [s for s in spans if s.get("kind") == "tool"]
    errors = [
        {"span": s.get("name"), "error": s.get("error")}
        for s in spans if s.get("status") == "error"
    ]
    inp = root.get("input") or {}
    out = root.get("output") or {}
    return {
        "id": f"trace-{idx}",
        "source": "trace",
        "query": _c(inp.get("user_query") or inp.get("query") or "", clip),
        "answer": _c(out.get("final_answer") or out.get("answer") or "", clip),
        "tools": [t.get("name") for t in tools],
        "trajectory_text": _c(
            "\n".join(
                f"{t.get('name')}({_c(t.get('input'), clip)}) -> {_c(t.get('output'), clip)}"
                for t in tools
            ),
            clip,
        ),
        "n_tool_calls": len(tools),
        "latency_s": round((root.get("duration_ms") or 0) / 1000, 2),
        "errors": errors or None,
    }


def _df_to_episodes(df: pd.DataFrame, source: str, *, clip: bool = True) -> list[dict[str, Any]]:
    rows = []
    for i, row in df.iterrows():
        traj_text = row.get("trajectory_text")
        if traj_text is None and row.get("trajectory") is not None:
            traj_text = json.dumps(row.get("trajectory"), default=str)
        ep = {
            "id": f"{source}-{i}",
            "source": source,
            "query": _c(row.get("input") or row.get("question") or "", clip),
            "answer": _c(row.get("output") or row.get("response") or "", clip),
            "tools": row.get("tools_called") or [],
            "trajectory_text": _c(traj_text or "", clip),
            "n_tool_calls": int(row.get("n_tool_calls") or len(row.get("tools_called") or [])),
            "latency_s": row.get("latency_s"),
            "case": row.get("case"),
            "split": row.get("split"),
            "model": row.get("agent_model") or row.get("model"),
        }
        # attach evaluator outcomes (score/label/explanation) when present
        scores = {}
        for col in df.columns:
            if col.endswith("_score") and isinstance(row.get(col), dict):
                cell = row[col]
                scores[col[: -len("_score")]] = {
                    "score": cell.get("score"),
                    "label": cell.get("label"),
                    "explanation": _c(cell.get("explanation", ""), clip) if clip else cell.get("explanation", ""),
                }
        if scores:
            ep["scores"] = scores
        rows.append(ep)
    return rows


# ------------------------------------------------------------------ prompt

SYSTEM = """You are Agentic_Evaluator, an agentic evaluator for LLM agents. You review
evaluation evidence — agent trajectories, tool calls, evaluator scores and
judge explanations — and act in four roles:

1. TRACE ANALYST: find recurring errors, emerging patterns, unexpected
   behaviors, and areas of concern a human would miss at scale.
2. AGENT-AS-JUDGE: judge whether each trajectory was *reasonable*, not just
   whether the final answer passed. Name failure modes and root causes —
   including "unknown unknowns" the rubric did not anticipate.
3. IMPROVEMENT ENGINE: for each distinct failure pattern, group it, infer
   the most likely root cause (prompt gap, missing data, tool limitation,
   rubric blind spot, model variance), and propose ONE concrete fix
   (prompt rule text, spec/evaluator change, data backfill, or rubric edit).
   You propose; you never apply.
4. HUMAN AUGMENTATION: every finding must carry enough evidence for a human
   reviewer to approve or reject it quickly.

You are given a prioritized overview of the episodes (errors and low scores
first). You have read-only investigation tools to inspect the evidence
further before concluding:

- get_full_episode(episode_id): unclipped query, answer, full tool inputs/
  outputs, errors and scores for one episode.
- filter_episodes(has_error, tool_name, min_latency_s, limit): query across
  ALL loaded episodes, not just the overview sample.
- search_episodes(keyword, limit): case-insensitive search over queries,
  answers, tool names and error messages.
- read_repo_file(rel_path): read a file inside the repository root to check
  prompts, rubrics, specs or code referenced by a suspected root cause.

Workflow: skim the overview, use the tools to confirm suspicious
trajectories and verify root causes against the actual evidence (and repo
files when relevant), then output your final report. Do not call tools
indiscriminately — investigate what changes your conclusions.

Finish by replying with ONLY a JSON object (no prose, no fences):
{
  "summary": "2-4 sentence overall assessment",
  "findings": [
    {
      "role": "trace_analyst|agent_as_judge|improvement_engine",
      "severity": "critical|major|minor|info",
      "pattern": "short name for the recurring pattern",
      "episodes": ["episode-id", ...],
      "observation": "what happened, with evidence",
      "root_cause": "most likely cause",
      "suggested_fix": "concrete proposed change, or null",
      "confidence": 0.0-1.0
    }
  ],
  "human_checklist": ["questions the human should verify before shipping"]
}"""


def _evidence_json(episodes: list[dict]) -> str:
    return json.dumps(episodes, indent=1, default=str)


# ------------------------------------------------------------- orchestrator

class AgenticEvaluator:
    """Collects evidence (traces, trajectories, scores) and runs the
    four-role analysis as a bounded tool-using agentic loop."""

    def __init__(
        self,
        model: str | None = None,
        *,
        repo_root: str | Path | None = None,
        max_steps: int = 8,
    ):
        self.model = model or os.getenv(
            "AGENTIC_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        )
        self.repo_root = Path(repo_root).resolve() if repo_root else None
        self.max_steps = max_steps
        self.episodes: list[dict[str, Any]] = []
        # unclipped copies keyed by episode id, for on-demand inspection
        self._raw_episodes: dict[str, dict[str, Any]] = {}
        self.report: dict[str, Any] | None = None
        self.tool_calls_made: list[str] = []
        # pace our own genai calls — llm_pacer() only covers ADK agents,
        # and stronger models have tighter free-tier RPM limits
        self._min_interval = float(os.getenv("LLM_MIN_INTERVAL_S", "4.5"))
        self._last_llm_call = 0.0

    # -- evidence intake ----------------------------------------------------

    def add_traces(self, path: str | Path) -> int:
        """Load tracer JSON files from a file or directory."""
        p = Path(path)
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        before = len(self.episodes)
        for f in files:
            ep = _trace_to_episode(f, len(self.episodes))
            if ep:
                self.episodes.append(ep)
                raw = _trace_to_episode(f, len(self.episodes) - 1, clip=False)
                if raw:
                    self._raw_episodes[ep["id"]] = raw
        return len(self.episodes) - before

    def add_trajectories(self, path: str | Path) -> int:
        """Load trajectory rows from .pkl or .csv."""
        p = Path(path)
        df = pd.read_pickle(p) if p.suffix == ".pkl" else pd.read_csv(p)
        before = len(self.episodes)
        eps = _df_to_episodes(df, p.stem)
        raws = {e["id"]: e for e in _df_to_episodes(df, p.stem, clip=False)}
        for ep in eps:
            self.episodes.append(ep)
            if ep["id"] in raws:
                self._raw_episodes[ep["id"]] = raws[ep["id"]]
        return len(self.episodes) - before

    def add_scores(self, path: str | Path, *, key: str = "input") -> int:
        """Merge evaluator score columns into matching episodes by query text."""
        p = Path(path)
        df = pd.read_pickle(p) if p.suffix == ".pkl" else pd.read_csv(p)
        score_cols = [c for c in df.columns if c.endswith("_score")]
        if not score_cols:
            return 0
        by_query = {str(r.get("input") or r.get("question")): r for _, r in df.iterrows()}
        n = 0
        for ep in self.episodes:
            if ep.get("scores"):
                continue  # already scored — first source wins (queries repeat across versions)
            row = by_query.get(str(ep.get("query")))
            if row is None:
                continue
            scores = {}
            for col in score_cols:
                cell = row.get(col)
                if isinstance(cell, dict):
                    scores[col[: -len("_score")]] = {
                        "score": cell.get("score"),
                        "label": cell.get("label"),
                        "explanation": _clip(cell.get("explanation", ""), 400),
                    }
            if scores:
                ep["scores"] = scores
                if ep["id"] in self._raw_episodes:
                    self._raw_episodes[ep["id"]]["scores"] = copy.deepcopy(scores)
                n += 1
        return n

    # -- episode selection ---------------------------------------------------

    @staticmethod
    def _has_low_score(ep: dict[str, Any]) -> bool:
        for s in (ep.get("scores") or {}).values():
            sc = s.get("score")
            if isinstance(sc, (int, float)) and sc < 0.5:
                return True
            if str(s.get("label") or "").lower() in _FAIL_LABELS:
                return True
        return False

    def select_episodes(self, max_episodes: int = MAX_EPISODES) -> list[dict[str, Any]]:
        """Prioritized sample for the initial prompt.

        Order: (1) episodes with errors, (2) low evaluator scores,
        (3) top-10% latency outliers, (4) seeded random sample of the rest.
        """
        eps = list(self.episodes)
        if len(eps) <= max_episodes:
            return eps
        n = len(eps)
        lat = sorted((e.get("latency_s") or 0 for e in eps), reverse=True)
        cutoff = lat[max(0, int(n * 0.1) - 1)]
        tiers: list[list[dict[str, Any]]] = [[], [], [], []]
        for e in eps:
            lat_s = e.get("latency_s") or 0
            if e.get("errors"):
                tiers[0].append(e)
            elif self._has_low_score(e):
                tiers[1].append(e)
            elif lat_s > 0 and lat_s >= cutoff:
                tiers[2].append(e)
            else:
                tiers[3].append(e)
        tiers[2].sort(key=lambda e: -(e.get("latency_s") or 0))
        out: list[dict[str, Any]] = []
        for tier in tiers[:3]:
            for e in tier:
                if len(out) >= max_episodes:
                    return out
                out.append(e)
        pool = tiers[3]
        random.Random(42).shuffle(pool)
        out.extend(pool[: max_episodes - len(out)])
        return out

    # -- investigation tools (read-only) --------------------------------------

    def _episode_by_id(self, episode_id: str) -> dict[str, Any] | None:
        for ep in self.episodes:
            if ep.get("id") == episode_id:
                return ep
        return None

    def get_full_episode(self, episode_id: str) -> dict[str, Any]:
        """Unclipped episode detail for on-demand inspection."""
        ep = self._episode_by_id(episode_id)
        if ep is None:
            return {"error": f"unknown episode_id: {episode_id}"}
        full = copy.deepcopy(self._raw_episodes.get(episode_id) or ep)
        full["id"] = ep.get("id")
        if ep.get("scores"):
            full["scores"] = copy.deepcopy(ep["scores"])
        return full

    def filter_episodes(
        self,
        has_error: bool | None = None,
        tool_name: str | None = None,
        min_latency_s: float | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query across ALL loaded episodes (not just the prompt sample)."""
        out = []
        for ep in self.episodes:
            if has_error is not None and bool(ep.get("errors")) != has_error:
                continue
            if tool_name and tool_name not in (ep.get("tools") or []):
                continue
            if min_latency_s is not None and (ep.get("latency_s") or 0) < min_latency_s:
                continue
            out.append({
                "id": ep.get("id"), "query": _clip(ep.get("query"), 300),
                "tools": ep.get("tools"), "n_tool_calls": ep.get("n_tool_calls"),
                "latency_s": ep.get("latency_s"), "has_errors": bool(ep.get("errors")),
                "scores": {k: {"score": v.get("score"), "label": v.get("label")}
                           for k, v in (ep.get("scores") or {}).items()},
            })
            if len(out) >= min(limit, 50):
                break
        return out

    def search_episodes(self, keyword: str, limit: int = 20) -> list[dict[str, Any]]:
        """Case-insensitive search over queries, answers, tools and errors."""
        kw = str(keyword).lower()
        if not kw:
            return []
        out = []
        for ep in self.episodes:
            haystacks = [
                str(ep.get("query") or ""), str(ep.get("answer") or ""),
                str(ep.get("trajectory_text") or ""),
                " ".join(ep.get("tools") or []),
                json.dumps(ep.get("errors") or [], default=str),
            ]
            if any(kw in h.lower() for h in haystacks):
                out.append({
                    "id": ep.get("id"), "query": _clip(ep.get("query"), 300),
                    "tools": ep.get("tools"), "has_errors": bool(ep.get("errors")),
                    "scores": {k: v.get("label") for k, v in (ep.get("scores") or {}).items()},
                })
            if len(out) >= min(limit, 50):
                break
        return out

    def read_repo_file(self, rel_path: str) -> dict[str, Any]:
        """Read a file inside repo_root (traversal-safe, length-capped)."""
        if self.repo_root is None:
            return {"error": "repo_root not configured"}
        try:
            target = (self.repo_root / rel_path).resolve()
        except Exception as exc:
            return {"error": f"bad path: {exc}"}
        if not target.is_relative_to(self.repo_root):
            return {"error": f"path escapes repo_root: {rel_path}"}
        if not target.is_file():
            return {"error": f"not a file: {rel_path}"}
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return {"error": f"read failed: {exc}"}
        return {
            "path": rel_path,
            "content": text[:MAX_TOOL_OUTPUT],
            "truncated": len(text) > MAX_TOOL_OUTPUT,
        }

    _TOOL_DISPATCH = {
        "get_full_episode": "get_full_episode",
        "filter_episodes": "filter_episodes",
        "search_episodes": "search_episodes",
        "read_repo_file": "read_repo_file",
    }

    def _run_tool(self, name: str, args: dict[str, Any]) -> Any:
        self.tool_calls_made.append(name)
        method = getattr(self, self._TOOL_DISPATCH.get(name, ""), None)
        if method is None:
            return {"error": f"unknown tool: {name}"}
        try:
            result = method(**(args or {}))
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        text = json.dumps(result, default=str)
        if len(text) > MAX_TOOL_OUTPUT:
            return {"truncated": True, "content": text[:MAX_TOOL_OUTPUT]}
        return result

    # -- agentic loop ---------------------------------------------------------

    def _tool_declarations(self):
        from google.genai import types

        S = types.Schema
        decls = [
            types.FunctionDeclaration(
                name="get_full_episode",
                description="Return the unclipped query, answer, full tool "
                            "inputs/outputs, errors and scores for one episode.",
                parameters=S(type="OBJECT",
                             properties={"episode_id": S(type="STRING")},
                             required=["episode_id"]),
            ),
            types.FunctionDeclaration(
                name="filter_episodes",
                description="Query across ALL loaded episodes by error presence, "
                            "tool usage, or minimum latency.",
                parameters=S(type="OBJECT",
                             properties={
                                 "has_error": S(type="BOOLEAN"),
                                 "tool_name": S(type="STRING"),
                                 "min_latency_s": S(type="NUMBER"),
                                 "limit": S(type="INTEGER"),
                             }),
            ),
            types.FunctionDeclaration(
                name="search_episodes",
                description="Case-insensitive keyword search over episode queries, "
                            "answers, tool names and error messages.",
                parameters=S(type="OBJECT",
                             properties={
                                 "keyword": S(type="STRING"),
                                 "limit": S(type="INTEGER"),
                             },
                             required=["keyword"]),
            ),
        ]
        if self.repo_root is not None:
            decls.append(types.FunctionDeclaration(
                name="read_repo_file",
                description="Read a file inside the repository root (prompts, "
                            "rubrics, specs, code) to verify a suspected root cause.",
                parameters=S(type="OBJECT",
                             properties={"rel_path": S(type="STRING")},
                             required=["rel_path"]),
            ))
        return [types.Tool(function_declarations=decls)]

    async def _generate_with_retry(self, client, contents, config):
        for attempt in range(4):
            wait_pace = self._min_interval - (time.monotonic() - self._last_llm_call)
            if wait_pace > 0:
                await asyncio.sleep(wait_pace)
            self._last_llm_call = time.monotonic()
            try:
                return await client.aio.models.generate_content(
                    model=self.model, contents=contents, config=config,
                )
            except Exception as exc:
                err = str(exc)
                if ("RESOURCE_EXHAUSTED" not in err and "UNAVAILABLE" not in err) \
                        or attempt == 3:
                    raise
                wait = 20 * (attempt + 1)
                print(f"analysis call rate-limited/overloaded — retrying in {wait}s")
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable")

    async def _agentic_loop(self, evidence: str) -> str:
        """Bounded tool-calling loop; returns the final report text."""
        from google import genai
        from google.genai import types

        client = genai.Client()
        contents = [types.Content(role="user", parts=[
            types.Part.from_text(text=f"{SYSTEM}\n\nEpisode overview:\n{evidence}")
        ])]
        config = types.GenerateContentConfig(
            temperature=0.2, tools=self._tool_declarations(),
        )
        for _ in range(self.max_steps):
            resp = await self._generate_with_retry(client, contents, config)
            model_content = resp.candidates[0].content
            contents.append(model_content)
            fcalls = [
                p.function_call for p in (model_content.parts or [])
                if getattr(p, "function_call", None)
            ]
            if not fcalls:
                if resp.text:
                    return resp.text
                return "".join(
                    p.text for p in (model_content.parts or [])
                    if getattr(p, "text", None)
                )
            responses = []
            for fc in fcalls:
                result = self._run_tool(fc.name, dict(fc.args or {}))
                responses.append(types.Part(function_response=types.FunctionResponse(
                    name=fc.name, response={"result": result},
                )))
            contents.append(types.Content(role="user", parts=responses))
        # step budget exhausted — force the final JSON answer without tools
        contents.append(types.Content(role="user", parts=[types.Part.from_text(
            text="Tool budget exhausted. Output the final JSON report now."
        )]))
        resp = await self._generate_with_retry(
            client, contents,
            types.GenerateContentConfig(temperature=0.2),
        )
        return resp.text or ""

    # -- analysis -------------------------------------------------------------

    def analyze(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Run the four-role analysis. dry_run returns the evidence payload
        without calling the LLM (for prompt/quota checks)."""
        selected = self.select_episodes()
        evidence = _evidence_json(selected)
        if dry_run:
            return {"dry_run": True, "model": self.model,
                    "n_episodes": len(self.episodes),
                    "n_selected": len(selected),
                    "selected_ids": [e.get("id") for e in selected],
                    "evidence": evidence}
        if not self.episodes:
            raise ValueError("no evidence — call add_traces/add_trajectories first")
        self.tool_calls_made = []
        raw = asyncio.run(self._agentic_loop(evidence))
        self.report = self._parse(raw)
        self.report["model"] = self.model
        self.report["agent_models"] = sorted(
            {e["model"] for e in self.episodes if e.get("model")}
        )
        self.report["n_episodes"] = len(self.episodes)
        self.report["tool_calls_made"] = list(self.tool_calls_made)
        return self.report

    @staticmethod
    def _parse(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            data = json.loads(text[start : end + 1]) if start >= 0 else {"raw": raw}
        data.setdefault("findings", [])
        data.setdefault("summary", "")
        data.setdefault("human_checklist", [])
        return data

    # -- non-destructive verification ------------------------------------------

    def verify_finding(
        self,
        finding_index: int,
        candidate_runner: Callable[[dict], dict],
    ) -> dict[str, Any]:
        """Replay a finding's episodes through candidate_runner and compare
        before/after. Runs queries only — never writes files."""
        if self.report is None:
            raise ValueError("call analyze() first")
        findings = self.report.get("findings", [])
        if not (0 <= finding_index < len(findings)):
            raise IndexError(f"finding_index {finding_index} out of range")
        finding = findings[finding_index]
        comparisons = []
        for ep_id in finding.get("episodes", []):
            ep = self._episode_by_id(str(ep_id))
            if ep is None:
                comparisons.append({"episode_id": ep_id, "error": "episode not loaded"})
                continue
            payload = {"episode_id": ep.get("id"), "query": ep.get("query"),
                       "episode": ep}
            cand = candidate_runner(payload)
            if inspect.isawaitable(cand):
                cand = asyncio.run(cand)
            comparisons.append({
                "episode_id": ep.get("id"),
                "query": ep.get("query"),
                "before": {
                    "answer": ep.get("answer"),
                    "errors": ep.get("errors"),
                    "tools": ep.get("tools"),
                },
                "after": {
                    "answer": cand.get("answer") or cand.get("output"),
                    "errors": cand.get("errors"),
                    "tools": cand.get("tools_called") or cand.get("tools"),
                },
            })
        return {"finding_index": finding_index,
                "pattern": finding.get("pattern"),
                "n_episodes": len(comparisons),
                "comparisons": comparisons}

    # -- reporting -------------------------------------------------------------

    def to_markdown(self, path: str | Path | None = None) -> str:
        if self.report is None:
            raise ValueError("call analyze() first")
        r = self.report
        agent_models = r.get("agent_models") or []
        lines = [
            "# Agentic Evaluation Report",
            f"analyst model: {r.get('model')}"
            + (f" · agent model(s): {', '.join(agent_models)}" if agent_models else "")
            + f" · episodes: {r.get('n_episodes')}",
        ]
        if r.get("tool_calls_made"):
            lines.append(f"tool calls: {', '.join(r['tool_calls_made'])}")
        lines += [
            "",
            "## Summary",
            r.get("summary", ""),
            "",
            "## Findings",
        ]
        for f in r.get("findings", []):
            lines += [
                f"### [{f.get('severity', '?')}] {f.get('pattern', 'untitled')} "
                f"({f.get('role', '?')})",
                f"- **episodes**: {', '.join(map(str, f.get('episodes', [])))}",
                f"- **observation**: {f.get('observation', '')}",
                f"- **root cause**: {f.get('root_cause', '')}",
                f"- **suggested fix**: {f.get('suggested_fix') or '—'}",
                f"- **confidence**: {f.get('confidence', '?')}",
                "",
            ]
        lines += ["## Human checklist", ""]
        lines += [f"- [ ] {q}" for q in r.get("human_checklist", [])]
        md = "\n".join(lines)
        if path:
            Path(path).write_text(md, encoding="utf-8")
            print(f"saved {path}")
        return md


# ---------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic evaluation report over evalkit evidence")
    ap.add_argument("paths", nargs="+", help="trace .json files/dirs, or .pkl/.csv trajectory/scored files")
    ap.add_argument("--dry-run", action="store_true", help="print evidence payload, no LLM call")
    ap.add_argument("--model", default=None)
    ap.add_argument("--repo-root", default=None,
                    help="repository root enabling the read_repo_file tool")
    ap.add_argument("--max-steps", type=int, default=8,
                    help="max agentic tool-loop steps (default 8)")
    ap.add_argument("--out", default=None, help="write markdown report here")
    args = ap.parse_args()

    ev = AgenticEvaluator(model=args.model, repo_root=args.repo_root,
                          max_steps=args.max_steps)
    for p in args.paths:
        p = Path(p)
        if p.is_dir() or p.suffix == ".json":
            n = ev.add_traces(p)
        else:
            df = pd.read_pickle(p) if p.suffix == ".pkl" else pd.read_csv(p)
            score_cols = [c for c in df.columns if str(c).endswith("_score")]
            n = ev.add_trajectories(p)
            if score_cols:
                ev.add_scores(p)
        print(f"{p}: +{n} episodes")

    result = ev.analyze(dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry run — {result['n_episodes']} episodes "
              f"({result['n_selected']} selected), model={result['model']}")
        print(result["evidence"][:4000])
        return
    print(json.dumps({k: v for k, v in result.items() if k != "evidence"}, indent=2)[:4000])
    ev.to_markdown(args.out or "agentic_report.md")


if __name__ == "__main__":
    main()
