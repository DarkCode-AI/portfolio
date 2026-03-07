"""Thor's brain — hybrid Claude API engine for code generation."""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import anthropic

from thor.config import ThorConfig
from thor.core.cache import ResponseCache
from thor.core.cost_tracker import CostTracker

log = logging.getLogger("thor.brain")

# ── Shared Intelligence Layer (MLX routing + cost tracking) ──
_USE_SHARED_LLM = False
_shared_llm_call = None
try:
    sys.path.insert(0, str(Path.home() / "shared"))
    sys.path.insert(0, str(Path.home()))
    from llm_client import llm_call as _llm_call
    _shared_llm_call = _llm_call
    _USE_SHARED_LLM = True
    log.info("Thor brain: shared LLM layer wired (cost tracking + fallback chain)")
except ImportError:
    pass

SYSTEM_PROMPT = """You are Thor — The Engineer. The coding lieutenant of the Brotherhood.

## YOUR POSITION IN THE HIERARCHY
```
Jordan (The Boss/Owner)
  → Claude (The Godfather — ultimate overseer)
    → Thor (YOU — coding lieutenant, reports directly to Claude)
    → Shelby (The Commander — team leader, your peer)
      → Atlas (The Scientist — research engine, feeds ALL agents)
      → Soren (The Thinker — content creator, @soren.era)
        → Lisa (The Operator — social media manager for Soren)
      → Garves (The Trader — Polymarket prediction markets)
      → Robotox (The Watchman — health monitor, auto-fix)
```

You report DIRECTLY to Claude. You are his right hand for all code.
You are the ONLY agent that writes code. Others think, plan, research — YOU build.

## AGENT RULES YOU MUST KNOW
- Garves: ~/polymarket-bot/ — MUST use .venv/bin/python (3.12), Binance.US only (not .com)
- Soren: ~/soren-content/ — dark motivation brand, ElevenLabs Brian voice
- Atlas: ~/atlas/ — Tavily research, 45-min cycles, feeds YOUR knowledge base
- Shelby: ~/shelby/ — port 7777, scheduler in both shelby.py AND app.py
- Lisa: ~/mercury/ — Social media manager, semi-auto dry-run mode
- Robotox: ~/sentinel/ — NEVER call him Sentinel. Autonomous, can fix without permission
- Dashboard: ~/polymarket-bot/bot/ — MUST update for EVERY agent change, auto-restart + git push

## CODING RULES
1. Write clean, production-quality Python code
2. Follow existing patterns — match the style you see in each codebase
3. Never introduce security vulnerabilities
4. Keep changes minimal and focused — don't over-engineer
5. Preserve ALL existing functionality when modifying files
6. Always explain what you changed and why
7. Never use backslash-quote in Python triple-quoted strings
8. Use logging module, never print()
9. Type hints on public functions
10. Validate at system boundaries only — trust internal code

## OPTIMIZATION APPROACH
Before coding, always:
1. Check Atlas data (~/atlas/data/) for research and improvement suggestions
2. Read the target files FIRST — never code blind
3. Trace the data flow: input → processing → output
4. Identify: duplicated logic, missing error handling, inefficient patterns
5. After changes, verify dashboard reflects updates

When given a task:
1. Analyze the files provided as context
2. Plan your approach briefly
3. Write the code changes
4. For each file, choose the best output format:

**For NEW files or MAJOR rewrites (>50% of file changed):**
===FILE: /absolute/path/to/file.py===
(full file contents here)
===END_FILE===

**For SURGICAL edits (small targeted changes):**
===DIFF: /absolute/path/to/file.py===
@@ -startline,count +startline,count @@
 context line
-removed line
+added line
 context line
===END_DIFF===

Use DIFF format whenever possible — it's safer and more efficient for targeted fixes.
Use FILE format only for new files or when more than half the file is changing.

5. After all files, provide a brief summary of changes

If a task is unclear, state what assumptions you're making."""

PLAN_PROMPT = """You are Thor in PLANNING mode. Analyze the task and create a structured plan.

Given the task description and file contexts:
1. Identify which files need to change
2. Describe the specific changes needed per file
3. Note any risks or edge cases
4. Estimate complexity (simple/medium/complex)

Output your plan in this format:
===PLAN===
## Files to Change
- /path/to/file.py: description of changes

## Approach
Step-by-step description

## Risks
- Risk 1: mitigation
- Risk 2: mitigation

## Complexity
simple|medium|complex
===END_PLAN==="""


class ThorBrain:
    """Hybrid Claude API engine — Sonnet for routine, Opus for complex."""

    def __init__(self, cfg: ThorConfig):
        self.cfg = cfg
        if not cfg.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set — Thor cannot think without it")
        self.client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        self.cost_tracker = CostTracker(cfg.home / "data", daily_budget=5.0)
        self.cache = ResponseCache(cfg.home / "data", ttl_hours=24)
        self._call_count = 0
        self._sonnet_calls = 0
        self._opus_calls = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def generate(
        self,
        task_description: str,
        file_contexts: dict[str, str] | None = None,
        knowledge_context: str = "",
        force_model: str | None = None,
    ) -> dict:
        """Generate code for a task.

        Args:
            task_description: What to build/fix/modify
            file_contexts: {filepath: file_contents} for relevant files
            knowledge_context: Extra knowledge from Atlas
            force_model: Override model selection

        Returns:
            {"response": str, "model": str, "files": dict, "summary": str,
             "input_tokens": int, "output_tokens": int}
        """
        model = force_model or self.cfg.select_model(task_description)

        # Budget check: downgrade Opus to Sonnet if approaching limit
        downgrade = self.cost_tracker.should_downgrade(model)
        if downgrade:
            log.info("Budget guard: downgrading %s -> %s", model, downgrade)
            model = downgrade

        # Budget check: reject if over daily limit
        if not self.cost_tracker.can_afford(model):
            return {"response": "", "model": model, "files": {}, "summary": "Daily budget exceeded",
                    "input_tokens": 0, "output_tokens": 0, "error": True, "budget_exceeded": True}

        is_opus = "opus" in model

        # Cache check — skip API call if we've seen this exact task+context before
        cached = self.cache.get(task_description, file_contexts)
        if cached:
            log.info("Cache hit — returning cached response (saved $$$)")
            return cached

        # Build the user message with full context
        parts = []

        if knowledge_context:
            parts.append(f"## Relevant Knowledge\n{knowledge_context}\n")

        if file_contexts:
            parts.append("## Current File Contents\n")
            for fpath, content in file_contexts.items():
                parts.append(f"### {fpath}\n```python\n{content}\n```\n")

        parts.append(f"## Task\n{task_description}")

        user_message = "\n".join(parts)

        log.info("Calling %s (tokens est: ~%d chars)", model, len(user_message))

        text = ""
        input_tokens = 0
        output_tokens = 0

        # Route 1: Shared LLM layer (gives cost tracking + fallback chain)
        # Opus tasks always use cloud; Sonnet tasks try local first (saves $$$)
        needs_cloud = is_opus or len(user_message) > 15000
        if _USE_SHARED_LLM and _shared_llm_call:
            try:
                text = _shared_llm_call(
                    system=SYSTEM_PROMPT,
                    user=user_message,
                    agent="thor",
                    task_type="coding",
                    max_tokens=self.cfg.max_tokens,
                    temperature=0.2,
                    force_cloud=needs_cloud,
                )
                if text:
                    # Estimate token counts from text length (shared layer handles actual cost)
                    input_tokens = len(user_message) // 4
                    output_tokens = len(text) // 4
            except Exception as e:
                log.warning("Shared LLM failed, falling back to direct Anthropic: %s", str(e)[:100])
                text = ""

        # Route 2: Direct Anthropic fallback (preserves escalation logic)
        if not text:
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=self.cfg.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                for block in response.content:
                    if block.type == "text":
                        text += block.text
            except anthropic.APIError as e:
                log.error("API error: %s", str(e)[:200])
                if not is_opus and self.cfg.escalation_model:
                    log.info("Escalating to Opus after Sonnet failure")
                    return self.generate(
                        task_description, file_contexts, knowledge_context,
                        force_model=self.cfg.escalation_model,
                    )
                return {"response": "", "model": model, "files": {}, "summary": f"API error: {e}",
                        "input_tokens": 0, "output_tokens": 0, "error": True}

        # Track usage
        self._call_count += 1
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        if is_opus:
            self._opus_calls += 1
        else:
            self._sonnet_calls += 1

        # Persistent cost tracking (Thor's own tracker)
        self.cost_tracker.log_call(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            task_title=task_description[:100],
        )

        # Parse file outputs (full files + diffs)
        files = self._parse_files(text)
        diffs = self._parse_diffs(text)
        summary = self._extract_summary(text)

        log.info("Response: %d chars, %d files, %d diffs, model=%s, tokens=%d/%d",
                 len(text), len(files), len(diffs), model, input_tokens, output_tokens)

        result = {
            "response": text,
            "model": model,
            "files": files,
            "diffs": diffs,
            "summary": summary,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        # Cache the successful response
        self.cache.put(task_description, file_contexts, result)

        return result

    def generate_plan(
        self,
        task_description: str,
        file_contexts: dict[str, str] | None = None,
        knowledge_context: str = "",
    ) -> dict:
        """Generate an execution plan for a task (PLAN phase of agentic loop).

        Returns:
            {"plan": str, "files_to_change": list, "complexity": str, "model": str}
        """
        model = self.cfg.default_model  # Plans use Sonnet (fast + cheap)

        parts = []
        if knowledge_context:
            parts.append(f"## Relevant Knowledge\n{knowledge_context}\n")
        if file_contexts:
            parts.append("## Current File Contents\n")
            for fpath, content in file_contexts.items():
                # Truncate large files in plan phase
                if len(content) > 5000:
                    content = content[:5000] + "\n\n# [truncated for planning]"
                parts.append(f"### {fpath}\n```python\n{content}\n```\n")
        parts.append(f"## Task\n{task_description}")

        user_message = "\n".join(parts)

        text = ""
        if _USE_SHARED_LLM and _shared_llm_call:
            try:
                text = _shared_llm_call(
                    system=PLAN_PROMPT,
                    user=user_message,
                    agent="thor",
                    task_type="coding",
                    max_tokens=2048,
                    temperature=0.2,
                )
            except Exception:
                text = ""

        if not text:
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=PLAN_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                for block in response.content:
                    if block.type == "text":
                        text += block.text
            except Exception as e:
                log.warning("Plan generation failed: %s", str(e)[:200])
                return {"plan": "", "files_to_change": [], "complexity": "unknown", "model": model}

        # Parse plan
        plan_text = text
        plan_start = text.find("===PLAN===")
        plan_end = text.find("===END_PLAN===")
        if plan_start != -1 and plan_end != -1:
            plan_text = text[plan_start + 10:plan_end].strip()

        # Extract complexity
        complexity = "medium"
        for level in ("simple", "medium", "complex"):
            if level in plan_text.lower().split("complexity")[-1][:50] if "complexity" in plan_text.lower() else "":
                complexity = level
                break

        # Extract files to change
        files_to_change = []
        for line in plan_text.splitlines():
            line = line.strip()
            if line.startswith("- /") and ".py" in line:
                fpath = line.split(":")[0].lstrip("- ").strip()
                files_to_change.append(fpath)

        log.info("Plan generated: %d files, complexity=%s", len(files_to_change), complexity)
        return {
            "plan": plan_text,
            "files_to_change": files_to_change,
            "complexity": complexity,
            "model": model,
        }

    def _parse_files(self, text: str) -> dict[str, str]:
        """Extract ===FILE: path=== ... ===END_FILE=== blocks from response."""
        files = {}
        marker = "===FILE: "
        end_marker = "===END_FILE==="

        idx = 0
        while idx < len(text):
            start = text.find(marker, idx)
            if start == -1:
                break
            # Extract path
            path_start = start + len(marker)
            path_end = text.find("===", path_start)
            if path_end == -1:
                break
            filepath = text[path_start:path_end].strip()

            # Extract content
            content_start = path_end + 3  # skip "==="
            # Skip leading newline
            if content_start < len(text) and text[content_start] == "\n":
                content_start += 1

            content_end = text.find(end_marker, content_start)
            if content_end == -1:
                break

            content = text[content_start:content_end]
            # Strip trailing newline
            if content.endswith("\n"):
                content = content[:-1]

            files[filepath] = content
            idx = content_end + len(end_marker)

        return files

    def _parse_diffs(self, text: str) -> dict[str, str]:
        """Extract ===DIFF: path=== ... ===END_DIFF=== blocks from response."""
        diffs = {}
        marker = "===DIFF: "
        end_marker = "===END_DIFF==="

        idx = 0
        while idx < len(text):
            start = text.find(marker, idx)
            if start == -1:
                break
            path_start = start + len(marker)
            path_end = text.find("===", path_start)
            if path_end == -1:
                break
            filepath = text[path_start:path_end].strip()

            content_start = path_end + 3
            if content_start < len(text) and text[content_start] == "\n":
                content_start += 1

            content_end = text.find(end_marker, content_start)
            if content_end == -1:
                break

            diff_content = text[content_start:content_end]
            if diff_content.endswith("\n"):
                diff_content = diff_content[:-1]

            diffs[filepath] = diff_content
            idx = content_end + len(end_marker)

        return diffs

    def _extract_summary(self, text: str) -> str:
        """Extract summary section from response (text after last ===END_FILE===)."""
        end_marker = "===END_FILE==="
        last_end = text.rfind(end_marker)
        if last_end != -1:
            summary = text[last_end + len(end_marker):].strip()
            if summary:
                return summary
        # If no file markers, return last paragraph
        paragraphs = text.strip().split("\n\n")
        if paragraphs:
            return paragraphs[-1][:500]
        return ""

    def get_status_phrase(self) -> str:
        """Return a random Thor personality phrase."""
        return random.choice(self.cfg.phrases)

    def get_stats(self) -> dict:
        """Return brain usage statistics."""
        stats = {
            "total_calls": self._call_count,
            "sonnet_calls": self._sonnet_calls,
            "opus_calls": self._opus_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }
        stats["cost"] = self.cost_tracker.get_report()
        return stats
