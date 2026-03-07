"""Thor — The Engineer. Main agent loop."""
from __future__ import annotations

import json
import logging
import signal
import sys
import time
from pathlib import Path

from thor.config import ThorConfig
from thor.core.brain import ThorBrain
from thor.core.coder import Coder
from thor.core.knowledge import KnowledgeBase
from thor.core.reporter import Reporter
from thor.core.task_queue import CodingTask, TaskQueue, TaskResult
from thor.core.reflexion import ReflexionMemory
from thor.core.progress import ProgressTracker
from thor.core.codebase_index import CodebaseIndex
from thor.core.tracking import track_task_completion
from thor.core.budget import ContextBudget
from thor.core.task_memory import TaskMemory
from thor.core.git_ops import GitOps
from thor.core.autofix import AutoFixPipeline

log = logging.getLogger("thor")


class ThorAgent:
    """Thor — The Engineer. Autonomous coding lieutenant."""

    def __init__(self, cfg: ThorConfig | None = None):
        self.cfg = cfg or ThorConfig()
        self._setup_logging()

        log.info("=" * 50)
        log.info("THOR — The Engineer")
        log.info("Methodical. Blueprints first. Clean execution.")
        log.info("=" * 50)

        self.brain = ThorBrain(self.cfg)
        self.coder = Coder()
        self.queue = TaskQueue(self.cfg.tasks_dir, self.cfg.results_dir)
        self.knowledge = KnowledgeBase(self.cfg.knowledge_dir)
        self.reporter = Reporter(self.cfg.home / "data")
        self.reflexion = ReflexionMemory(self.cfg.home / "data")
        self.progress = ProgressTracker(self.cfg.home / "data")
        self.codebase_index = CodebaseIndex(self.cfg.home / "data")
        self.task_memory = TaskMemory()
        self.git = GitOps()
        self.autofix = AutoFixPipeline(self.cfg.tasks_dir)
        self._running = False
        self._agentic_max_iterations = 3
        self._start_time = None

        # Agent Hub
        self._hub = None
        try:
            sys.path.insert(0, str(Path.home() / ".agent-hub"))
            from hub import AgentHub
            self._hub = AgentHub("thor")
            self._hub.register(capabilities=[
                "coding", "file_operations", "testing",
                "refactoring", "debugging", "code_review",
            ])
            log.info("Agent Hub: connected")
        except Exception:
            log.debug("Agent Hub: not available")

        # Shared Intelligence Layer — AgentBrain for decision memory
        self._brain = None
        try:
            sys.path.insert(0, str(Path.home() / "shared"))
            sys.path.insert(0, str(Path.home()))
            from agent_brain import AgentBrain
            self._brain = AgentBrain(
                "thor",
                system_prompt="You are Thor, the coding engineer of the Brotherhood.",
                task_type="coding",
            )
            log.info("Shared brain: wired (memory + learning)")
        except Exception:
            log.debug("Shared brain: not available")

    def _setup_logging(self) -> None:
        """Configure logging to file and console."""
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        log_datefmt = "%H:%M:%S"

        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(log_format, log_datefmt))

        # File handler
        file_handler = logging.FileHandler(self.cfg.log_file, mode="a")
        file_handler.setFormatter(logging.Formatter(log_format, log_datefmt))

        root = logging.getLogger("thor")
        root.setLevel(getattr(logging, self.cfg.log_level.upper(), logging.INFO))
        root.addHandler(console)
        root.addHandler(file_handler)

    def run_batch(self) -> int:
        """Batch mode: process ALL pending tasks once, then exit.

        Used by 12h auto-wake schedule and dashboard Wake button.
        Returns number of tasks processed.
        """
        log.info("=" * 50)
        log.info("THOR — Batch Mode (wake, process, sleep)")
        log.info("=" * 50)

        self.reporter.update_status(state="batch")

        # Ingest latest Atlas knowledge
        try:
            count = self.knowledge.ingest_atlas_data()
            if count:
                log.info("Ingested %d knowledge entries from Atlas", count)
        except Exception:
            pass

        # Process all pending tasks
        processed = 0
        while True:
            tasks = self.queue.get_pending()
            if not tasks:
                break
            task = tasks[0]
            log.info("Batch: processing task %d/%d — %s",
                     processed + 1, len(tasks) + processed, task.title)
            try:
                self._process_task(task)
                processed += 1
            except Exception as e:
                log.error("Batch task error: %s", str(e)[:300])
                # Mark as failed so we don't loop forever
                task.status = "failed"
                task.error = str(e)[:500]
                task.completed_at = time.time()
                self.queue.update_task(task)
                processed += 1

            # Budget check: stop if daily budget exhausted
            if not self.brain.cost_tracker.can_afford(
                self.cfg.select_model("generic task")):
                log.warning("Daily budget exhausted — stopping batch")
                break

        self.reporter.update_status(state="sleeping")
        log.info("Thor batch complete: %d tasks processed. Going back to sleep.", processed)
        return processed

    def run(self) -> None:
        """Main loop: poll for tasks, process them, report results."""
        self._running = True
        self._start_time = time.time()
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self.reporter.update_status(state="starting")
        log.info("Thor online. Polling for tasks every %ds.", self.cfg.poll_interval_s)

        # Initial Atlas knowledge ingestion
        try:
            count = self.knowledge.ingest_atlas_data()
            if count:
                log.info("Ingested %d knowledge entries from Atlas", count)
        except Exception as e:
            log.warning("Atlas ingestion failed: %s", e)

        # Build codebase index if stale
        try:
            if self.codebase_index.is_stale(max_age_hours=6):
                log.info("Codebase index is stale, rebuilding...")
                stats = self.codebase_index.build()
                log.info("Codebase index: %d files, %d functions, %d classes",
                         stats["total_files"], stats["total_functions"], stats["total_classes"])
        except Exception as e:
            log.warning("Codebase index build failed: %s", e)

        self.reporter.update_status(state="idle", queue_depth=0)

        while self._running:
            try:
                self._tick()
            except Exception as e:
                log.error("Tick error: %s", str(e)[:300])
                self.reporter.log_activity("error", details=str(e)[:200], success=False)

            time.sleep(self.cfg.poll_interval_s)

        self.reporter.update_status(state="offline")
        log.info("Thor offline. Shutdown complete.")

    def _handle_shutdown(self, signum, frame):
        log.info("Shutdown signal received.")
        self._running = False

    def _tick(self) -> None:
        """Single tick: check for pending tasks, process the next one."""
        # Heartbeat every tick (keeps Thor from showing DEAD during idle)
        if self._hub:
            try:
                stats = self.queue.get_stats()
                self._hub.heartbeat(
                    status="idle" if stats.get("pending", 0) == 0 else "working",
                    metrics={
                        "tasks_completed": stats.get("completed", 0),
                        "pending": stats.get("pending", 0),
                    },
                )
            except Exception:
                pass

        # Check broadcasts
        self._check_broadcasts()

        # Read brain notes from dashboard (only log new ones)
        try:
            sys.path.insert(0, str(Path.home() / "polymarket-bot"))
            from bot.brain_reader import read_brain_notes
            brain_notes = read_brain_notes("thor")
            if brain_notes:
                if not hasattr(self, '_seen_brain_count'):
                    self._seen_brain_count = 0
                if len(brain_notes) > self._seen_brain_count:
                    for note in brain_notes[self._seen_brain_count:]:
                        log.info("[BRAIN:%s] %s: %s", note.get("type", "note").upper(), note.get("topic", "?"), note.get("content", "")[:120])
                    self._seen_brain_count = len(brain_notes)
        except Exception:
            pass

        # Check for Robotox auto-fix events
        try:
            fix_tasks = self.autofix.process_all()
            for fix_data in fix_tasks:
                task = CodingTask(
                    title=fix_data["title"],
                    description=fix_data["description"],
                    target_files=fix_data.get("target_files", []),
                    agent=fix_data.get("agent", ""),
                    priority=fix_data.get("priority", "normal"),
                    assigned_by=fix_data.get("assigned_by", "robotox"),
                )
                self.queue.submit(task)
                log.info("Auto-fix task queued: %s", task.title[:80])
        except Exception as e:
            log.debug("Autofix check error: %s", e)

        # Get pending tasks
        tasks = self.queue.get_pending()
        if not tasks:
            # Idle — run proactive suggestions every 30 min
            self._maybe_suggest()
            return

        stats = self.queue.get_stats()
        log.info("Queue: %d pending, %d in progress", stats["pending"], stats["in_progress"])
        self.reporter.update_status(
            state="processing",
            queue_depth=stats["pending"],
        )

        # Process next task
        task = tasks[0]
        self._process_task(task)

    def _process_task(self, task: CodingTask) -> None:
        """Process a single coding task using the multi-turn agentic loop.

        Flow: PLAN → IMPLEMENT → VALIDATE → REFINE (up to N iterations).
        """
        log.info("=" * 40)
        log.info("TASK: %s", task.title)
        log.info("Agent: %s | Priority: %s | By: %s", task.agent, task.priority, task.assigned_by)
        log.info("=" * 40)

        # Mark in progress + start progress tracking
        task.status = "in_progress"
        task.started_at = time.time()
        self.queue.update_task(task)
        self.progress.start_task(task.id, task.title, task.agent or "", task.description)
        self.reporter.update_status(
            state="coding",
            current_task=task.title,
            model=self.cfg.select_model(task.description, task.target_files),
        )

        # ── Phase 0: Gather Context ──
        all_files, file_contexts, knowledge_ctx = self._gather_context(task)

        # ── Phase 0.5: Check Task Memory for similar past tasks ──
        try:
            similar = self.task_memory.find_similar(task.description, task.agent)
            memory_ctx = self.task_memory.format_context(similar)
            if memory_ctx:
                knowledge_ctx = memory_ctx + "\n" + knowledge_ctx
                log.info("Task memory: found %d similar past tasks", len(similar))
        except Exception:
            pass

        # ── Phase 1: PLAN ──
        self.progress.add_step(task.id, "PLAN phase: analyzing task", "in_progress")
        plan_result = self.brain.generate_plan(
            task_description=task.description,
            file_contexts=file_contexts,
            knowledge_context=knowledge_ctx,
        )
        plan_text = plan_result.get("plan", "")
        if plan_text:
            log.info("PLAN: complexity=%s, files=%d",
                     plan_result.get("complexity", "?"), len(plan_result.get("files_to_change", [])))
            self.progress.add_step(task.id, f"Plan: {plan_result.get('complexity', '?')} complexity, {len(plan_result.get('files_to_change', []))} files")

            # Read additional files identified by plan
            for fpath in plan_result.get("files_to_change", []):
                if fpath not in file_contexts:
                    p = Path(fpath).expanduser()
                    if p.exists() and p.stat().st_size < 500_000:
                        try:
                            file_contexts[fpath] = p.read_text()
                            all_files.append(fpath)
                        except Exception:
                            pass

        # ── Context Budget Management ──
        model = task.force_model or self.cfg.select_model(task.description, task.target_files)
        budget = ContextBudget(model)
        task_desc, knowledge_ctx, file_contexts = budget.fit_to_budget(
            task.description, knowledge_ctx, file_contexts,
        )

        # ── Phase 2-4: IMPLEMENT → VALIDATE → REFINE loop ──
        iteration = 0
        last_error = ""
        written = []
        summary = ""
        model_used = ""
        test_passed = None
        test_output = ""
        total_input_tokens = 0
        total_output_tokens = 0

        while iteration < self._agentic_max_iterations:
            iteration += 1
            phase_label = "IMPLEMENT" if iteration == 1 else f"REFINE #{iteration - 1}"
            log.info("── %s (iteration %d/%d) ──", phase_label, iteration, self._agentic_max_iterations)
            self.progress.add_step(task.id, f"{phase_label}: generating code", "in_progress")

            # Build task description with plan + previous error feedback
            enriched_desc = task_desc
            if plan_text:
                enriched_desc = f"## Execution Plan\n{plan_text}\n\n## Task\n{enriched_desc}"
            if last_error:
                enriched_desc += f"\n\n## Previous Attempt Failed\n{last_error}\nFix the issue and try again."

            # ── IMPLEMENT: Call the brain ──
            force_model = task.force_model or None
            if iteration > 1 and not force_model:
                force_model = self.cfg.escalation_model  # Escalate on retries

            if getattr(task, "tdd", False) and iteration == 1:
                result = self._run_tdd(task, enriched_desc, file_contexts, knowledge_ctx, force_model)
            else:
                result = self.brain.generate(
                    task_description=enriched_desc,
                    file_contexts=file_contexts,
                    knowledge_context=knowledge_ctx,
                    force_model=force_model,
                )

            files_to_write = result.get("files", {})
            diffs_to_apply = result.get("diffs", {})
            summary = result.get("summary", "")
            model_used = result.get("model", "")
            total_input_tokens += result.get("input_tokens", 0)
            total_output_tokens += result.get("output_tokens", 0)
            self.progress.record_model(task.id, model_used)
            self.progress.add_step(task.id, f"Brain: {len(files_to_write)} files, {len(diffs_to_apply)} diffs ({model_used})")

            # API error
            if result.get("error"):
                log.error("Brain error: %s", summary[:200])
                if iteration < self._agentic_max_iterations:
                    last_error = f"API error: {summary[:200]}"
                    continue
                self._fail_task(task, summary[:200], total_input_tokens + total_output_tokens)
                return

            # Empty response
            if not files_to_write and not diffs_to_apply and not summary:
                log.warning("Brain returned empty response")
                if iteration < self._agentic_max_iterations:
                    last_error = "Empty response from brain — be more specific"
                    continue
                self._fail_task(task, "Empty response after all iterations", total_input_tokens + total_output_tokens)
                return

            # ── VALIDATE: Syntax + Quality Gate ──
            validation_error = self._validate_code(task, files_to_write)
            if validation_error:
                log.warning("Validation failed: %s", validation_error[:200])
                if iteration < self._agentic_max_iterations:
                    last_error = validation_error
                    self.progress.record_retry(task.id, f"Validation: {validation_error[:100]}")
                    continue
                # Last iteration — proceed with caution
                log.warning("Proceeding despite validation issues (final iteration)")

            # ── Write files + Apply diffs ──
            written = []
            if files_to_write:
                written.extend(self.coder.write_files(files_to_write))
            if diffs_to_apply:
                written.extend(self.coder.apply_diffs(diffs_to_apply))

            if written:
                log.info("Written %d files: %s", len(written), ", ".join(written))
                self.progress.record_files_written(task.id, written)
                self.progress.add_step(task.id, f"Wrote {len(written)} files")

            # Auto-execute scripts
            for fpath in written:
                if "/scripts/" in fpath or fpath.endswith("_generator.py"):
                    log.info("Auto-executing: %s", fpath)
                    ok, output = self.coder.run_script(fpath, timeout=60)
                    if ok:
                        log.info("Script output: %s", output[:300])

            # ── Run tests ──
            test_passed, test_output = self._run_tests(task, written)
            if test_passed is False and iteration < self._agentic_max_iterations:
                log.warning("Tests failed, entering REFINE phase")
                self.coder.rollback_last()
                reflection = self.reflexion.generate_reflection("test", test_output[:300], task.description)
                self.reflexion.add_reflection(
                    task.id, task.title, task.agent or "", iteration,
                    "test", test_output[:300], reflection, written,
                )
                last_error = f"Tests failed:\n{test_output[:500]}\nReflection: {reflection}"
                self.progress.record_retry(task.id, f"Test failed: {test_output[:100]}")
                continue

            # ── All validations passed — break out of loop ──
            log.info("Iteration %d: all checks passed", iteration)
            break

        # ── Phase 5: Complete + Post-task ──
        self._complete_task(
            task, written, summary, model_used, test_passed, test_output,
            total_input_tokens, total_output_tokens, iteration,
        )

    def _gather_context(self, task: CodingTask) -> tuple[list[str], dict[str, str], str]:
        """Phase 0: Read files, knowledge, reflections, codebase index."""
        all_files = list(set(task.target_files + task.context_files))
        resolved_files = []
        for f in all_files:
            p = Path(f).expanduser()
            if p.exists():
                resolved_files.append(str(p))
            else:
                found = False
                agent_key = (task.agent or "").lower().strip()
                search_dirs = []
                if agent_key and agent_key in self.cfg.project_dirs:
                    search_dirs.append(self.cfg.project_dirs[agent_key])
                for d in self.cfg.project_dirs.values():
                    if d not in search_dirs:
                        search_dirs.append(d)
                for proj_dir in search_dirs:
                    candidate = Path(proj_dir) / f
                    if candidate.exists():
                        resolved_files.append(str(candidate))
                        found = True
                        break
                if not found:
                    log.warning("File not found: %s", f)
        all_files = resolved_files

        if not all_files and task.agent and task.agent.lower() in self.cfg.project_dirs:
            project_dir = self.cfg.project_dirs[task.agent.lower()]
            discovered = self.coder.discover_project_files(project_dir)
            discovered.sort(key=lambda f: len(f))
            all_files = discovered[:10]
            log.info("Auto-discovered %d files for '%s'", len(all_files), task.agent)

        file_contexts = self.coder.read_files(all_files) if all_files else {}
        log.info("Read %d files for context", len(file_contexts))
        self.progress.record_files_read(task.id, all_files)
        self.progress.add_step(task.id, f"Read {len(file_contexts)} files for context")

        knowledge_ctx = self.knowledge.get_context_for_task(task.description, task.agent)

        relevant_reflections = self.reflexion.get_relevant_reflections(task.description, task.agent)
        reflection_ctx = self.reflexion.format_context(relevant_reflections)
        if reflection_ctx:
            knowledge_ctx = reflection_ctx + "\n" + knowledge_ctx
            log.info("Added %d reflections to context", len(relevant_reflections))

        try:
            index_ctx = self.codebase_index.format_context_for_brain(all_files)
            if index_ctx:
                knowledge_ctx = index_ctx + "\n" + knowledge_ctx
        except Exception:
            pass

        return all_files, file_contexts, knowledge_ctx

    def _validate_code(self, task: CodingTask, files_to_write: dict[str, str]) -> str:
        """Validate syntax + quality gate. Returns error string or empty if OK."""
        if not files_to_write:
            return ""

        # Syntax check
        syntax_errors = self.coder.validate_syntax(files_to_write)
        bad_files = {f: e for f, e in syntax_errors.items() if e}
        if bad_files:
            error_detail = "; ".join(f"{f}: {e}" for f, e in bad_files.items())
            reflection = self.reflexion.generate_reflection("syntax", error_detail, task.description)
            self.reflexion.add_reflection(
                task.id, task.title, task.agent or "", task.retries + 1,
                "syntax", error_detail, reflection, list(bad_files.keys()),
            )
            return f"Syntax errors: {error_detail}\nReflection: {reflection}"

        # Quality gate
        qg = self.coder.quality_gate(files_to_write)
        if not qg["passed"]:
            issues = []
            for f, fr in qg.get("files", {}).items():
                for s in fr.get("security", []):
                    issues.append(f"  {f}: {s}")
                for lint_issue in fr.get("lint", [])[:3]:
                    issues.append(f"  {f}: {lint_issue}")
            feedback = "\n".join(issues[:10])
            error_detail = f"score={qg['score']}, security={qg['total_security_issues']}"
            reflection = self.reflexion.generate_reflection("quality_gate", feedback, task.description)
            self.reflexion.add_reflection(
                task.id, task.title, task.agent or "", task.retries + 1,
                "quality_gate", error_detail, reflection, list(qg.get("files", {}).keys()),
            )
            return f"Quality gate failed (score={qg['score']}):\n{feedback}\nReflection: {reflection}"

        # AI code review (first iteration only, skip on retries to save cost)
        if task.retries == 0:
            try:
                from thor.core.reviewer import CodeReviewer
                reviewer = CodeReviewer(self.cfg.home / "data")
                review = reviewer.review(task.description, files_to_write)
                self.progress.add_step(task.id, f"Review: {review['verdict']} (score={review.get('score', '?')})")
                if review["verdict"] == "FAIL":
                    feedback = reviewer.format_feedback(review)
                    return f"Code review failed (score={review.get('score', 0)}):\n{feedback}"
            except Exception:
                pass

        return ""

    def _run_tests(self, task: CodingTask, written: list[str]) -> tuple[bool | None, str]:
        """Run tests on written files. Returns (passed, output)."""
        test_cmd = task.test_command
        if not test_cmd and written:
            test_cmd = self._auto_detect_tests(written)
        if test_cmd:
            log.info("Running tests: %s", test_cmd[:120])
            return self.coder.run_test(test_cmd, self.cfg.test_timeout_s)

        # No tests — syntax check only
        test_passed = None
        test_output = ""
        if written:
            for fpath in written:
                if fpath.endswith(".py"):
                    ok, out = self.coder.run_test(
                        f"{sys.executable} -c \"import py_compile; py_compile.compile('{fpath}', doraise=True)\"",
                        timeout=10,
                    )
                    if not ok:
                        test_passed = False
                        test_output += f"{fpath}: {out[:200]}\n"
            if test_passed is None:
                test_passed = True
                test_output = "Syntax check passed (no test suite found)"
        return test_passed, test_output

    def _run_tdd(self, task, enriched_desc, file_contexts, knowledge_ctx, force_model):
        """TDD mode: generate tests first, then implementation."""
        log.info("TDD MODE: generating tests first")
        self.progress.add_step(task.id, "TDD: Generating test cases", "in_progress")
        from thor.core.tdd import TDDRunner
        tdd = TDDRunner(self.brain, self.coder)

        test_result = tdd.generate_tests(
            task_description=enriched_desc,
            target_files=list(file_contexts.keys()),
            file_contexts=file_contexts,
            knowledge_context=knowledge_ctx,
            force_model=force_model,
        )

        if test_result.get("test_code"):
            self.coder.write_files({test_result["test_file"]: test_result["test_code"]})
            self.progress.add_step(task.id, f"TDD: Tests generated ({test_result['test_file']})")
            if not task.test_command:
                task.test_command = f"python3 -m pytest {test_result['test_file']} -v --tb=short"

            return tdd.generate_implementation(
                task_description=enriched_desc,
                test_code=test_result["test_code"],
                test_file=test_result["test_file"],
                file_contexts=file_contexts,
                knowledge_context=knowledge_ctx,
                force_model=force_model,
            )

        log.warning("TDD: failed to generate tests, falling back to normal")
        return self.brain.generate(
            task_description=enriched_desc,
            file_contexts=file_contexts,
            knowledge_context=knowledge_ctx,
            force_model=force_model,
        )

    def _complete_task(
        self,
        task: CodingTask,
        written: list[str],
        summary: str,
        model_used: str,
        test_passed: bool | None,
        test_output: str,
        input_tokens: int,
        output_tokens: int,
        iterations: int,
    ) -> None:
        """Phase 5: Save result, update tracking, notify, remember."""
        phrase = self.brain.get_status_phrase()
        task_result = TaskResult(
            task_id=task.id,
            files_written={f: "" for f in written},  # Don't store full content in result
            summary=summary,
            model_used=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            test_passed=test_passed,
            test_output=test_output[:1000],
            phrase=phrase,
        )
        result_id = self.queue.save_result(task_result)

        task.status = "completed"
        task.completed_at = time.time()
        task.result_id = result_id
        self.queue.update_task(task)
        self.progress.complete_task(task.id, summary[:500])

        elapsed = task.completed_at - task.started_at
        log.info("COMPLETE: %s (%.1fs, %s, %d iterations)", task.title, elapsed, model_used, iterations)
        log.info("Files: %s", ", ".join(written) or "none")
        log.info("Thor says: \"%s\"", phrase)

        self.reporter.log_activity(
            "task_completed",
            task_id=task.id,
            details=f"{task.title}: {summary[:200]}",
            files=written,
            model=model_used,
            tokens=input_tokens + output_tokens,
            success=True,
        )
        self.reporter.update_status(state="idle", queue_depth=self.queue.get_stats()["pending"])

        # Publish to shared event bus
        try:
            sys.path.insert(0, str(Path.home()))
            from shared.events import publish as bus_publish
            bus_publish(
                agent="thor",
                event_type="task_completed",
                data={
                    "task_id": task.id, "title": task.title, "model": model_used,
                    "files": written, "elapsed": round(elapsed, 1),
                    "target_agent": task.agent or "", "iterations": iterations,
                },
                severity="info",
                summary=f"Completed: {task.title} ({model_used}, {elapsed:.1f}s, {iterations} iter)",
            )
        except Exception:
            pass

        # Robotox feedback: publish fix result if task was from robotox
        if task.assigned_by == "robotox":
            try:
                from core.autofix import AutoFixPipeline
                AutoFixPipeline.publish_result(
                    task_id=task.id,
                    agent=task.agent or "",
                    title=task.title,
                    success=test_passed is not False,
                    summary=summary[:500] if summary else "",
                    files=written,
                )
            except Exception as e:
                log.debug("Robotox feedback publish failed: %s", e)

        # Post-task tracking — Excel, dashboard suggestions
        try:
            agent_name = (task.agent or "Thor").title() if task.agent else "Thor"
            track_task_completion(
                task_id=task.id,
                agent=agent_name,
                category="Improvement",
                feature=task.title[:100],
                description=summary[:500] if summary else task.description[:500],
                files_changed=written,
                status="Complete",
                metadata={"assigned_by": task.assigned_by, "model": model_used, "iterations": iterations},
            )
            log.info("Excel tracker updated")
        except Exception as e:
            log.warning("Post-task tracking failed: %s", e)

        # Task memory — remember this task for future reference
        try:
            outcome = "success" if test_passed is not False else "failed"
            self.task_memory.remember_task(
                task_id=task.id,
                title=task.title,
                description=task.description[:1000],
                agent=task.agent or "thor",
                files_changed=written,
                model_used=model_used,
                outcome=outcome,
                test_passed=test_passed,
                summary=summary[:500],
                elapsed_s=elapsed,
                tokens_used=input_tokens + output_tokens,
            )
            if test_passed:
                self.task_memory.learn_pattern(
                    "successful_task",
                    f"{task.agent or 'thor'}: {task.title[:80]}",
                    agent=task.agent or "thor",
                    confidence=0.7,
                )
        except Exception:
            pass

        # Git commit (auto-commit for completed tasks)
        if written:
            try:
                # Detect which repo the files belong to
                for fpath in written:
                    repo_dir = self._find_repo_root(fpath)
                    if repo_dir:
                        self.git.repo_dir = repo_dir
                        msg = self.git.auto_commit_message(written, task.title)
                        ok, result = self.git.commit_changes(written, msg, cwd=repo_dir)
                        if ok:
                            log.info("Git commit: %s — %s", result, msg[:60])
                        break
            except Exception as e:
                log.debug("Git commit skipped: %s", e)

        # Record in shared brain
        if self._brain:
            try:
                _ctx = f"Task: {task.title[:100]}, Agent: {task.agent or 'thor'}, Model: {model_used}"
                _dec = f"Completed task: {summary[:200]}" if summary else f"Completed: {task.title[:200]}"
                _did = self._brain.remember_decision(
                    _ctx, _dec, confidence=0.8 if test_passed else 0.5,
                    tags=[task.agent or "thor", model_used],
                )
                _score = 0.8 if test_passed else 0.4
                self._brain.remember_outcome(
                    _did, f"Tests {'passed' if test_passed else 'failed'}, {len(written)} files, {iterations} iterations",
                    score=_score,
                )
            except Exception:
                pass

        # Broadcast + Telegram
        self._notify_completion(task, summary, written, model_used, elapsed, phrase)

    def _fail_task(self, task: CodingTask, error: str, tokens: int = 0) -> None:
        """Mark task as failed and notify."""
        self.progress.fail_task(task.id, error[:200])
        task.status = "failed"
        task.error = error[:500]
        task.completed_at = time.time()
        self.queue.update_task(task)
        self.reporter.log_activity("task_failed", task.id, error[:200], success=False)
        self.reporter.update_status(state="idle", queue_depth=self.queue.get_stats()["pending"])
        self._notify_failure(task, error[:200])

        # Remember failure in task memory
        try:
            elapsed = task.completed_at - (task.started_at or task.completed_at)
            self.task_memory.remember_task(
                task_id=task.id,
                title=task.title,
                description=task.description[:1000],
                agent=task.agent or "thor",
                files_changed=[],
                model_used="",
                outcome="failed",
                test_passed=False,
                summary=error[:500],
                elapsed_s=elapsed,
                tokens_used=tokens,
                lessons=f"Failed: {error[:200]}",
            )
        except Exception:
            pass

    @staticmethod
    def _find_repo_root(filepath: str) -> str | None:
        """Walk up from filepath to find a .git directory (repo root)."""
        p = Path(filepath).resolve()
        for parent in [p.parent] + list(p.parents):
            if (parent / ".git").is_dir():
                return str(parent)
            if parent == Path.home() or str(parent) == "/":
                break
        return None

        # Heartbeat
        if self._hub:
            try:
                self._hub.heartbeat(status="coding", metrics={
                    "tasks_completed": self.queue.get_stats()["completed"],
                    "current_task": "",
                    "model": model_used,
                })
            except Exception:
                pass

    def _check_broadcasts(self) -> None:
        """Check and acknowledge brotherhood broadcasts."""
        bc_file = self.cfg.broadcast_file
        if not bc_file.exists():
            return
        try:
            data = json.loads(bc_file.read_text())
            if isinstance(data, list):
                changed = False
                for entry in data:
                    if isinstance(entry, dict) and not entry.get("acknowledged"):
                        msg = entry.get("message", "")
                        log.info("[BROADCAST] %s", msg[:200])
                        entry["acknowledged"] = True
                        entry["ack_timestamp"] = time.time()
                        changed = True
                        self.reporter.log_activity("broadcast_received", details=msg[:200])
                if changed:
                    bc_file.write_text(json.dumps(data, indent=2))
            elif isinstance(data, dict) and not data.get("acknowledged"):
                log.info("[BROADCAST] %s", data.get("message", "")[:200])
                data["acknowledged"] = True
                data["ack_timestamp"] = time.time()
                bc_file.write_text(json.dumps(data, indent=2))
                self.reporter.log_activity("broadcast_received", details=data.get("message", "")[:200])
        except Exception as e:
            log.debug("Broadcast check error: %s", e)

    def _maybe_suggest(self) -> None:
        """Proactive suggestions disabled — was spamming TG notifications."""
        return

    def _notify_completion(self, task: CodingTask, summary: str, files: list,
                           model: str, elapsed: float, phrase: str) -> None:
        """Broadcast task completion to Shelby and notify Jordan via Telegram."""
        try:
            # 1. Broadcast to all agents (Shelby will see it)
            sys.path.insert(0, str(Path.home() / "shelby"))
            from core.broadcast import broadcast_to_all
            bc_msg = (
                f"THOR — Task Complete: {task.title}\n"
                f"Files: {', '.join(files) if files else 'none'} | "
                f"Model: {model} | Time: {elapsed:.1f}s\n"
                f"Summary: {summary[:200]}"
            )
            broadcast_to_all(bc_msg, priority="normal", source="thor")
            log.info("Broadcast sent to brotherhood")
        except Exception as e:
            log.warning("Broadcast failed: %s", str(e)[:100])

        # 2. Send Telegram notification via Shelby
        self._send_telegram(task, summary, files, model, elapsed, phrase)

    def _send_telegram(self, task: CodingTask, summary: str, files: list,
                       model: str, elapsed: float, phrase: str) -> None:
        """Send task completion notification to Jordan via Telegram."""
        try:
            sys.path.insert(0, str(Path.home() / "shelby"))
            from core.telegram import TelegramBot
            bot = TelegramBot()
            if not bot.token:
                return

            file_list = ", ".join(files[:5]) if files else "none"
            if len(files) > 5:
                file_list += f" (+{len(files)-5} more)"

            is_high = task.priority in ("high", "critical")
            priority_icon = "🔴" if task.priority == "critical" else "🟠" if is_high else "🟢"

            lines = [
                f"⚡ *THOR — Task Complete*",
                f"",
                f"{priority_icon} *{task.title}*",
                f"📁 Files: `{file_list}`",
                f"🤖 Model: {model}",
                f"⏱ Time: {elapsed:.1f}s",
                f"",
                f"*Summary:* {summary[:300]}",
                f"",
                f'_"{phrase}"_',
            ]
            bot.send("\n".join(lines))
            log.info("Telegram notification sent")
        except Exception as e:
            log.warning("Telegram notification failed: %s", str(e)[:100])

    def _auto_detect_tests(self, written_files: list[str]) -> str | None:
        """Auto-detect test commands for written files.

        Searches for pytest/unittest in the project directory of each written file.
        Returns a test command string or None if no tests found.
        """
        tested_dirs = set()
        for fpath in written_files:
            if not fpath.endswith(".py"):
                continue
            p = Path(fpath)
            # Walk up to find project root (has tests/ dir, or pyproject.toml, or setup.py)
            for parent in [p.parent] + list(p.parents):
                if parent == Path.home() or str(parent) == "/":
                    break
                tests_dir = parent / "tests"
                test_dir = parent / "test"
                if tests_dir.is_dir() or test_dir.is_dir() or (parent / "pyproject.toml").exists():
                    tested_dirs.add(str(parent))
                    break
        if not tested_dirs:
            return None
        # Run pytest in each detected project root
        cmds = []
        for d in sorted(tested_dirs):
            tests_path = Path(d) / "tests"
            test_path = Path(d) / "test"
            if tests_path.is_dir():
                cmds.append(f"cd {d} && python3 -m pytest tests/ -x -q --tb=short 2>&1 | tail -20")
            elif test_path.is_dir():
                cmds.append(f"cd {d} && python3 -m pytest test/ -x -q --tb=short 2>&1 | tail -20")
            else:
                cmds.append(f"cd {d} && python3 -m pytest -x -q --tb=short 2>&1 | tail -20")
        return " && ".join(cmds) if cmds else None

    def _notify_failure(self, task: CodingTask, error: str) -> None:
        """Notify about task failure via broadcast + Telegram."""
        # Publish failure to shared event bus
        try:
            sys.path.insert(0, str(Path.home()))
            from shared.events import publish as bus_publish
            bus_publish(
                agent="thor",
                event_type="task_failed",
                data={"task_id": task.id, "title": task.title, "error": error[:200]},
                severity="warning",
                summary=f"Failed: {task.title} — {error[:80]}",
            )
        except Exception:
            pass

        try:
            sys.path.insert(0, str(Path.home() / "shelby"))
            from core.broadcast import broadcast_to_all
            bc_msg = (
                f"THOR — Task FAILED: {task.title}\n"
                f"Error: {error[:200]}\n"
                f"Retries: {task.retries}/{self.cfg.max_retries}"
            )
            broadcast_to_all(bc_msg, priority="high", source="thor")
        except Exception:
            pass

        try:
            sys.path.insert(0, str(Path.home() / "shelby"))
            from core.telegram import TelegramBot
            bot = TelegramBot()
            if not bot.token:
                return
            lines = [
                f"🚨 *THOR — Task Failed*",
                f"",
                f"*Task:* {task.title}",
                f"*Error:* `{error[:200]}`",
                f"*Retries:* {task.retries}/{self.cfg.max_retries}",
                f"",
                f"Manual review may be needed.",
            ]
            bot.send("\n".join(lines))
        except Exception:
            pass

    def submit_task(
        self,
        title: str,
        description: str,
        target_files: list[str] | None = None,
        context_files: list[str] | None = None,
        agent: str = "",
        priority: str = "normal",
        test_command: str = "",
        assigned_by: str = "claude",
    ) -> str:
        """Submit a task to Thor's queue. Returns task ID."""
        task = CodingTask(
            title=title,
            description=description,
            target_files=target_files or [],
            context_files=context_files or [],
            agent=agent,
            priority=priority,
            test_command=test_command,
            assigned_by=assigned_by,
        )
        return self.queue.submit(task)

    def get_status(self) -> dict:
        """Get Thor's full status."""
        # Calculate uptime
        uptime = "0h 0m"
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            uptime = f"{hours}h {minutes}m"
        
        return {
            "name": self.cfg.name,
            "archetype": self.cfg.archetype,
            "color": self.cfg.color,
            "running": self._running,
            "queue": self.queue.get_stats(),
            "brain": self.brain.get_stats(),
            "knowledge": self.knowledge.get_stats(),
            "daily": self.reporter.get_daily_summary(),
            "recent_activity": self.reporter.get_recent_activity(10),
            "uptime": uptime,
        }