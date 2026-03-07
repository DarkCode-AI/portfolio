"""Thor's coder — file operations, code writing, diff-based editing, and test execution."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger("thor.coder")


class Coder:
    """Handles file reading, writing, backups, and test execution."""

    # Thor must NEVER self-modify these files. Robotox escalation patches
    # destroyed agent.py (1053→226 lines) and brain.py (481→67 lines) on
    # Mar 6-7 2026. Jordan's order: core files require manual approval.
    PROTECTED_PATHS = frozenset({
        "thor/agent.py",
        "thor/__init__.py",
        "thor/__main__.py",
        "thor/config.py",
        "thor/core/brain.py",
        "thor/core/coder.py",
        "thor/core/engine.py",
        "thor/core/task_queue.py",
    })

    def __init__(self, backup_dir: Path | None = None):
        self.backup_dir = backup_dir or Path.home() / "thor" / "data" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._files_written: list[str] = []

    def _is_protected(self, path: Path) -> bool:
        """Check if a file is in Thor's protected core — self-modification blocked."""
        try:
            rel = path.resolve().relative_to(Path.home().resolve())
            return str(rel) in self.PROTECTED_PATHS
        except ValueError:
            return False

    def read_files(self, file_paths: list[str]) -> dict[str, str]:
        """Read multiple files and return {path: contents}."""
        contents = {}
        for fpath in file_paths:
            p = Path(fpath).expanduser()
            if not p.exists():
                log.warning("File not found: %s", fpath)
                continue
            if p.stat().st_size > 500_000:  # skip files > 500KB
                log.warning("File too large, skipping: %s (%d bytes)", fpath, p.stat().st_size)
                continue
            try:
                contents[fpath] = p.read_text()
            except Exception as e:
                log.warning("Cannot read %s: %s", fpath, e)
        return contents

    def _sanitize_path(self, fpath: str) -> Path:
        """Fix AI-hallucinated paths (e.g. /home/mike/ -> ~/). Reject unsafe paths."""
        home = str(Path.home())
        # Replace common hallucinated home dirs with actual home
        for bad_prefix in ["/home/mike", "/home/ubuntu", "/home/user", "/root"]:
            if fpath.startswith(bad_prefix):
                fpath = home + fpath[len(bad_prefix):]
                log.warning("Fixed hallucinated path: %s -> %s", bad_prefix, home)
                break
        p = Path(fpath).expanduser()
        # Safety: only allow writes under home directory
        try:
            p.resolve().relative_to(Path.home().resolve())
        except ValueError:
            log.error("BLOCKED write outside home dir: %s", fpath)
            return None
        # Safety: Thor must not self-modify its own core files
        if self._is_protected(p):
            log.error("BLOCKED self-modification of protected file: %s", fpath)
            return None
        return p

    def write_files(self, files: dict[str, str], backup: bool = True) -> list[str]:
        """Write files to disk, creating backups of existing files.

        Args:
            files: {filepath: new_contents}
            backup: Whether to backup existing files before overwriting

        Returns:
            List of successfully written file paths
        """
        written = []
        for fpath, content in files.items():
            p = self._sanitize_path(fpath)
            if p is None:
                continue

            # Create parent directories if needed
            p.parent.mkdir(parents=True, exist_ok=True)

            # Backup existing file
            if backup and p.exists():
                self._backup_file(p)

            try:
                p.write_text(content)
                written.append(fpath)
                log.info("Written: %s (%d chars)", fpath, len(content))
            except Exception as e:
                log.error("Failed to write %s: %s", fpath, e)

        self._files_written.extend(written)
        return written

    def validate_syntax_before_write(self, files: dict[str, str]) -> dict[str, str | None]:
        """Validate Python syntax before writing files to disk.
        
        Returns:
            {filepath: error_message_or_None}
        """
        results = {}
        for fpath, content in files.items():
            if not fpath.endswith(".py"):
                results[fpath] = None
                continue
            try:
                compile(content, fpath, "exec")
                results[fpath] = None
            except SyntaxError as e:
                results[fpath] = f"Line {e.lineno}: {e.msg}"
                log.warning("Syntax error detected before write in %s line %d: %s", fpath, e.lineno, e.msg)
        return results

    def _backup_file(self, path: Path) -> None:
        """Create a timestamped backup of a file."""
        ts = int(time.time())
        backup_name = f"{path.name}.{ts}.bak"
        backup_path = self.backup_dir / backup_name
        try:
            shutil.copy2(path, backup_path)
            log.debug("Backed up: %s -> %s", path, backup_path)
        except Exception as e:
            log.warning("Backup failed for %s: %s", path, e)

    def run_test(self, command: str, timeout: int = 30) -> tuple[bool, str]:
        """Run a test command and return (passed, output).

        Args:
            command: Shell command to run
            timeout: Timeout in seconds

        Returns:
            (success: bool, output: str)
        """
        if not command:
            return True, "No test command specified"

        log.info("Running test: %s", command[:100])
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path.home()),
            )
            output = result.stdout + result.stderr
            passed = result.returncode == 0

            if passed:
                log.info("Test PASSED")
            else:
                # Enhanced error reporting for test failures
                if "SyntaxError" in output:
                    log.error("Test FAILED due to SYNTAX ERROR (exit %d): %s", result.returncode, output[:500])
                elif "ImportError" in output or "ModuleNotFoundError" in output:
                    log.error("Test FAILED due to IMPORT ERROR (exit %d): %s", result.returncode, output[:500])
                else:
                    log.warning("Test FAILED (exit %d): %s", result.returncode, output[:200])

            return passed, output[:2000]
        except subprocess.TimeoutExpired:
            log.warning("Test timed out after %ds", timeout)
            return False, f"Test timed out after {timeout}s"
        except Exception as e:
            log.error("Test execution error: %s", e)
            return False, str(e)

    def rollback_last(self) -> list[str]:
        """Rollback the most recently written files from backups."""
        rolled_back = []
        for fpath in reversed(self._files_written):
            p = Path(fpath)
            # Find most recent backup
            backups = sorted(
                self.backup_dir.glob(f"{p.name}.*.bak"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if backups:
                shutil.copy2(backups[0], p)
                rolled_back.append(fpath)
                log.info("Rolled back: %s", fpath)
        self._files_written.clear()
        return rolled_back

    def run_script(self, script_path: str, timeout: int = 60) -> tuple[bool, str]:
        """Execute a Python script that Thor generated.

        Args:
            script_path: Path to the script to run
            timeout: Max execution time in seconds

        Returns:
            (success: bool, output: str)
        """
        p = Path(script_path).expanduser()
        if not p.exists():
            return False, f"Script not found: {script_path}"
        if not p.suffix == ".py":
            return False, f"Not a Python file: {script_path}"

        log.info("Executing script: %s", script_path)
        try:
            result = subprocess.run(
                ["python3", str(p)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(p.parent),
            )
            output = result.stdout + result.stderr
            passed = result.returncode == 0
            if passed:
                log.info("Script executed successfully")
            else:
                log.warning("Script failed (exit %d): %s", result.returncode, output[:300])
            return passed, output[:3000]
        except subprocess.TimeoutExpired:
            return False, f"Script timed out after {timeout}s"
        except Exception as e:
            return False, str(e)[:500]

    def validate_syntax(self, files: dict[str, str]) -> dict[str, str | None]:
        """Validate Python syntax for generated files.

        Returns:
            {filepath: error_message_or_None}
        """
        results = {}
        for fpath, content in files.items():
            if not fpath.endswith(".py"):
                results[fpath] = None
                continue
            try:
                compile(content, fpath, "exec")
                results[fpath] = None
            except SyntaxError as e:
                results[fpath] = f"Line {e.lineno}: {e.msg}"
                log.warning("Syntax error in %s line %d: %s", fpath, e.lineno, e.msg)
        return results

    def quality_gate(self, files: dict[str, str]) -> dict:
        """Run full quality pipeline on generated Python files.

        Checks: syntax, ruff lint, bandit security scan.
        Returns a quality report with score and per-file details.
        """
        import tempfile

        report = {
            "passed": True,
            "score": 100,
            "files": {},
            "total_lint_issues": 0,
            "total_security_issues": 0,
            "syntax_errors": 0,
        }

        for fpath, content in files.items():
            if not fpath.endswith(".py"):
                continue

            file_report = {"syntax": None, "lint": [], "security": [], "lint_count": 0, "security_count": 0}

            # 1. Syntax check
            try:
                compile(content, fpath, "exec")
            except SyntaxError as e:
                file_report["syntax"] = f"Line {e.lineno}: {e.msg}"
                report["syntax_errors"] += 1
                report["passed"] = False
                report["score"] -= 30
                log.warning("QG syntax error in %s: %s", fpath, file_report["syntax"])

            # 2. Ruff lint + 3. Bandit security — write temp file
            tmp = None
            try:
                tmp = Path(tempfile.mktemp(suffix=".py"))
                tmp.write_text(content)

                # Ruff lint (fast, Rust-based)
                file_report["lint"] = self._run_ruff(tmp, fpath)
                file_report["lint_count"] = len(file_report["lint"])
                report["total_lint_issues"] += file_report["lint_count"]

                # Bandit security scan
                file_report["security"] = self._run_bandit(tmp, fpath)
                file_report["security_count"] = len(file_report["security"])
                report["total_security_issues"] += file_report["security_count"]

            except Exception as e:
                log.warning("QG tool error for %s: %s", fpath, e)
            finally:
                if tmp and tmp.exists():
                    tmp.unlink()

            report["files"][fpath] = file_report

        # Score deductions
        report["score"] -= report["total_lint_issues"] * 2  # -2 per lint issue
        report["score"] -= report["total_security_issues"] * 10  # -10 per security issue
        report["score"] = max(0, report["score"])

        if report["total_security_issues"] > 0:
            report["passed"] = False
            log.warning("QG FAILED: %d security issues found", report["total_security_issues"])

        if report["score"] < 60:
            report["passed"] = False
            log.warning("QG FAILED: score %d < 60 threshold", report["score"])

        log.info(
            "Quality gate: score=%d, lint=%d, security=%d, passed=%s",
            report["score"], report["total_lint_issues"],
            report["total_security_issues"], report["passed"],
        )
        return report

    def _run_ruff(self, tmp_path: Path, original_path: str) -> list[str]:
        """Run ruff linter on a temp file. Returns list of issue strings."""
        try:
            result = subprocess.run(
                ["python3", "-m", "ruff", "check", "--select=E,F", "--ignore=E501,F401,E402,F841", "--no-fix", str(tmp_path)],
                capture_output=True, text=True, timeout=15,
            )
            issues = []
            for line in result.stdout.strip().split("\n"):
                if line.strip() and not line.startswith("Found"):
                    # Replace temp path with original for readability
                    issues.append(line.replace(str(tmp_path), os.path.basename(original_path)))
            return issues
        except FileNotFoundError:
            log.debug("ruff not installed, skipping lint")
            return []
        except Exception as e:
            log.debug("ruff error: %s", e)
            return []

    def _run_bandit(self, tmp_path: Path, original_path: str) -> list[str]:
        """Run bandit security scanner on a temp file. Returns list of issue strings."""
        try:
            result = subprocess.run(
                ["python3", "-m", "bandit", "-q", "-ll", str(tmp_path)],
                capture_output=True, text=True, timeout=15,
            )
            issues = []
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line and ("Issue:" in line or "Severity:" in line or "CWE:" in line):
                    issues.append(line.replace(str(tmp_path), os.path.basename(original_path)))
            return issues
        except FileNotFoundError:
            log.debug("bandit not installed, skipping security scan")
            return []
        except Exception as e:
            log.debug("bandit error: %s", e)
            return []

    def apply_diff(self, filepath: str, diff_text: str) -> tuple[bool, str]:
        """Apply a unified diff patch to a file.

        Args:
            filepath: Path to the file to patch
            diff_text: Unified diff text (lines starting with +/- and context)

        Returns:
            (success, result_or_error)
        """
        p = self._sanitize_path(filepath)
        if p is None:
            return False, "Path blocked by safety check"

        if not p.exists():
            return False, f"File not found: {filepath}"

        original = p.read_text()
        original_lines = original.splitlines(keepends=True)

        # Parse the diff hunks
        try:
            patched_lines = self._apply_unified_diff(original_lines, diff_text)
        except Exception as e:
            log.warning("Diff apply failed for %s: %s", filepath, e)
            return False, f"Diff apply failed: {e}"

        patched = "".join(patched_lines)

        # Backup before writing
        if p.exists():
            self._backup_file(p)

        p.write_text(patched)
        self._files_written.append(filepath)
        log.info("Diff applied: %s (%d lines changed)", filepath, self._count_diff_changes(diff_text))
        return True, filepath

    def apply_diffs(self, diffs: dict[str, str]) -> list[str]:
        """Apply multiple diff patches. Returns list of successfully patched files."""
        applied = []
        for filepath, diff_text in diffs.items():
            ok, result = self.apply_diff(filepath, diff_text)
            if ok:
                applied.append(result)
            else:
                log.error("Failed to apply diff to %s: %s", filepath, result)
        return applied

    def _apply_unified_diff(self, original_lines: list[str], diff_text: str) -> list[str]:
        """Apply unified diff hunks to original lines.

        Uses CONTEXT-AWARE matching: finds the right position by matching
        context lines (lines starting with ' ') against the original file,
        rather than trusting @@ line numbers (which LLMs often get wrong).
        """
        diff_lines = diff_text.splitlines(keepends=True)
        result = list(original_lines)

        # Parse @@ hunks
        hunks = []
        current_hunk = None
        for line in diff_lines:
            stripped = line.rstrip("\n\r")
            if stripped.startswith("@@"):
                parts = stripped.split("@@")
                if len(parts) >= 3:
                    ranges = parts[1].strip().split()
                    old_start = int(ranges[0].split(",")[0].lstrip("-"))
                    current_hunk = {"old_start": old_start, "lines": []}
                    hunks.append(current_hunk)
            elif current_hunk is not None:
                current_hunk["lines"].append(line)

        if not hunks:
            return self._apply_simple_diff(original_lines, diff_lines)

        # Apply hunks in reverse order (bottom-up)
        for hunk in reversed(hunks):
            # Extract the "old" lines (context + removed) for matching
            old_lines = []
            new_lines = []
            for line in hunk["lines"]:
                if line.startswith("-"):
                    old_lines.append(line[1:])
                elif line.startswith("+"):
                    new_lines.append(line[1:])
                elif line.startswith(" "):
                    old_lines.append(line[1:])
                    new_lines.append(line[1:])
                elif not line.startswith("\\"):
                    old_lines.append(line)
                    new_lines.append(line)

            if not old_lines:
                continue

            # Context-aware matching: find where old_lines appear in result
            match_offset = self._find_context_match(result, old_lines, hunk["old_start"] - 1)

            if match_offset is not None:
                result[match_offset:match_offset + len(old_lines)] = new_lines
            else:
                log.warning("Could not find context match for hunk at line %d, skipping", hunk["old_start"])

        return result

    def _find_context_match(
        self, lines: list[str], pattern: list[str], hint_offset: int
    ) -> int | None:
        """Find where a pattern of lines appears in the file.

        Uses hint_offset (from @@ header) as starting point, then searches
        outward in expanding window if exact position doesn't match.
        """
        if not pattern:
            return None

        def _lines_match(a: str, b: str) -> bool:
            """Compare lines ignoring trailing whitespace."""
            return a.rstrip() == b.rstrip()

        def _check_at(offset: int) -> bool:
            """Check if pattern matches at this offset."""
            if offset < 0 or offset + len(pattern) > len(lines):
                return False
            return all(_lines_match(lines[offset + i], pattern[i]) for i in range(len(pattern)))

        # Try hint position first
        if _check_at(hint_offset):
            return hint_offset

        # Search outward from hint (up to 50 lines in each direction)
        for delta in range(1, 50):
            if _check_at(hint_offset - delta):
                return hint_offset - delta
            if _check_at(hint_offset + delta):
                return hint_offset + delta

        # Last resort: search entire file
        for i in range(len(lines)):
            if _check_at(i):
                return i

        return None

    def _apply_simple_diff(self, original_lines: list[str], diff_lines: list[str]) -> list[str]:
        """Fallback: apply diff without @@ headers using +/- line matching."""
        result = list(original_lines)
        removals = []
        additions = []

        for line in diff_lines:
            if line.startswith("-") and not line.startswith("---"):
                removals.append(line[1:])
            elif line.startswith("+") and not line.startswith("+++"):
                additions.append(line[1:])

        # Find and replace the removal block with additions
        if removals:
            removal_text = "".join(removals)
            original_text = "".join(result)
            if removal_text in original_text:
                new_text = original_text.replace(removal_text, "".join(additions), 1)
                result = new_text.splitlines(keepends=True)

        return result

    @staticmethod
    def _count_diff_changes(diff_text: str) -> int:
        """Count the number of changed lines in a diff."""
        count = 0
        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                count += 1
            elif line.startswith("-") and not line.startswith("---"):
                count += 1
        return count

    def discover_project_files(self, project_dir: str, extensions: list[str] | None = None) -> list[str]:
        """List Python files in a project directory for context."""
        if extensions is None:
            extensions = [".py"]

        p = Path(project_dir).expanduser()
        if not p.exists():
            return []

        files = []
        for ext in extensions:
            for f in p.rglob(f"*{ext}"):
                # Skip common non-essential dirs
                parts = f.parts
                if any(skip in parts for skip in ("__pycache__", ".git", "node_modules", ".venv", "venv")):
                    continue
                files.append(str(f))

        return sorted(files)
