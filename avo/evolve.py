from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .agent import VariationDecision
from .isolation import RESULT_PREFIX
from .lineage import GateDecision, commit_score

DEFAULT_ALLOWED_SUBCOMMANDS = frozenset({"env", "compile", "score"})
SHELL_TOKENS = frozenset({"&&", "||", ";", "|", ">", ">>", "<", "`"})
DEFAULT_ATTEMPT_HISTORY_LIMIT = 5
DEFAULT_PATCH_ROOTS = ("candidates/",)
REJECTED_PATCH_MARKERS = frozenset(
    {
        "Binary files ",
        "GIT binary patch",
        "deleted file mode ",
        "rename from ",
        "rename to ",
        "copy from ",
        "copy to ",
        "similarity index ",
        "dissimilarity index ",
    }
)


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int | None
    timed_out: bool
    stdout_tail: str
    stderr_tail: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "command": self.command,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


@dataclass(frozen=True)
class PatchResult:
    ok: bool
    patch_paths: list[str]
    returncode: int | None
    stdout_tail: str
    stderr_tail: str
    rejected_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "patch_paths": self.patch_paths,
            "returncode": self.returncode,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "rejected_reason": self.rejected_reason,
        }


@dataclass(frozen=True)
class VariationAttempt:
    decision: VariationDecision
    command_result: CommandResult
    started_at: str
    completed_at: str
    score_payload: dict[str, Any] | None = None
    patch_result: PatchResult | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.as_dict(),
            "command_result": self.command_result.as_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "score_payload": self.score_payload,
            "patch_result": (
                self.patch_result.as_dict() if self.patch_result is not None else None
            ),
        }


@dataclass(frozen=True)
class EvolutionStep:
    attempt: VariationAttempt
    gate_decision: GateDecision | None
    patch_cleanup_result: PatchResult | None = None

    @property
    def accepted(self) -> bool:
        return self.gate_decision is not None and self.gate_decision.accepted

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt.as_dict(),
            "gate_decision": (
                self.gate_decision.as_dict() if self.gate_decision is not None else None
            ),
            "patch_cleanup_result": (
                self.patch_cleanup_result.as_dict()
                if self.patch_cleanup_result is not None
                else None
            ),
        }


def command_from_decision(
    decision: VariationDecision,
    *,
    allowed_subcommands: frozenset[str] = DEFAULT_ALLOWED_SUBCOMMANDS,
) -> list[str]:
    parts = shlex.split(decision.next_command)
    if len(parts) < 2:
        raise ValueError("next_command must start with an avo subcommand")
    if parts[0] != "avo":
        raise ValueError("next_command must start with 'avo'")
    if any(part in SHELL_TOKENS for part in parts):
        raise ValueError("next_command must not contain shell control tokens")

    subcommand = parts[1]
    if subcommand not in allowed_subcommands:
        allowed = ", ".join(sorted(allowed_subcommands))
        raise ValueError(f"unsupported avo subcommand '{subcommand}'; allowed: {allowed}")
    return [sys.executable, "-m", "avo", *parts[1:]]


def paths_from_unified_diff(
    patch_text: str,
    *,
    allowed_roots: tuple[str, ...] = DEFAULT_PATCH_ROOTS,
) -> list[str]:
    paths: set[str] = set()
    current_diff_paths: set[str] = set()
    in_hunk = False
    saw_diff_git = False

    for raw_line in patch_text.splitlines():
        if _contains_rejected_patch_marker(raw_line):
            raise ValueError(f"unsupported patch marker: {_rejected_patch_marker(raw_line)}")

        if raw_line.startswith("diff --git "):
            saw_diff_git = True
            in_hunk = False
            current_diff_paths = _paths_from_diff_git_line(raw_line, allowed_roots=allowed_roots)
            paths.update(current_diff_paths)
            continue

        if raw_line.startswith("@@"):
            in_hunk = True
            continue

        if in_hunk:
            continue

        if raw_line.startswith("new file mode "):
            if raw_line != "new file mode 100644":
                raise ValueError("new files must use mode 100644")
            continue

        if raw_line.startswith(("old mode ", "new mode ")):
            raise ValueError("mode changes are not supported")

        if raw_line.startswith("--- ") or raw_line.startswith("+++ "):
            path = _path_from_file_header(raw_line)
            if path is None:
                continue
            normalized = _validate_patch_path(path, allowed_roots=allowed_roots)
            if not current_diff_paths:
                raise ValueError("file headers must follow a diff --git path")
            if normalized not in current_diff_paths:
                raise ValueError("file header path does not match diff --git path")
            paths.add(normalized)

    if not saw_diff_git or not paths:
        raise ValueError("patch must contain at least one diff --git path")
    return sorted(paths)


def apply_candidate_patch(
    patch_text: str,
    *,
    cwd: Path,
    dry_run: bool = False,
) -> PatchResult:
    try:
        patch_paths = paths_from_unified_diff(patch_text)
        _validate_patch_worktree(cwd, patch_paths)
    except ValueError as exc:
        return PatchResult(
            ok=False,
            patch_paths=[],
            returncode=None,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=str(exc),
        )

    check = _run_git_apply(
        cwd,
        patch_text,
        "--check",
        "--whitespace=error",
    )
    if check.returncode != 0:
        return PatchResult(
            ok=False,
            patch_paths=patch_paths,
            returncode=check.returncode,
            stdout_tail=_tail(check.stdout),
            stderr_tail=_tail(check.stderr),
            rejected_reason="git apply --check failed",
        )

    if dry_run:
        return PatchResult(
            ok=True,
            patch_paths=patch_paths,
            returncode=check.returncode,
            stdout_tail=_tail(check.stdout),
            stderr_tail=_tail(check.stderr),
        )

    applied = _run_git_apply(
        cwd,
        patch_text,
        "--whitespace=error",
    )
    return PatchResult(
        ok=applied.returncode == 0,
        patch_paths=patch_paths,
        returncode=applied.returncode,
        stdout_tail=_tail(applied.stdout),
        stderr_tail=_tail(applied.stderr),
        rejected_reason=None if applied.returncode == 0 else "git apply failed",
    )


def revert_candidate_patch(
    patch_text: str,
    *,
    cwd: Path,
    dry_run: bool = False,
) -> PatchResult:
    try:
        patch_paths = paths_from_unified_diff(patch_text)
        _validate_patch_worktree(cwd, patch_paths)
    except ValueError as exc:
        return PatchResult(
            ok=False,
            patch_paths=[],
            returncode=None,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=str(exc),
        )

    check = _run_git_apply(
        cwd,
        patch_text,
        "--reverse",
        "--check",
        "--whitespace=error",
    )
    if check.returncode != 0:
        return PatchResult(
            ok=False,
            patch_paths=patch_paths,
            returncode=check.returncode,
            stdout_tail=_tail(check.stdout),
            stderr_tail=_tail(check.stderr),
            rejected_reason="git apply --reverse --check failed",
        )

    if dry_run:
        return PatchResult(
            ok=True,
            patch_paths=patch_paths,
            returncode=check.returncode,
            stdout_tail=_tail(check.stdout),
            stderr_tail=_tail(check.stderr),
        )

    reverted = _run_git_apply(
        cwd,
        patch_text,
        "--reverse",
        "--whitespace=error",
    )
    return PatchResult(
        ok=reverted.returncode == 0,
        patch_paths=patch_paths,
        returncode=reverted.returncode,
        stdout_tail=_tail(reverted.stdout),
        stderr_tail=_tail(reverted.stderr),
        rejected_reason=None if reverted.returncode == 0 else "git apply --reverse failed",
    )


def run_decision_command(
    decision: VariationDecision,
    *,
    cwd: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
    allowed_subcommands: frozenset[str] = DEFAULT_ALLOWED_SUBCOMMANDS,
) -> VariationAttempt:
    command = command_from_decision(decision, allowed_subcommands=allowed_subcommands)
    started_at = _utc_now()
    patch_result = _maybe_apply_candidate_patch(decision, cwd=cwd)
    if patch_result is not None and not patch_result.ok:
        return VariationAttempt(
            decision=decision,
            command_result=CommandResult(
                command=command,
                returncode=None,
                timed_out=False,
                stdout_tail="",
                stderr_tail=_patch_failure_summary(patch_result),
            ),
            started_at=started_at,
            completed_at=_utc_now(),
            patch_result=patch_result,
        )

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        result = CommandResult(
            command=command,
            returncode=completed.returncode,
            timed_out=False,
            stdout_tail=_tail(completed.stdout),
            stderr_tail=_tail(completed.stderr),
        )
        score_payload = _extract_score_payload(completed.stdout)
    except subprocess.TimeoutExpired as exc:
        result = CommandResult(
            command=command,
            returncode=None,
            timed_out=True,
            stdout_tail=_tail(exc.stdout or ""),
            stderr_tail=_tail(exc.stderr or ""),
        )
        score_payload = None
    return VariationAttempt(
        decision=decision,
        command_result=result,
        started_at=started_at,
        completed_at=_utc_now(),
        score_payload=score_payload,
        patch_result=patch_result,
    )


def finalize_attempt(
    lineage: Path,
    attempt: VariationAttempt,
    *,
    message: str = "evolve: accept candidate",
    source_root: Path | None = None,
) -> EvolutionStep:
    gate_decision = None
    if attempt.score_payload is not None:
        source_files = _candidate_source_snapshot(source_root, attempt.patch_result)
        candidate_patch = attempt.decision.candidate_patch if source_files else None
        gate_decision = commit_score(
            lineage,
            attempt.score_payload,
            message=message,
            source_files=source_files,
            candidate_patch=candidate_patch,
        )
    return EvolutionStep(attempt=attempt, gate_decision=gate_decision)


def cleanup_rejected_candidate_patch(step: EvolutionStep, *, cwd: Path) -> EvolutionStep:
    if step.accepted:
        return step
    patch_result = step.attempt.patch_result
    if patch_result is None or not patch_result.ok:
        return step
    cleanup_result = revert_candidate_patch(step.attempt.decision.candidate_patch, cwd=cwd)
    return EvolutionStep(
        attempt=step.attempt,
        gate_decision=step.gate_decision,
        patch_cleanup_result=cleanup_result,
    )


def write_attempt(path: Path, attempt: VariationAttempt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(attempt.as_dict(), indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def write_step(path: Path, step: EvolutionStep) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(step.as_dict(), indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def write_step_record(directory: Path, step: EvolutionStep) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = _filename_safe_timestamp(step.attempt.completed_at)
    path = directory / f"{stem}.json"
    suffix = 1
    while path.exists():
        path = directory / f"{stem}-{suffix}.json"
        suffix += 1
    write_step(path, step)
    return path


def summarize_attempt_history(
    directory: Path | None,
    *,
    limit: int = DEFAULT_ATTEMPT_HISTORY_LIMIT,
) -> str:
    if directory is None or limit <= 0 or not directory.exists():
        return ""
    paths = sorted(path for path in directory.glob("*.json") if path.is_file())
    records = []
    for path in paths[-limit:]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            records.append(_summarize_step_payload(path.name, payload))
    if not records:
        return ""
    return "Recent attempts, oldest to newest:\n" + "\n".join(f"- {record}" for record in records)


def _extract_score_payload(stdout: str) -> dict[str, Any] | None:
    stripped = stdout.strip()
    if stripped:
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed.get("payload")
            if isinstance(payload, dict):
                return payload
            if _looks_like_score_payload(parsed):
                return parsed

    for line in reversed(stdout.splitlines()):
        if not line.startswith(RESULT_PREFIX):
            continue
        try:
            parsed = json.loads(line.removeprefix(RESULT_PREFIX))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) and _looks_like_score_payload(parsed) else None
    return None


def _looks_like_score_payload(payload: dict[str, Any]) -> bool:
    return (
        isinstance(payload.get("all_correct"), bool)
        and "geomean_tflops" in payload
        and isinstance(payload.get("cases"), list)
    )


def _maybe_apply_candidate_patch(decision: VariationDecision, *, cwd: Path) -> PatchResult | None:
    if not decision.candidate_patch.strip():
        return None
    return apply_candidate_patch(decision.candidate_patch, cwd=cwd)


def _candidate_source_snapshot(
    source_root: Path | None,
    patch_result: PatchResult | None,
) -> dict[str, str] | None:
    if source_root is None or patch_result is None or not patch_result.ok:
        return None
    return {
        path: (source_root / path).read_text(encoding="utf-8")
        for path in patch_result.patch_paths
    }


def _patch_failure_summary(result: PatchResult) -> str:
    if result.rejected_reason:
        return f"candidate patch rejected: {result.rejected_reason}"
    if result.stderr_tail:
        return f"candidate patch failed: {result.stderr_tail}"
    return "candidate patch failed"


def _contains_rejected_patch_marker(line: str) -> bool:
    return _rejected_patch_marker(line) is not None


def _rejected_patch_marker(line: str) -> str | None:
    for marker in REJECTED_PATCH_MARKERS:
        if line.startswith(marker):
            return marker.strip()
    if "120000" in line and "mode " in line:
        return "symlink mode 120000"
    return None


def _paths_from_diff_git_line(line: str, *, allowed_roots: tuple[str, ...]) -> set[str]:
    parts = line.split()
    if len(parts) != 4:
        raise ValueError("diff --git paths with whitespace or quoting are not supported")
    left = _validate_prefixed_diff_path(parts[2], allowed_roots=allowed_roots)
    right = _validate_prefixed_diff_path(parts[3], allowed_roots=allowed_roots)
    if left != right:
        raise ValueError("renames and cross-path patches are not supported")
    return {left}


def _path_from_file_header(line: str) -> str | None:
    _, _, rest = line.partition(" ")
    path = rest.split("\t", maxsplit=1)[0]
    if path == "/dev/null":
        return None
    return path


def _validate_prefixed_diff_path(path: str, *, allowed_roots: tuple[str, ...]) -> str:
    if not path.startswith(("a/", "b/")):
        raise ValueError("diff paths must use a/ and b/ prefixes")
    return _validate_patch_path(path[2:], allowed_roots=allowed_roots)


def _validate_patch_path(path: str, *, allowed_roots: tuple[str, ...]) -> str:
    if path.startswith(("a/", "b/")):
        path = path[2:]
    if not path:
        raise ValueError("patch path is empty")
    if "\x00" in path or "\\" in path:
        raise ValueError("patch path contains unsupported characters")
    if any(char.isspace() for char in path):
        raise ValueError("patch paths with whitespace are not supported")
    if path.startswith("/"):
        raise ValueError("absolute patch paths are not supported")

    posix_path = PurePosixPath(path)
    if posix_path.is_absolute():
        raise ValueError("absolute patch paths are not supported")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise ValueError("path traversal is not supported")
    if ".git" in posix_path.parts:
        raise ValueError("patch paths must not contain .git")

    normalized = posix_path.as_posix()
    if not any(
        normalized.startswith(root) and normalized != root.rstrip("/") for root in allowed_roots
    ):
        roots = ", ".join(root.rstrip("/") for root in allowed_roots)
        raise ValueError(f"patch paths must be under: {roots}")
    return normalized


def _validate_patch_worktree(cwd: Path, patch_paths: list[str]) -> None:
    if not cwd.exists():
        raise ValueError(f"cwd does not exist: {cwd}")
    if not cwd.is_dir():
        raise ValueError(f"cwd is not a directory: {cwd}")
    for patch_path in patch_paths:
        if _path_has_symlink(cwd, patch_path):
            raise ValueError(f"patch path crosses an existing symlink: {patch_path}")


def _path_has_symlink(cwd: Path, patch_path: str) -> bool:
    current = cwd
    for part in PurePosixPath(patch_path).parts:
        current = current / part
        if current.is_symlink():
            return True
        if not current.exists():
            return False
    return False


def _run_git_apply(cwd: Path, patch_text: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "apply", *args],
        cwd=cwd,
        input=patch_text,
        text=True,
        capture_output=True,
        check=False,
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _filename_safe_timestamp(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "-" for char in value)
    return cleaned.strip("-") or _utc_now().replace(":", "-")


def _summarize_step_payload(name: str, payload: dict[str, Any]) -> str:
    attempt = payload.get("attempt") if isinstance(payload.get("attempt"), dict) else {}
    decision = attempt.get("decision") if isinstance(attempt.get("decision"), dict) else {}
    command_result = (
        attempt.get("command_result") if isinstance(attempt.get("command_result"), dict) else {}
    )
    patch_result = (
        attempt.get("patch_result") if isinstance(attempt.get("patch_result"), dict) else None
    )
    patch_cleanup_result = (
        payload.get("patch_cleanup_result")
        if isinstance(payload.get("patch_cleanup_result"), dict)
        else None
    )
    score_payload = attempt.get("score_payload")
    gate_decision = payload.get("gate_decision")

    status = _command_status(command_result)
    patch = _patch_status(patch_result)
    cleanup = _patch_cleanup_status(patch_cleanup_result)
    gate = _gate_status(gate_decision)
    score = _score_status(score_payload)
    command = _shorten(str(decision.get("next_command") or "<missing command>"), 180)
    hypothesis = _shorten(str(decision.get("hypothesis") or "<missing hypothesis>"), 180)
    return (
        f"{name}: {status}; {patch}; {cleanup}; {gate}; {score}; "
        f"command={command}; hypothesis={hypothesis}"
    )


def _command_status(command_result: dict[str, Any]) -> str:
    if command_result.get("timed_out"):
        return "command timed out"
    returncode = command_result.get("returncode")
    if returncode == 0:
        return "command ok"
    if returncode is None:
        return "command not run"
    return f"command returncode={returncode}"


def _patch_status(patch_result: Any) -> str:
    if not isinstance(patch_result, dict):
        return "no candidate patch"
    ok = patch_result.get("ok")
    paths = patch_result.get("patch_paths")
    reason = _shorten(str(patch_result.get("rejected_reason") or ""), 120)
    if ok:
        return f"patch ok paths={paths}"
    return f"patch rejected reason={reason}"


def _patch_cleanup_status(cleanup_result: Any) -> str:
    if not isinstance(cleanup_result, dict):
        return "no patch cleanup"
    ok = cleanup_result.get("ok")
    reason = _shorten(str(cleanup_result.get("rejected_reason") or ""), 120)
    if ok:
        return "patch cleanup ok"
    return f"patch cleanup failed reason={reason}"


def _gate_status(gate_decision: Any) -> str:
    if not isinstance(gate_decision, dict):
        return "no gate decision"
    accepted = gate_decision.get("accepted")
    reason = _shorten(str(gate_decision.get("reason") or "<missing reason>"), 120)
    return f"gate accepted={accepted} reason={reason}"


def _score_status(score_payload: Any) -> str:
    if not isinstance(score_payload, dict):
        return "no score payload"
    geomean = score_payload.get("geomean_tflops")
    correct = score_payload.get("all_correct")
    return f"score all_correct={correct} geomean_tflops={geomean}"


def _shorten(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:]
