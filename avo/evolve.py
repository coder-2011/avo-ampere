from __future__ import annotations

import ast
import difflib
import hashlib
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .agent import (
    SUPPORT_ONLY_TRANSFORM_OPS,
    VariationDecision,
    candidate_patch_structural_advisories,
    promoted_preflight_track_names_for_classes,
    validate_candidate_patch_structural_preflight,
)
from .isolation import RESULT_PREFIX
from .lineage import GateDecision, commit_score

DEFAULT_ALLOWED_SUBCOMMANDS = frozenset({"env", "compile", "profile", "score"})
SHELL_TOKENS = frozenset({"&&", "||", ";", "|", ">", ">>", "<", "`"})
DEFAULT_ATTEMPT_HISTORY_LIMIT = 5
DEFAULT_PATCH_ROOTS = ("candidates/",)
CANDIDATE_SOURCE_SUFFIXES = frozenset({".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"})
SKIPPED_SOURCE_PARTS = frozenset({"__pycache__"})
SUPERVISOR_REPEAT_THRESHOLD = 3
SUPERVISOR_EXHAUSTION_THRESHOLD = 5
PROMOTED_PREFLIGHT_TRACKS_FILENAME = "preflight_tracks.json"
IGNORED_FAILURE_CLASSES = frozenset(
    {"accepted", "unknown", "command_failed", "planner_provider_error"}
)
COMPILE_FAILURE_DETAIL_MARKERS = (
    "nvcc",
    "ptxas",
    "ninja: build stopped",
    "error building extension",
    "compilation failed",
    "compileerror",
    ".cu(",
    ".cuh(",
    "attention_kernel.cu",
)
PROMOTABLE_FAILURE_CLASS_TRACKS = {
    "planning_edit_channel": "edit_channel_consistency",
    "planning_missing_edit_payload": "edit_channel_consistency",
    "planning_no_patch_compile": "compile_diagnostic_repetition",
    "planning_support_only_transform": "semantic_transform_contract",
    "planning_transform_semantic_mismatch": "semantic_transform_contract",
    "planning_transform_preflight": "transform_materialization",
    "planning_predicted_correctness_failure": "planning_risk_contract",
    "planning_validation": "planning_validation",
    "raw_diff_preflight": "edit_channel_integrity",
    "structured_transform_preflight": "transform_materialization",
    "patch_safety_preflight": "patch_safety",
    "no_effect_or_skeleton": "no_effect_skeleton",
    "incomplete_or_malformed_edit": "incomplete_edit",
    "unsupported_wmma_shape": "wmma_fragment_shape",
    "cuda_syntax_error": "cuda_text_shape",
    "stale_or_undefined_symbol": "symbol_lifecycle",
    "correctness_failed": "correctness_preflight",
    "correctness_nonfinite_output": "correctness_preflight",
}
DEFAULT_STRATEGY_RESET_DIRECTIONS = (
    "work decomposition/query-tile ownership",
    "memory layout plus vectorized K/V pipeline",
    "register/online-softmax scheduling",
    "measurement diagnostic tied to a bottleneck",
)
FAMILY_STRATEGY_RESET_DIRECTIONS = {
    "async_copy_pipeline": (
        "work decomposition/query-tile ownership",
        "register/online-softmax scheduling",
        "measurement diagnostic for memory-vs-barrier cost",
    ),
    "shared_memory_staging": (
        "work decomposition/query-tile ownership",
        "register/online-softmax scheduling",
        "memory layout plus vectorized K/V pipeline",
    ),
    "thread_count_or_warp_mapping": (
        "memory layout plus vectorized K/V pipeline",
        "register/online-softmax scheduling",
        "source-verifiable work decomposition",
    ),
    "query_tile_work_mapping": (
        "memory layout plus vectorized K/V pipeline",
        "register/online-softmax scheduling",
        "measurement diagnostic for K/V reuse benefit",
    ),
    "synchronization_or_barrier": (
        "measurement diagnostic tied to a bottleneck",
        "memory layout plus vectorized K/V pipeline",
        "register/online-softmax scheduling",
    ),
    "wmma_contract_or_tile_shape": (
        "supported WMMA contract repair",
        "register/online-softmax scheduling",
        "source-verifiable work decomposition",
    ),
}
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
    advisories: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "patch_paths": self.patch_paths,
            "returncode": self.returncode,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "rejected_reason": self.rejected_reason,
            "advisories": list(self.advisories),
        }


@dataclass(frozen=True)
class VariationAttempt:
    decision: VariationDecision
    command_result: CommandResult
    started_at: str
    completed_at: str
    score_payload: dict[str, Any] | None = None
    patch_result: PatchResult | None = None
    materialized_patch: str | None = None

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
            "materialized_patch": self.materialized_patch,
        }


@dataclass(frozen=True)
class EvolutionStep:
    attempt: VariationAttempt
    gate_decision: GateDecision | None
    patch_cleanup_result: PatchResult | None = None
    repair_attempts: tuple[VariationAttempt, ...] = ()
    repair_cleanup_results: tuple[PatchResult, ...] = ()

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
            "repair_attempts": [attempt.as_dict() for attempt in self.repair_attempts],
            "repair_cleanup_results": [
                cleanup.as_dict() for cleanup in self.repair_cleanup_results
            ],
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
        "--recount",
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
        "--recount",
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
        "--recount",
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
        "--recount",
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
    promoted_preflight_classes: frozenset[str] | None = None,
) -> VariationAttempt:
    command = command_from_decision(decision, allowed_subcommands=allowed_subcommands)
    started_at = _utc_now()
    patch_result, materialized_patch = _maybe_apply_candidate_edit(
        decision,
        cwd=cwd,
        promoted_preflight_classes=promoted_preflight_classes or frozenset(),
    )
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
            materialized_patch=materialized_patch,
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
        materialized_patch=materialized_patch,
    )


def attempt_has_repairable_compile_failure(attempt: VariationAttempt) -> bool:
    if attempt.patch_result is None or not attempt.patch_result.ok:
        return False
    if attempt.command_result.ok or attempt.command_result.timed_out:
        return False
    return _command_or_detail_looks_like_compile_failure(
        attempt.decision.next_command,
        " ".join(attempt.command_result.command),
        _command_result_text(attempt.command_result),
    )


def attempt_has_repairable_transform_materialization_failure(attempt: VariationAttempt) -> bool:
    patch_result = attempt.patch_result
    if patch_result is None or patch_result.ok:
        return False
    if attempt.decision.candidate_transform is None:
        return False
    detail = " ".join(
        part
        for part in (
            patch_result.rejected_reason or "",
            patch_result.stderr_tail,
            attempt.command_result.stderr_tail,
        )
        if part
    )
    return _is_repairable_transform_materialization_error(detail)


def attempt_has_repairable_correctness_failure(attempt: VariationAttempt) -> bool:
    if attempt.patch_result is None or not attempt.patch_result.ok:
        return False
    if attempt.command_result.timed_out or not attempt.command_result.ok:
        return False
    if not isinstance(attempt.score_payload, dict):
        return False
    if attempt.score_payload.get("all_correct") is not False:
        return False
    if _classify_score_failure(attempt.score_payload) == "score_environment_error":
        return False
    has_edit_payload = (
        attempt.decision.candidate_transform is not None
        or bool((attempt.materialized_patch or attempt.decision.candidate_patch).strip())
    )
    return has_edit_payload


def attempt_has_repairable_worker_crash(attempt: VariationAttempt) -> bool:
    if attempt.patch_result is None or not attempt.patch_result.ok:
        return False
    if attempt.command_result.timed_out or attempt.command_result.ok:
        return False
    has_edit_payload = (
        attempt.decision.candidate_transform is not None
        or bool((attempt.materialized_patch or attempt.decision.candidate_patch).strip())
    )
    if not has_edit_payload:
        return False
    return _looks_like_worker_crash(
        attempt.command_result.returncode,
        _command_result_text(attempt.command_result).lower(),
    )


def compile_failure_class_for_attempt(attempt: VariationAttempt) -> str:
    return _classify_compile_failure(_command_result_text(attempt.command_result).lower())


def correctness_failure_class_for_attempt(attempt: VariationAttempt) -> str:
    if not isinstance(attempt.score_payload, dict):
        return "correctness_failed"
    return _classify_score_failure(attempt.score_payload)


def correctness_failure_summary_for_attempt(attempt: VariationAttempt) -> str:
    if not isinstance(attempt.score_payload, dict):
        return ""
    return _tail(_score_payload_error_text(attempt.score_payload))


def failure_class_for_step(step: EvolutionStep) -> str:
    return _step_failure_class(step.as_dict())


def finalize_attempt(
    lineage: Path,
    attempt: VariationAttempt,
    *,
    message: str = "evolve: accept candidate",
    source_root: Path | None = None,
    repair_attempts: tuple[VariationAttempt, ...] = (),
    repair_cleanup_results: tuple[PatchResult, ...] = (),
) -> EvolutionStep:
    gate_decision = None
    if attempt.score_payload is not None:
        source_files = _candidate_source_snapshot(source_root, attempt)
        candidate_patch = (
            _attempt_patch_text(attempt)
            if attempt.patch_result is not None and attempt.patch_result.ok
            else None
        )
        gate_decision = commit_score(
            lineage,
            attempt.score_payload,
            message=message,
            source_files=source_files,
            candidate_patch=candidate_patch,
        )
    return EvolutionStep(
        attempt=attempt,
        gate_decision=gate_decision,
        repair_attempts=repair_attempts,
        repair_cleanup_results=repair_cleanup_results,
    )


def planning_failure_step(
    error: Exception,
    *,
    repair_attempts: tuple[VariationAttempt, ...] = (),
    repair_cleanup_results: tuple[PatchResult, ...] = (),
) -> EvolutionStep:
    timestamp = _utc_now()
    decision = VariationDecision(
        hypothesis="agent planning failed validation",
        files_to_inspect=[],
        candidate_edit="No edit; planner returned invalid decision.",
        candidate_patch="",
        expected_effect="No candidate command was executed.",
        risk=f"{type(error).__name__}: {error}",
        next_command="avo env",
    )
    attempt = VariationAttempt(
        decision=decision,
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "agent-plan"],
            returncode=None,
            timed_out=False,
            stdout_tail="",
            stderr_tail=_tail(f"agent planning failed validation: {type(error).__name__}: {error}"),
        ),
        started_at=timestamp,
        completed_at=timestamp,
    )
    return EvolutionStep(
        attempt=attempt,
        gate_decision=None,
        repair_attempts=repair_attempts,
        repair_cleanup_results=repair_cleanup_results,
    )


def cleanup_rejected_candidate_patch(step: EvolutionStep, *, cwd: Path) -> EvolutionStep:
    if step.accepted:
        return step
    patch_result = step.attempt.patch_result
    if patch_result is None or not patch_result.ok:
        return step
    cleanup_result = revert_candidate_patch(_attempt_patch_text(step.attempt), cwd=cwd)
    if cleanup_result.ok:
        cleanup_result = _verify_rejected_patch_cleanup(cwd, cleanup_result)
    return EvolutionStep(
        attempt=step.attempt,
        gate_decision=step.gate_decision,
        patch_cleanup_result=cleanup_result,
        repair_attempts=step.repair_attempts,
        repair_cleanup_results=step.repair_cleanup_results,
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


def _attempt_patch_text(attempt: VariationAttempt) -> str:
    return attempt.materialized_patch or attempt.decision.candidate_patch


def summarize_attempt_history(
    directory: Path | None,
    *,
    limit: int = DEFAULT_ATTEMPT_HISTORY_LIMIT,
    current_best_geomean: float | None = None,
) -> str:
    if directory is None or limit <= 0 or not directory.exists():
        return ""
    paths = _load_step_payloads(directory)
    records = []
    payloads = []
    for path, payload in paths[-limit:]:
        payloads.append(payload)
        records.append(
            _summarize_step_payload(
                path.name,
                payload,
                current_best_geomean=current_best_geomean,
            )
        )
    if not records:
        return _summarize_promoted_preflight_tracks(directory)
    summary = "Recent attempts, oldest to newest:\n" + "\n".join(
        f"- {record}" for record in records
    )
    supervisor_signal = _summarize_supervisor_signal(payloads)
    if supervisor_signal:
        summary = f"{summary}\n{supervisor_signal}"
    family_signal = _summarize_transform_family_signal(payloads)
    if family_signal:
        summary = f"{summary}\n{family_signal}"
    strategy_signal = _summarize_strategy_reset_signal(
        payloads,
        supervisor_signal=supervisor_signal,
        family_signal=family_signal,
    )
    if strategy_signal:
        summary = f"{summary}\n{strategy_signal}"
    followup_signal = _summarize_followup_signal(payloads)
    if followup_signal:
        summary = f"{summary}\n{followup_signal}"
    stale_signal = _summarize_stale_accepted_signal(
        payloads,
        current_best_geomean=current_best_geomean,
    )
    if stale_signal:
        summary = f"{summary}\n{stale_signal}"
    promoted_summary = _summarize_promoted_preflight_tracks(directory)
    if promoted_summary:
        summary = f"{summary}\n{promoted_summary}"
    return summary


def update_promoted_preflight_tracks(
    directory: Path | None,
    *,
    threshold: int = SUPERVISOR_REPEAT_THRESHOLD,
) -> dict[str, Any]:
    state = _load_promoted_preflight_state(directory)
    if directory is None or threshold <= 0:
        return state
    payloads = [payload for _, payload in _load_step_payloads(directory)]
    recurring_classes = _recurring_unaccepted_failure_classes(payloads, threshold=threshold)
    promotable_classes = {
        failure_class: (count, track_names)
        for failure_class, count in recurring_classes.items()
        if failure_class in PROMOTABLE_FAILURE_CLASS_TRACKS
        if (track_names := _concrete_promoted_preflight_track_names(failure_class))
    }
    if not promotable_classes:
        return state
    tracks = _state_tracks(state)
    updated_at = _utc_now()
    for failure_class, (count, track_names) in sorted(promotable_classes.items()):
        existing = tracks.get(failure_class)
        entry = {
            "active": True,
            "failure_class": failure_class,
            "track": PROMOTABLE_FAILURE_CLASS_TRACKS[failure_class],
            "track_names": list(track_names),
            "threshold": threshold,
            "recent_count": count,
            "updated_at": updated_at,
        }
        if isinstance(existing, dict):
            entry["promoted_at"] = existing.get("promoted_at") or entry["updated_at"]
        else:
            entry["promoted_at"] = entry["updated_at"]
        tracks[failure_class] = entry
    state = {"version": 1, "tracks": tracks}
    _write_promoted_preflight_state(directory, state)
    return state


def _concrete_promoted_preflight_track_names(failure_class: str) -> tuple[str, ...]:
    return promoted_preflight_track_names_for_classes(frozenset({failure_class}))


def validate_decision_against_attempt_history(
    decision: VariationDecision,
    directory: Path | None,
    *,
    extra_payloads: tuple[dict[str, Any], ...] = (),
) -> None:
    payloads = []
    if directory is not None and directory.exists():
        payloads.extend(payload for _, payload in _load_step_payloads(directory))
    payloads.extend(extra_payloads)
    if not payloads:
        return
    if (
        decision.candidate_transform is not None
        and _decision_subcommand(decision) == "compile"
        and _has_successful_compile_only_transform(payloads, decision.candidate_transform)
    ):
        raise ValueError(
            "next_command repeats a successful compile-only candidate_transform; score the "
            "same structured transform on a validation workload or choose a materially "
            "different transform family"
        )
    if (
        decision.candidate_transform is not None
        and _has_scored_unaccepted_transform(payloads, decision.candidate_transform)
    ):
        raise ValueError(
            "candidate_transform repeats a previously scored unaccepted transform; choose a "
            "materially different structured transform or run a diagnostic that changes the "
            "next search direction"
        )
    pending_transform = _pending_compile_only_transform(payloads)
    if pending_transform is None:
        return
    subcommand = _decision_subcommand(decision)
    if (
        subcommand == "score"
        and decision.candidate_transform == pending_transform
        and not decision.candidate_patch.strip()
    ):
        return
    if subcommand == "score" and decision.candidate_transform is None:
        raise ValueError(
            "next_command scores without the pending compile-only candidate_transform; "
            "include the exact candidate_transform JSON from the follow-up signal"
        )
    raise ValueError(
        "pending compile-only candidate_transform must be scored before compiling or "
        "scoring a different transform"
    )


def pending_compile_only_transform(directory: Path | None) -> dict[str, Any] | None:
    if directory is None or not directory.exists():
        return None
    return _pending_compile_only_transform(
        [payload for _, payload in _load_step_payloads(directory)]
    )


def load_promoted_preflight_classes(directory: Path | None) -> frozenset[str]:
    state = _load_promoted_preflight_state(directory)
    return frozenset(
        failure_class
        for failure_class, entry in _state_tracks(state).items()
        if isinstance(entry, dict) and entry.get("active") is True
    )


def _load_step_payloads(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    payloads = []
    for path in (path for path in directory.glob("*.json") if path.is_file()):
        if path.name == PROMOTED_PREFLIGHT_TRACKS_FILENAME:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and _is_step_payload(payload):
            payloads.append((path, payload))
    return sorted(payloads, key=lambda item: _step_payload_order_key(*item))


def _step_payload_order_key(path: Path, payload: dict[str, Any]) -> tuple[datetime, str]:
    timestamp = _step_payload_timestamp(payload)
    if timestamp is not None:
        return timestamp, path.name
    try:
        fallback = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        fallback = datetime.min.replace(tzinfo=UTC)
    return fallback, path.name


def _step_payload_timestamp(payload: dict[str, Any]) -> datetime | None:
    attempt = payload.get("attempt") if isinstance(payload.get("attempt"), dict) else {}
    for key in ("completed_at", "started_at"):
        parsed = _parse_step_timestamp(str(attempt.get(key) or ""))
        if parsed is not None:
            return parsed
    return None


def _parse_step_timestamp(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _promoted_preflight_state_path(directory: Path | None) -> Path | None:
    if directory is None:
        return None
    return directory / PROMOTED_PREFLIGHT_TRACKS_FILENAME


def _load_promoted_preflight_state(directory: Path | None) -> dict[str, Any]:
    path = _promoted_preflight_state_path(directory)
    if path is None or not path.exists():
        return {"version": 1, "tracks": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "tracks": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "tracks": {}}
    return {"version": 1, "tracks": _state_tracks(payload)}


def _write_promoted_preflight_state(directory: Path, state: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PROMOTED_PREFLIGHT_TRACKS_FILENAME
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _state_tracks(state: dict[str, Any]) -> dict[str, Any]:
    tracks = state.get("tracks")
    return dict(tracks) if isinstance(tracks, dict) else {}


def _summarize_promoted_preflight_tracks(directory: Path | None) -> str:
    state = _load_promoted_preflight_state(directory)
    entries = [
        entry
        for entry in _state_tracks(state).values()
        if isinstance(entry, dict) and entry.get("active") is True
    ]
    if not entries:
        return ""
    lines = ["Active hard preflight tracks:"]
    for entry in sorted(entries, key=lambda item: str(item.get("failure_class") or "")):
        failure_class = str(entry.get("failure_class") or "unknown")
        track = str(entry.get("track") or "unknown")
        track_names = _entry_track_names(entry)
        concrete = f"; checks={','.join(track_names)}" if track_names else ""
        lines.append(
            f"- class={failure_class}; track={track}{concrete}; "
            "promoted from recurring attempts"
        )
    return "\n".join(lines)


def _entry_track_names(entry: dict[str, Any]) -> tuple[str, ...]:
    raw_track_names = entry.get("track_names")
    if isinstance(raw_track_names, list):
        names = tuple(str(name) for name in raw_track_names if isinstance(name, str))
        if names:
            return tuple(sorted(names))
    failure_class = entry.get("failure_class")
    if isinstance(failure_class, str):
        return promoted_preflight_track_names_for_classes(frozenset({failure_class}))
    return ()


def _is_step_payload(payload: dict[str, Any]) -> bool:
    attempt = payload.get("attempt")
    if not isinstance(attempt, dict):
        return False
    return isinstance(attempt.get("decision"), dict) and isinstance(
        attempt.get("command_result"),
        dict,
    )


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


def _summarize_supervisor_signal(payloads: list[dict[str, Any]]) -> str:
    repeated = _repeated_unaccepted_fingerprint(payloads, threshold=SUPERVISOR_REPEAT_THRESHOLD)
    recurring_classes = _recurring_unaccepted_failure_classes(
        payloads,
        threshold=SUPERVISOR_REPEAT_THRESHOLD,
    )
    if repeated is not None:
        message = (
            "Supervisor signal: the last "
            f"{SUPERVISOR_REPEAT_THRESHOLD} attempts share command/edit fingerprint "
            f"{repeated} and were not accepted. Choose a materially different optimization "
            "direction or diagnostic before repeating it."
        )
        repeated_class = _repeated_unaccepted_failure_class(
            payloads,
            threshold=SUPERVISOR_REPEAT_THRESHOLD,
        )
        if repeated_class is not None:
            if _concrete_promoted_preflight_track_names(repeated_class):
                message = (
                    f"{message} Failure class {repeated_class!r} also recurred and is "
                    "eligible for hard preflight promotion; choose a different transform "
                    "family."
                )
            else:
                message = (
                    f"{message} Failure class {repeated_class!r} also recurred, but no "
                    "concrete hard preflight track exists for it; use the class feedback "
                    "before repeating this transform family."
                )
        return message
    repeated_class = _repeated_unaccepted_failure_class(
        payloads,
        threshold=SUPERVISOR_REPEAT_THRESHOLD,
    )
    if repeated_class is not None:
        if not _concrete_promoted_preflight_track_names(repeated_class):
            return (
                "Supervisor signal: the last "
                f"{SUPERVISOR_REPEAT_THRESHOLD} attempts share failure class "
                f"{repeated_class!r}. No concrete hard preflight track exists for this "
                "class; use the class feedback and choose a different structured "
                "transform family before repeating it."
            )
        return (
            "Supervisor signal: the last "
            f"{SUPERVISOR_REPEAT_THRESHOLD} attempts share failure class "
            f"{repeated_class!r}. This class is eligible for hard preflight "
            "promotion; choose a different structured transform family before "
            "repeating it."
        )
    if recurring_classes:
        recurring = ", ".join(
            f"{failure_class}(count={count})"
            for failure_class, count in sorted(recurring_classes.items())
        )
        return (
            "Supervisor signal: recent unaccepted attempts include recurring failure "
            f"classes: {recurring}. The attempt-memory updater promotes eligible "
            "classes with concrete checks to hard preflight tracks; choose a different "
            "structured transform family before repeating them."
        )
    if _unaccepted_tail_count(payloads) >= SUPERVISOR_EXHAUSTION_THRESHOLD:
        return (
            "Supervisor signal: the last "
            f"{SUPERVISOR_EXHAUSTION_THRESHOLD} attempts produced no accepted candidate. "
            "Review the lineage and recent failures, then reset strategy toward a different "
            "Ampere optimization direction."
        )
    return ""


def _summarize_transform_family_signal(payloads: list[dict[str, Any]]) -> str:
    recurring_families = _recurring_unaccepted_transform_families(
        payloads,
        threshold=SUPERVISOR_REPEAT_THRESHOLD,
    )
    if not recurring_families:
        return ""
    recurring = ", ".join(
        f"{family}(count={count})"
        for family, count in sorted(recurring_families.items())
    )
    return (
        "Semantic-family signal: recent unaccepted attempts include recurring "
        f"transform families: {recurring}. Choose a materially different "
        "optimization family unless the next transform changes the dataflow, "
        "pipeline overlap, or validation contract in a way the prior family did not."
    )


def _summarize_strategy_reset_signal(
    payloads: list[dict[str, Any]],
    *,
    supervisor_signal: str,
    family_signal: str,
) -> str:
    if not supervisor_signal and not family_signal:
        return ""
    families = _recent_unaccepted_transform_families(payloads)
    directions = _strategy_reset_directions_for_families(families)
    direction_text = "; ".join(directions)
    avoid_text = ""
    if families:
        avoid_text = " Recent families to avoid repeating unchanged: " + ", ".join(
            sorted(families)
        )
        avoid_text += "."
    return (
        "Strategy reset candidates: "
        f"{direction_text}. Choose one only if it can be expressed as a scoped, "
        "source-verifiable candidate_transform or as a no-edit diagnostic tied to a "
        f"specific bottleneck.{avoid_text}"
    )


def _recent_unaccepted_transform_families(payloads: list[dict[str, Any]]) -> frozenset[str]:
    families: set[str] = set()
    for payload in _unaccepted_tail(payloads):
        families.update(_payload_transform_families(payload))
    return frozenset(family for family in families if family not in IGNORED_TRANSFORM_FAMILIES)


def _strategy_reset_directions_for_families(families: frozenset[str]) -> tuple[str, ...]:
    directions: list[str] = []
    for family in sorted(families):
        directions.extend(FAMILY_STRATEGY_RESET_DIRECTIONS.get(family, ()))
    if not directions:
        directions.extend(DEFAULT_STRATEGY_RESET_DIRECTIONS)
    deduped = list(dict.fromkeys(directions))
    for fallback in DEFAULT_STRATEGY_RESET_DIRECTIONS:
        if len(deduped) >= 3:
            break
        if fallback not in deduped:
            deduped.append(fallback)
    return tuple(deduped[:3])


def _summarize_followup_signal(payloads: list[dict[str, Any]]) -> str:
    if not payloads:
        return ""
    pending_transform = _pending_compile_only_transform(payloads)
    if pending_transform is not None:
        transform_json = json.dumps(pending_transform, sort_keys=True, separators=(",", ":"))
        return (
            "Follow-up signal: the latest semantic structured transform compiled successfully but "
            "has not been scored. Do not repeat the compile-only check; score the same "
            "candidate_transform on the next validation workload, or choose a materially "
            "different transform family. Compile-only patches are cleaned up before "
            "follow-up scoring, so a no_edit score would score the unmodified seed; include "
            "this candidate_transform again with edit_mode=transform and candidate_patch=\"\". "
            "For the score command, candidate_edit may summarize the prior compiled edit, "
            "but the executable edit payload must still be the exact candidate_transform JSON; "
            "do not return a prose-only score decision. "
            "Exact pending candidate_transform JSON: "
            f"{transform_json}"
        )
    materialization_failure = _latest_transform_materialization_failure(payloads)
    if materialization_failure is None:
        return ""
    rejected_reason, transform = materialization_failure
    transform_json = json.dumps(transform, sort_keys=True, separators=(",", ":"))
    scope_hint = _transform_scope_repair_hint(transform)
    scope_hint_text = f" {scope_hint}" if scope_hint else ""
    return (
        "Follow-up signal: the latest semantic structured transform failed materialization "
        "before compile. Keep the same semantic move only if it remains useful, but repair "
        "the candidate_transform anchors/matches with larger unique surrounding-code "
        "snippets; do not restate the CUDA edit in prose without candidate_transform."
        f"{scope_hint_text} "
        f"Materialization error: {_shorten(rejected_reason, 500)}. "
        "Rejected candidate_transform JSON to repair, not reuse unchanged: "
        f"{transform_json}"
    )


def _summarize_stale_accepted_signal(
    payloads: list[dict[str, Any]],
    *,
    current_best_geomean: float | None,
) -> str:
    if current_best_geomean is None or current_best_geomean <= 0.0:
        return ""
    stale_count = sum(
        1
        for payload in payloads
        if _is_stale_accepted_payload(
            payload,
            current_best_geomean=current_best_geomean,
        )
    )
    if stale_count == 0:
        return ""
    return (
        "Lineage correction: recent attempt history contains "
        f"{stale_count} accepted score(s) above the current lineage best "
        f"{current_best_geomean:.12g}. Treat entries marked class=stale_accepted as "
        "reverted or noisy historical acceptances, not as current best state."
    )


def _latest_transform_materialization_failure(
    payloads: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]] | None:
    for payload in reversed(payloads):
        if _step_payload_accepted(payload) or _payload_has_score_payload(payload):
            return None
        if _successful_compile_only_transform(payload) is not None:
            return None
        attempt = payload.get("attempt") if isinstance(payload.get("attempt"), dict) else {}
        patch_result = attempt.get("patch_result")
        if not isinstance(patch_result, dict) or patch_result.get("ok") is not False:
            continue
        rejected_reason = str(patch_result.get("rejected_reason") or "")
        if not _is_repairable_transform_materialization_error(rejected_reason):
            continue
        transform = _decision_transform(payload)
        if transform is None or not _transform_has_semantic_step(transform):
            continue
        return rejected_reason, transform
    return None


def _is_repairable_transform_materialization_error(rejected_reason: str) -> bool:
    text = rejected_reason.lower()
    return (
        "candidate transform rejected" in text
        and "expected exactly one" in text
        and ("anchor" in text or "match" in text)
    )


def _transform_scope_repair_hint(transform: dict[str, Any]) -> str:
    for step in _candidate_transform_steps(transform):
        op = str(step.get("op") or "")
        if op not in {"insert_before_once", "insert_after_once"}:
            continue
        text = str(step.get("text") or "")
        anchor = str(step.get("anchor") or "")
        if "key_start" in text and "key_start" not in anchor:
            return (
                "Scope hint: inserted text references key_start, so choose an anchor inside "
                "the key_start loop and include the loop header or surrounding body context."
            )
    return ""


def _successful_compile_only_transform(payload: dict[str, Any]) -> dict[str, Any] | None:
    if _step_failure_class(payload) != "compile_only_diagnostic":
        return None
    cleanup_result = payload.get("patch_cleanup_result")
    if isinstance(cleanup_result, dict) and cleanup_result.get("ok") is False:
        return None
    attempt = payload.get("attempt") if isinstance(payload.get("attempt"), dict) else {}
    patch_result = attempt.get("patch_result")
    if not isinstance(patch_result, dict) or patch_result.get("ok") is not True:
        return None
    decision = attempt.get("decision") if isinstance(attempt.get("decision"), dict) else {}
    transform = decision.get("candidate_transform")
    if not isinstance(transform, dict) or not _transform_has_semantic_step(transform):
        return None
    return transform


def _has_successful_compile_only_transform(
    payloads: list[dict[str, Any]],
    transform: dict[str, Any],
) -> bool:
    transform_identity = _transform_identity(transform)
    invalidated_identities = _preflight_rejected_transform_identities(payloads)
    if transform_identity in invalidated_identities:
        return False
    return any(
        _transform_identity(previous) == transform_identity
        for previous in (
            _successful_compile_only_transform(payload)
            for payload in payloads
        )
        if previous is not None
    )


def _has_scored_unaccepted_transform(
    payloads: list[dict[str, Any]],
    transform: dict[str, Any],
) -> bool:
    transform_identity = _transform_identity(transform)
    for payload in reversed(payloads):
        if not _payload_has_score_for_transform(payload, transform_identity):
            continue
        if _step_payload_accepted(payload):
            return False
        if _step_failure_class(payload) == "score_environment_error":
            continue
        return True
    return False


def _payload_has_score_for_transform(payload: dict[str, Any], transform_identity: str) -> bool:
    attempt = payload.get("attempt") if isinstance(payload.get("attempt"), dict) else {}
    if _attempt_has_score_for_transform(attempt, transform_identity):
        return True
    repair_attempts = payload.get("repair_attempts")
    if not isinstance(repair_attempts, list):
        return False
    return any(
        isinstance(attempt, dict)
        and _attempt_has_score_for_transform(attempt, transform_identity)
        for attempt in repair_attempts
    )


def _attempt_has_score_for_transform(attempt: dict[str, Any], transform_identity: str) -> bool:
    if not isinstance(attempt.get("score_payload"), dict):
        return False
    decision = attempt.get("decision") if isinstance(attempt.get("decision"), dict) else {}
    previous_transform = decision.get("candidate_transform")
    return (
        isinstance(previous_transform, dict)
        and _transform_identity(previous_transform) == transform_identity
    )


def _pending_compile_only_transform(payloads: list[dict[str, Any]]) -> dict[str, Any] | None:
    invalidated_identities: set[str] = set()
    for payload in reversed(payloads):
        rejected_identity = _preflight_rejected_transform_identity(payload)
        if rejected_identity is not None:
            invalidated_identities.add(rejected_identity)
            continue
        transform = _successful_compile_only_transform(payload)
        if transform is None:
            if _payload_has_score_payload(payload):
                return None
            continue
        if _transform_identity(transform) in invalidated_identities:
            continue
        return transform
    return None


def _preflight_rejected_transform_identities(payloads: list[dict[str, Any]]) -> set[str]:
    identities: set[str] = set()
    for payload in payloads:
        identity = _preflight_rejected_transform_identity(payload)
        if identity is not None:
            identities.add(identity)
    return identities


def _preflight_rejected_transform_identity(payload: dict[str, Any]) -> str | None:
    transform = _decision_transform(payload)
    if transform is None:
        return None
    attempt = payload.get("attempt") if isinstance(payload.get("attempt"), dict) else {}
    patch_result = attempt.get("patch_result")
    if not isinstance(patch_result, dict) or patch_result.get("ok") is not False:
        return None
    rejected_reason = str(patch_result.get("rejected_reason") or "")
    if "candidate structural preflight rejected" not in rejected_reason:
        return None
    return _transform_identity(transform)


def _decision_transform(payload: dict[str, Any]) -> dict[str, Any] | None:
    attempt = payload.get("attempt") if isinstance(payload.get("attempt"), dict) else {}
    decision = attempt.get("decision") if isinstance(attempt.get("decision"), dict) else {}
    transform = decision.get("candidate_transform")
    return transform if isinstance(transform, dict) else None


def _transform_has_semantic_step(transform: dict[str, Any]) -> bool:
    steps = transform.get("steps") if transform.get("op") == "batch" else [transform]
    if not isinstance(steps, list):
        return False
    return any(
        isinstance(step, dict) and str(step.get("op") or "") not in SUPPORT_ONLY_TRANSFORM_OPS
        for step in steps
    )


def _step_score_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    attempt = payload.get("attempt") if isinstance(payload.get("attempt"), dict) else {}
    score_payload = attempt.get("score_payload")
    return score_payload if isinstance(score_payload, dict) else None


def _payload_has_score_payload(payload: dict[str, Any]) -> bool:
    if isinstance(_step_score_payload(payload), dict):
        return True
    repair_attempts = payload.get("repair_attempts")
    if not isinstance(repair_attempts, list):
        return False
    return any(
        isinstance(attempt, dict) and isinstance(attempt.get("score_payload"), dict)
        for attempt in repair_attempts
    )


def _transform_identity(transform: dict[str, Any]) -> str:
    return json.dumps(transform, sort_keys=True, separators=(",", ":"))


def _decision_subcommand(decision: VariationDecision) -> str:
    try:
        parts = shlex.split(decision.next_command)
    except ValueError:
        return ""
    if len(parts) < 2 or parts[0] != "avo":
        return ""
    return parts[1]


def _repeated_unaccepted_failure_class(
    payloads: list[dict[str, Any]],
    *,
    threshold: int,
) -> str | None:
    if len(payloads) < threshold:
        return None
    window = payloads[-threshold:]
    if any(_step_payload_accepted(payload) for payload in window):
        return None
    classes = {_step_failure_class(payload) for payload in window}
    if len(classes) == 1:
        failure_class = next(iter(classes))
        if failure_class not in IGNORED_FAILURE_CLASSES:
            return failure_class
    return None


def _recurring_unaccepted_failure_classes(
    payloads: list[dict[str, Any]],
    *,
    threshold: int,
) -> dict[str, int]:
    if threshold <= 0:
        return {}
    counts: dict[str, int] = {}
    for payload in _unaccepted_tail(payloads):
        failure_class = _step_failure_class(payload)
        if failure_class in IGNORED_FAILURE_CLASSES:
            continue
        counts[failure_class] = counts.get(failure_class, 0) + 1
    return {
        failure_class: count
        for failure_class, count in counts.items()
        if count >= threshold
    }


def _recurring_unaccepted_transform_families(
    payloads: list[dict[str, Any]],
    *,
    threshold: int,
) -> dict[str, int]:
    if threshold <= 0:
        return {}
    counts: dict[str, int] = {}
    for payload in _unaccepted_tail(payloads):
        for family in _payload_transform_families(payload):
            if family in IGNORED_TRANSFORM_FAMILIES:
                continue
            counts[family] = counts.get(family, 0) + 1
    return {family: count for family, count in counts.items() if count >= threshold}


IGNORED_TRANSFORM_FAMILIES = frozenset(
    {
        "diagnostic_or_planning",
        "unknown_transform",
    }
)

ASYNC_COPY_COMPILE_MARKERS = (
    "cp.async",
    "__pipeline",
    "cuda::memcpy_async",
    "memcpy_async",
    "async-copy",
    "async copy",
)


def _payload_transform_families(payload: dict[str, Any]) -> frozenset[str]:
    families = {_step_transform_family(payload)}
    repair_attempts = payload.get("repair_attempts")
    if isinstance(repair_attempts, list):
        families.update(
            _step_transform_family({"attempt": attempt})
            for attempt in repair_attempts
            if isinstance(attempt, dict)
        )
    return frozenset(families)


def _step_transform_family(payload: dict[str, Any]) -> str:
    attempt = payload.get("attempt") if isinstance(payload.get("attempt"), dict) else {}
    decision = attempt.get("decision") if isinstance(attempt.get("decision"), dict) else {}
    candidate_transform = decision.get("candidate_transform")
    candidate_patch = str(decision.get("candidate_patch") or "")
    materialized_patch = str(attempt.get("materialized_patch") or "")
    has_edit_payload = (
        isinstance(candidate_transform, dict)
        or bool(candidate_patch.strip())
        or bool(materialized_patch.strip())
    )
    if not has_edit_payload:
        return "diagnostic_or_planning"
    text = _payload_transform_family_text(payload)
    if _payload_transform_sets_constexpr(payload, {"kThreads", "kWarps"}):
        return "thread_count_or_warp_mapping"
    if (
        "kquerytilesperblock" in text
        or "query tiles per block" in text
        or "multi-query" in text
        or "multi query" in text
    ):
        return "query_tile_work_mapping"
    if any(marker in text for marker in ("cp.async", "__pipeline", "async copy", "async-copy")):
        return "async_copy_pipeline"
    if "__syncwarp" in text:
        return "synchronization_or_barrier"
    if re.search(
        r"\b(?:thread\s+count|threads?\s+per\s+block|warp\s+only|warp-only|"
        r"warp\s+mapping|kthreads|kwarps)\b",
        text,
    ):
        return "thread_count_or_warp_mapping"
    if "shared" in text and any(
        marker in text for marker in ("stage", "staging", "tile", "buffer", "smem")
    ):
        return "shared_memory_staging"
    if "__syncthreads" in text or "barrier" in text:
        return "synchronization_or_barrier"
    if "q_frags" in text or ("register" in text and "reuse" in text):
        return "register_reuse"
    if "wmma" in text and any(marker in text for marker in ("fragment", "shape", "tile")):
        return "wmma_contract_or_tile_shape"
    if "softmax" in text or "row_max" in text or "row_sum" in text or "rescale" in text:
        return "online_softmax_or_rescale"
    if "unroll" in text or "#pragma" in text:
        return "scheduler_or_unroll"
    return "unknown_transform"


def _payload_transform_sets_constexpr(payload: dict[str, Any], names: set[str]) -> bool:
    attempt = payload.get("attempt") if isinstance(payload.get("attempt"), dict) else {}
    decision = attempt.get("decision") if isinstance(attempt.get("decision"), dict) else {}
    transform = decision.get("candidate_transform")
    if not isinstance(transform, dict):
        return False
    return any(
        str(step.get("op") or "") == "set_constexpr_int" and str(step.get("name") or "") in names
        for step in _candidate_transform_steps(transform)
        if isinstance(step, dict)
    )


def _payload_transform_family_text(payload: dict[str, Any]) -> str:
    attempt = payload.get("attempt") if isinstance(payload.get("attempt"), dict) else {}
    decision = attempt.get("decision") if isinstance(attempt.get("decision"), dict) else {}
    parts = [
        str(decision.get(key) or "")
        for key in ("candidate_edit",)
    ]
    transform = decision.get("candidate_transform")
    if isinstance(transform, dict):
        parts.append(json.dumps(transform, sort_keys=True))
    parts.append(str(decision.get("candidate_patch") or ""))
    parts.append(str(attempt.get("materialized_patch") or ""))
    return " ".join(parts).lower()


def _unaccepted_tail(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tail: list[dict[str, Any]] = []
    for payload in reversed(payloads):
        if _step_payload_accepted(payload):
            break
        tail.append(payload)
    tail.reverse()
    return tail


def _repeated_unaccepted_fingerprint(
    payloads: list[dict[str, Any]],
    *,
    threshold: int,
) -> str | None:
    if len(payloads) < threshold:
        return None
    window = payloads[-threshold:]
    if any(_step_payload_accepted(payload) for payload in window):
        return None
    if all(_step_failure_class(payload) == "planner_provider_error" for payload in window):
        return None
    fingerprints = {_step_payload_fingerprint(payload) for payload in window}
    if len(fingerprints) == 1:
        return next(iter(fingerprints))
    return None


def _unaccepted_tail_count(payloads: list[dict[str, Any]]) -> int:
    count = 0
    for payload in reversed(payloads):
        if _step_payload_accepted(payload):
            break
        count += 1
    return count


def _step_payload_accepted(payload: dict[str, Any]) -> bool:
    gate = payload.get("gate_decision")
    return isinstance(gate, dict) and gate.get("accepted") is True


def _step_payload_fingerprint(payload: dict[str, Any]) -> str:
    attempt = payload.get("attempt")
    decision = attempt.get("decision") if isinstance(attempt, dict) else None
    if not isinstance(decision, dict):
        decision = {}
    components = {
        "candidate_patch": str(decision.get("candidate_patch") or "").strip(),
        "candidate_transform": decision.get("candidate_transform") or {},
        "files_to_inspect": _string_list(decision.get("files_to_inspect")),
        "next_command": _fingerprint_next_command(str(decision.get("next_command") or "")),
    }
    planning_detail = _planning_failure_detail(payload)
    if planning_detail:
        components["planning_failure_class"] = _classify_planning_failure(
            planning_detail.lower()
        )
        components["planning_failure_detail"] = _shorten(planning_detail, 260)
    encoded = json.dumps(components, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _planning_failure_detail(payload: dict[str, Any]) -> str:
    attempt = payload.get("attempt") if isinstance(payload.get("attempt"), dict) else {}
    command_result = (
        attempt.get("command_result") if isinstance(attempt.get("command_result"), dict) else {}
    )
    detail = _result_detail(command_result)
    if "agent planning failed validation" not in detail.lower():
        return ""
    decision = attempt.get("decision") if isinstance(attempt.get("decision"), dict) else {}
    risk = str(decision.get("risk") or "")
    return risk or detail


def _fingerprint_next_command(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        return " ".join(command.split())
    if len(parts) >= 2 and parts[:2] == ["avo", "compile"]:
        source = _command_option_value(parts, "--source")
        if source:
            return f"avo compile --source {source}"
        return "avo compile"
    return " ".join(parts)


def _command_option_value(parts: list[str], option: str) -> str:
    prefix = f"{option}="
    for index, part in enumerate(parts):
        if part == option and index + 1 < len(parts):
            return parts[index + 1]
        if part.startswith(prefix):
            return part[len(prefix) :]
    return ""


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(str(item) for item in value)


def _maybe_apply_candidate_edit(
    decision: VariationDecision,
    *,
    cwd: Path,
    promoted_preflight_classes: frozenset[str],
) -> tuple[PatchResult | None, str | None]:
    if decision.candidate_transform is not None:
        try:
            patch_text = materialize_candidate_transform(decision.candidate_transform, cwd=cwd)
            _preflight_materialized_candidate_patch(
                patch_text,
                allow_cuda_source_edits=True,
                promoted_preflight_classes=promoted_preflight_classes,
            )
        except ValueError as exc:
            return (
                PatchResult(
                    ok=False,
                    patch_paths=[],
                    returncode=None,
                    stdout_tail="",
                    stderr_tail="",
                    rejected_reason=f"candidate transform rejected: {exc}",
                ),
                None,
            )
        advisories = candidate_patch_structural_advisories(patch_text)
        return (
            _patch_result_with_advisories(
                apply_candidate_patch(patch_text, cwd=cwd),
                advisories,
            ),
            patch_text,
        )
    if not decision.candidate_patch.strip():
        return None, None
    try:
        _preflight_materialized_candidate_patch(
            decision.candidate_patch,
            allow_cuda_source_edits=False,
            promoted_preflight_classes=promoted_preflight_classes,
        )
    except ValueError as exc:
        return (
            PatchResult(
                ok=False,
                patch_paths=[],
                returncode=None,
                stdout_tail="",
                stderr_tail="",
                rejected_reason=f"candidate structural preflight rejected: {exc}",
            ),
            None,
        )
    advisories = candidate_patch_structural_advisories(decision.candidate_patch)
    return (
        _patch_result_with_advisories(
            apply_candidate_patch(decision.candidate_patch, cwd=cwd),
            advisories,
        ),
        decision.candidate_patch,
    )


def _patch_result_with_advisories(
    result: PatchResult,
    advisories: tuple[str, ...],
) -> PatchResult:
    if not advisories:
        return result
    return PatchResult(
        ok=result.ok,
        patch_paths=result.patch_paths,
        returncode=result.returncode,
        stdout_tail=result.stdout_tail,
        stderr_tail=result.stderr_tail,
        rejected_reason=result.rejected_reason,
        advisories=advisories,
    )


def _preflight_materialized_candidate_patch(
    patch_text: str,
    *,
    allow_cuda_source_edits: bool,
    promoted_preflight_classes: frozenset[str],
) -> None:
    try:
        validate_candidate_patch_structural_preflight(
            patch_text,
            allow_cuda_source_edits=allow_cuda_source_edits,
            promoted_preflight_classes=promoted_preflight_classes,
        )
    except ValueError as exc:
        promoted = ""
        match = re.search(r"classified as ([a-z_]+)", str(exc))
        if match and match.group(1) in promoted_preflight_classes:
            promoted = " promoted"
        raise ValueError(f"candidate structural{promoted} preflight rejected: {exc}") from exc


def materialize_candidate_transform(transform: dict[str, Any], *, cwd: Path) -> str:
    old_by_path: dict[str, str] = {}
    new_by_path: dict[str, str] = {}
    steps = _candidate_transform_steps(transform)
    is_batch = transform.get("op") == "batch"
    for step in steps:
        relative_path = _normalize_transform_path(step["path"])
        if relative_path not in old_by_path:
            source = cwd / relative_path
            if not source.is_file() or _has_symlink_component(cwd, source):
                raise ValueError(f"transform path is not a regular candidate file: {relative_path}")
            old_by_path[relative_path] = source.read_text(encoding="utf-8")
            new_by_path[relative_path] = old_by_path[relative_path]
        current = new_by_path[relative_path]
        updated = _apply_candidate_transform_step(step, current)
        if current == updated:
            if is_batch:
                continue
            raise ValueError(f"transform step produced no source change: {relative_path}")
        new_by_path[relative_path] = updated
    changed_paths = [
        path for path in sorted(old_by_path) if old_by_path[path] != new_by_path[path]
    ]
    if not changed_paths:
        raise ValueError("transform produced no source change")
    return "".join(
        _unified_diff_for_content(path, old_by_path[path], new_by_path[path])
        for path in changed_paths
    )


def _candidate_transform_steps(transform: dict[str, Any]) -> list[dict[str, Any]]:
    if transform.get("op") == "batch":
        steps = transform.get("steps")
        if steps is None and isinstance(transform.get("steps_json"), str):
            try:
                steps = json.loads(str(transform["steps_json"]))
            except json.JSONDecodeError as exc:
                raise ValueError(f"batch transform steps_json is invalid JSON: {exc}") from exc
        if not isinstance(steps, list) or not steps:
            raise ValueError("batch transform must contain steps")
        if not all(isinstance(step, dict) for step in steps):
            raise ValueError("batch transform steps must be objects")
        return steps
    return [transform]


def _apply_candidate_transform_step(step: dict[str, Any], content: str) -> str:
    op = str(step["op"])
    if op in {"replace_once", "replace_block_once"}:
        return _transform_replace_once(
            content,
            find=str(step["find"]),
            replacement=str(step["replace"]),
            op=op,
        )
    if op == "insert_before_once":
        return _transform_insert_once(
            content,
            anchor=str(step["anchor"]),
            text=str(step["text"]),
            before=True,
        )
    if op == "insert_after_once":
        return _transform_insert_once(
            content,
            anchor=str(step["anchor"]),
            text=str(step["text"]),
            before=False,
        )
    if op == "add_include":
        return _transform_add_include(content, header=str(step["header"]))
    if op == "set_constexpr_int":
        return _transform_set_constexpr_int(
            content,
            name=str(step["name"]),
            value=int(step["value"]),
        )
    if op == "add_int_to_python_set":
        return _transform_add_int_to_python_set(
            content,
            name=str(step["name"]),
            value=int(step["value"]),
        )
    raise ValueError(f"unsupported transform op: {op}")


def _normalize_transform_path(raw_path: Any) -> str:
    if not isinstance(raw_path, str):
        raise ValueError("transform path must be a string")
    normalized = _normalize_candidate_source_path(raw_path)
    if normalized is None:
        raise ValueError("transform path must be a repo-relative path under candidates/")
    if Path(normalized).suffix not in CANDIDATE_SOURCE_SUFFIXES:
        raise ValueError("transform path must reference a candidate source file")
    return normalized


def _transform_replace_once(
    content: str,
    *,
    find: str,
    replacement: str,
    op: str = "replace_once",
) -> str:
    count = content.count(find)
    if count != 1:
        raise ValueError(
            _transform_match_error(
                op,
                "match",
                count=count,
                content=content,
                needle=find,
            )
        )
    return content.replace(find, replacement, 1)


def _transform_insert_once(content: str, *, anchor: str, text: str, before: bool) -> str:
    count = content.count(anchor)
    if count != 1:
        raise ValueError(
            _transform_match_error(
                "insert transform",
                "anchor",
                count=count,
                content=content,
                needle=anchor,
            )
        )
    index = content.index(anchor)
    if not before:
        index += len(anchor)
    return f"{content[:index]}{_linewise_insert_text(content, index, text)}{content[index:]}"


def _linewise_insert_text(content: str, index: int, text: str) -> str:
    if not text:
        return text
    prefix = ""
    suffix = ""
    if index > 0 and not content[:index].endswith("\n") and not text.startswith("\n"):
        prefix = "\n"
    if index < len(content) and not content[index:].startswith("\n") and not text.endswith("\n"):
        suffix = "\n"
    return f"{prefix}{text}{suffix}"


def _transform_match_error(
    op: str,
    label: str,
    *,
    count: int,
    content: str,
    needle: str,
) -> str:
    line_numbers = _transform_match_line_numbers(content, needle)
    line_hint = ""
    if line_numbers:
        rendered = ", ".join(str(line) for line in line_numbers[:6])
        if len(line_numbers) > 6:
            rendered += ", ..."
        line_hint = f"; matching start lines: {rendered}"
    return (
        f"{op} expected exactly one {label}, found {count}{line_hint}. "
        "Use a larger unique anchor including surrounding code."
    )


def _transform_match_line_numbers(content: str, needle: str) -> list[int]:
    if not needle:
        return []
    line_numbers: list[int] = []
    start = 0
    while True:
        index = content.find(needle, start)
        if index < 0:
            break
        line_numbers.append(content.count("\n", 0, index) + 1)
        start = index + max(1, len(needle))
    return line_numbers


def _transform_add_include(content: str, *, header: str) -> str:
    include_line = f"#include {_normalize_include_header(header)}"
    if re.search(rf"(?m)^\s*{re.escape(include_line)}\s*$", content):
        return content
    matches = list(
        re.finditer(
            r"(?m)^#\s*include[^\S\n]+[<\"][^>\"]+[>\"][^\S\n]*(?:\n|$)",
            content,
        )
    )
    if not matches:
        return f"{include_line}\n{content}"
    insert_at = matches[-1].end()
    separator = "" if content[:insert_at].endswith("\n") else "\n"
    return f"{content[:insert_at]}{separator}{include_line}\n{content[insert_at:]}"


def _normalize_include_header(raw_header: str) -> str:
    header = raw_header.strip()
    if not header or any(char in header for char in "\r\n"):
        raise ValueError("add_include header must be a single non-empty header")
    if re.fullmatch(r"<[^<>\"\n]+>", header) or re.fullmatch(r'"[^"\n]+"', header):
        return header
    if not re.fullmatch(r"[A-Za-z0-9_./+-]+", header):
        raise ValueError("add_include header contains unsupported characters")
    return f"<{header}>"


def _transform_set_constexpr_int(content: str, *, name: str, value: int) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("set_constexpr_int name must be a C++ identifier")
    pattern = re.compile(
        rf"(?P<prefix>\bconstexpr\s+int\s+{re.escape(name)}\s*=\s*)"
        r"(?P<value>[-+]?\d+)(?P<suffix>\s*;)"
    )
    matches = list(pattern.finditer(content))
    if len(matches) != 1:
        raise ValueError(
            f"set_constexpr_int expected exactly one constexpr int {name}, found {len(matches)}"
        )
    return pattern.sub(rf"\g<prefix>{value}\g<suffix>", content, count=1)


def _transform_add_int_to_python_set(content: str, *, name: str, value: int) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("add_int_to_python_set name must be a Python identifier")
    pattern = re.compile(
        rf"(?m)^(?P<prefix>{re.escape(name)}\s*=\s*\{{)"
        r"(?P<body>[^}]*)"
        r"(?P<suffix>\})$"
    )
    matches = list(pattern.finditer(content))
    if len(matches) != 1:
        raise ValueError(
            f"add_int_to_python_set expected exactly one set assignment {name}, "
            f"found {len(matches)}"
        )
    match = matches[0]
    values = _python_int_set_values(match.group("body"))
    if value in values:
        return content
    values.append(value)
    body = ", ".join(str(item) for item in values)
    replacement = f"{match.group('prefix')}{body}{match.group('suffix')}"
    return content[: match.start()] + replacement + content[match.end() :]


def _python_int_set_values(body: str) -> list[int]:
    values: list[int] = []
    for raw_item in body.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise ValueError("add_int_to_python_set only supports integer set values") from exc
    return values


def _unified_diff_for_content(relative_path: str, old: str, new: str) -> str:
    diff_lines = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=f"a/{relative_path}",
        tofile=f"b/{relative_path}",
        lineterm="",
    )
    return f"diff --git a/{relative_path} b/{relative_path}\n" + "\n".join(diff_lines) + "\n"


def _candidate_source_snapshot(
    source_root: Path | None,
    attempt: VariationAttempt,
) -> dict[str, str] | None:
    if source_root is None:
        return None
    snapshot_paths: set[str] = set()
    if attempt.patch_result is not None and attempt.patch_result.ok:
        snapshot_paths.update(attempt.patch_result.patch_paths)
    snapshot_paths.update(_scored_candidate_source_paths(source_root, attempt.score_payload))
    snapshot_paths.update(_candidate_python_dependency_source_paths(source_root, snapshot_paths))
    if not snapshot_paths:
        return None
    return {
        path: (source_root / path).read_text(encoding="utf-8")
        for path in sorted(snapshot_paths)
        if _is_snapshot_source_file(source_root, path)
    }


def _scored_candidate_source_paths(
    source_root: Path,
    score_payload: dict[str, Any] | None,
) -> set[str]:
    if score_payload is None:
        return set()
    candidate_path = _candidate_path_from_score(source_root, score_payload.get("candidate_path"))
    if candidate_path is None:
        return set()

    paths = {candidate_path}
    candidate = PurePosixPath(candidate_path)
    for companion in _candidate_companion_directories(candidate):
        paths.update(_candidate_source_paths_under(source_root, companion))
    paths.update(_declared_candidate_source_paths(source_root, score_payload))
    return paths


def _declared_candidate_source_paths(
    source_root: Path,
    score_payload: dict[str, Any],
) -> set[str]:
    declared = score_payload.get("candidate_source_files")
    if not isinstance(declared, list):
        return set()
    paths: set[str] = set()
    for raw_path in declared:
        normalized = _candidate_path_from_score(source_root, raw_path)
        if normalized is not None and _is_snapshot_source_file(source_root, normalized):
            paths.add(normalized)
    return paths


def _candidate_python_dependency_source_paths(
    source_root: Path,
    initial_paths: set[str],
) -> set[str]:
    dependencies: set[str] = set()
    pending = [path for path in sorted(initial_paths) if path.endswith(".py")]
    seen: set[str] = set()
    while pending and len(seen) < 64:
        relative_path = pending.pop(0)
        if relative_path in seen or not _is_snapshot_source_file(source_root, relative_path):
            continue
        seen.add(relative_path)
        for dependency in sorted(_candidate_python_import_paths(source_root, relative_path)):
            if dependency in dependencies:
                continue
            dependencies.add(dependency)
            if dependency.endswith(".py") and dependency not in seen:
                pending.append(dependency)
    return dependencies


def _candidate_python_import_paths(source_root: Path, relative_path: str) -> set[str]:
    path = source_root / relative_path
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    paths: set[str] = set()
    path_bindings = _candidate_python_path_bindings(relative_path, tree)
    sequence_bindings = _candidate_python_path_sequence_bindings(
        relative_path,
        tree,
        path_bindings,
    )
    package_parts = _candidate_python_package_parts(relative_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                paths.update(_candidate_module_source_paths(source_root, alias.name.split(".")))
        elif isinstance(node, ast.ImportFrom):
            module_parts = _candidate_import_from_module_parts(package_parts, node)
            if module_parts is None:
                continue
            paths.update(_candidate_module_source_paths(source_root, module_parts))
            for alias in node.names:
                if alias.name == "*":
                    continue
                paths.update(
                    _candidate_module_source_paths(source_root, [*module_parts, alias.name])
                )
        elif isinstance(node, ast.Call):
            paths.update(
                _candidate_python_extension_source_paths(
                    source_root,
                    relative_path,
                    node,
                    path_bindings,
                    sequence_bindings,
                )
            )
    return paths


def _candidate_python_path_bindings(
    relative_path: str,
    tree: ast.AST,
) -> dict[str, PurePosixPath]:
    bindings: dict[str, PurePosixPath] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign):
            resolved = _candidate_python_path_expr(relative_path, node.value, bindings)
            if resolved is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = resolved
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            resolved = _candidate_python_path_expr(relative_path, node.value, bindings)
            if resolved is not None:
                bindings[node.target.id] = resolved
    return bindings


def _candidate_python_path_sequence_bindings(
    relative_path: str,
    tree: ast.AST,
    path_bindings: dict[str, PurePosixPath],
) -> dict[str, tuple[PurePosixPath, ...]]:
    bindings: dict[str, tuple[PurePosixPath, ...]] = {}
    for node in getattr(tree, "body", []):
        value = getattr(node, "value", None)
        if value is None:
            continue
        resolved = _candidate_python_path_sequence_expr(
            relative_path,
            value,
            path_bindings,
            bindings,
        )
        if not resolved:
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = tuple(resolved)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bindings[node.target.id] = tuple(resolved)
    return bindings


def _candidate_python_extension_source_paths(
    source_root: Path,
    relative_path: str,
    node: ast.Call,
    path_bindings: dict[str, PurePosixPath],
    sequence_bindings: dict[str, tuple[PurePosixPath, ...]],
) -> set[str]:
    paths: set[str] = set()
    for keyword in node.keywords:
        if keyword.arg != "sources":
            continue
        for source_path in _candidate_python_path_sequence_expr(
            relative_path,
            keyword.value,
            path_bindings,
            sequence_bindings,
        ):
            normalized = _normalize_candidate_source_path(source_path.as_posix())
            if normalized is not None and _is_snapshot_source_file(source_root, normalized):
                paths.add(normalized)
    return paths


def _candidate_python_path_sequence_expr(
    relative_path: str,
    node: ast.AST,
    path_bindings: dict[str, PurePosixPath],
    sequence_bindings: dict[str, tuple[PurePosixPath, ...]],
) -> tuple[PurePosixPath, ...]:
    if isinstance(node, ast.Name):
        return sequence_bindings.get(node.id, ())
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        paths = []
        for item in node.elts:
            resolved = _candidate_python_path_expr(relative_path, item, path_bindings)
            if resolved is not None:
                paths.append(resolved)
        return tuple(paths)
    resolved = _candidate_python_path_expr(relative_path, node, path_bindings)
    return (resolved,) if resolved is not None else ()


def _candidate_python_path_expr(
    relative_path: str,
    node: ast.AST | None,
    bindings: dict[str, PurePosixPath],
) -> PurePosixPath | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        path = PurePosixPath(node.value)
        if path.is_absolute() or str(path).startswith("candidates/"):
            return path
        return PurePosixPath(relative_path).parent / path
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return PurePosixPath(relative_path)
        return bindings.get(node.id)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {"Path", "PurePath"} and node.args:
            return _candidate_python_path_expr(relative_path, node.args[0], bindings)
        if isinstance(node.func, ast.Name) and node.func.id == "str" and node.args:
            return _candidate_python_path_expr(relative_path, node.args[0], bindings)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "resolve":
            return _candidate_python_path_expr(relative_path, node.func.value, bindings)
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        resolved = _candidate_python_path_expr(relative_path, node.value, bindings)
        return resolved.parent if resolved is not None else None
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and (left := _candidate_python_path_expr(relative_path, node.left, bindings)) is not None
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, str)
    ):
        return left / node.right.value
    return None


def _candidate_python_package_parts(relative_path: str) -> list[str]:
    path = PurePosixPath(relative_path)
    if path.name == "__init__.py":
        return list(path.with_suffix("").parts[:-1])
    return list(path.parent.parts)


def _candidate_import_from_module_parts(
    package_parts: list[str],
    node: ast.ImportFrom,
) -> list[str] | None:
    if node.level:
        if node.level > len(package_parts):
            return None
        parts = package_parts[: len(package_parts) - node.level + 1]
    else:
        parts = []
    if node.module:
        parts.extend(part for part in node.module.split(".") if part)
    return parts


def _candidate_module_source_paths(source_root: Path, module_parts: list[str]) -> set[str]:
    if not module_parts or module_parts[0] != "candidates":
        return set()
    module_path = PurePosixPath(*module_parts)
    paths: set[str] = set()
    file_path = f"{module_path.as_posix()}.py"
    if _is_snapshot_source_file(source_root, file_path):
        paths.add(file_path)
    package_init = f"{module_path.as_posix()}/__init__.py"
    if _is_snapshot_source_file(source_root, package_init):
        paths.add(package_init)
        paths.update(_candidate_source_paths_under(source_root, module_path))
    return paths


def _candidate_path_from_score(source_root: Path, raw_path: object) -> str | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    if "\x00" in raw_path or "\\" in raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        try:
            relative = path.resolve(strict=False).relative_to(source_root.resolve(strict=False))
        except ValueError:
            return None
        raw_path = relative.as_posix()
    return _normalize_candidate_source_path(raw_path)


def _candidate_companion_directories(candidate_path: PurePosixPath) -> list[PurePosixPath]:
    if candidate_path.suffix != ".py":
        return []
    stems = [candidate_path.stem]
    if candidate_path.stem.endswith("_seed"):
        stems.insert(0, candidate_path.stem.removesuffix("_seed"))
    return [candidate_path.parent / stem for stem in dict.fromkeys(stems)]


def _candidate_source_paths_under(source_root: Path, directory: PurePosixPath) -> set[str]:
    normalized = _normalize_candidate_source_path(directory.as_posix())
    if normalized is None:
        return set()
    root = source_root / normalized
    if not root.is_dir() or _has_symlink_component(source_root, root):
        return set()

    paths = set()
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.suffix not in CANDIDATE_SOURCE_SUFFIXES:
            continue
        relative = file_path.relative_to(source_root).as_posix()
        normalized_file = _normalize_candidate_source_path(relative)
        if normalized_file is not None and _is_snapshot_source_file(source_root, normalized_file):
            paths.add(normalized_file)
    return paths


def _normalize_candidate_source_path(path: str) -> str | None:
    posix_path = PurePosixPath(path)
    if posix_path.is_absolute():
        return None
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        return None
    if any(part in SKIPPED_SOURCE_PARTS for part in posix_path.parts):
        return None
    if ".git" in posix_path.parts:
        return None
    if not posix_path.as_posix().startswith("candidates/"):
        return None
    return posix_path.as_posix()


def _is_snapshot_source_file(source_root: Path, relative_path: str) -> bool:
    normalized = _normalize_candidate_source_path(relative_path)
    if normalized is None:
        return False
    path = source_root / normalized
    if path.suffix not in CANDIDATE_SOURCE_SUFFIXES:
        return False
    if not path.is_file() or _has_symlink_component(source_root, path):
        return False
    try:
        path.resolve(strict=True).relative_to(source_root.resolve(strict=True))
    except (FileNotFoundError, ValueError):
        return False
    return True


def _has_symlink_component(source_root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        return True
    current = source_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _patch_failure_summary(result: PatchResult) -> str:
    if result.rejected_reason:
        return f"candidate patch rejected: {result.rejected_reason}"
    if result.stderr_tail:
        return f"candidate patch failed: {result.stderr_tail}"
    return "candidate patch failed"


def _verify_rejected_patch_cleanup(cwd: Path, result: PatchResult) -> PatchResult:
    failure = _patch_cleanup_failure_reason(cwd, result.patch_paths)
    if failure is None:
        return result
    return PatchResult(
        ok=False,
        patch_paths=result.patch_paths,
        returncode=result.returncode,
        stdout_tail=result.stdout_tail,
        stderr_tail=_tail("\n".join(part for part in (result.stderr_tail, failure) if part)),
        rejected_reason=failure,
    )


def _patch_cleanup_failure_reason(cwd: Path, patch_paths: list[str]) -> str | None:
    if not patch_paths or not _is_inside_git_worktree(cwd):
        return None
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *patch_paths],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        detail = _tail(status.stderr) or f"git status returned {status.returncode}"
        return f"candidate patch cleanup status check failed: {detail}"
    dirty = "\n".join(line.strip() for line in status.stdout.splitlines() if line.strip())
    if dirty:
        return f"candidate patch cleanup left paths dirty: {_tail(dirty)}"
    return None


def _is_inside_git_worktree(cwd: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


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


def _summarize_step_payload(
    name: str,
    payload: dict[str, Any],
    *,
    current_best_geomean: float | None = None,
) -> str:
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
    repair_attempts = payload.get("repair_attempts")
    repair_count = len(repair_attempts) if isinstance(repair_attempts, list) else 0

    status = _command_status(command_result)
    patch = _patch_status(patch_result)
    cleanup = _patch_cleanup_status(patch_cleanup_result)
    gate = _gate_status(gate_decision)
    score = _score_status(score_payload)
    stale_accepted = _is_stale_accepted_payload(
        payload,
        current_best_geomean=current_best_geomean,
    )
    failure_class = "stale_accepted" if stale_accepted else _step_failure_class(payload)
    transform_family = _step_transform_family(payload)
    repair_details = _repair_attempts_status(repair_attempts)
    planning = _planning_failure_status(payload)
    command = _shorten(str(decision.get("next_command") or "<missing command>"), 180)
    hypothesis = _shorten(str(decision.get("hypothesis") or "<missing hypothesis>"), 180)
    lineage_note = _stale_accepted_status(
        payload,
        current_best_geomean=current_best_geomean,
    )
    return (
        f"{name}: class={failure_class}; family={transform_family}; {status}; {patch}; {cleanup}; "
        f"repairs={repair_count}{repair_details}; {gate}{lineage_note}; {score}; "
        f"command={command}; hypothesis={hypothesis}{planning}"
    )


def _is_stale_accepted_payload(
    payload: dict[str, Any],
    *,
    current_best_geomean: float | None,
) -> bool:
    if current_best_geomean is None or current_best_geomean <= 0.0:
        return False
    if not _step_payload_accepted(payload):
        return False
    score_payload = _step_score_payload(payload)
    if not isinstance(score_payload, dict):
        return False
    try:
        candidate_geomean = float(score_payload.get("geomean_tflops") or 0.0)
    except (TypeError, ValueError):
        return False
    return candidate_geomean > current_best_geomean + 1e-9


def _stale_accepted_status(
    payload: dict[str, Any],
    *,
    current_best_geomean: float | None,
) -> str:
    if not _is_stale_accepted_payload(
        payload,
        current_best_geomean=current_best_geomean,
    ):
        return ""
    return (
        f"; lineage status=stale accepted above current best {current_best_geomean:.12g}; "
        "treat as reverted/noisy, not current best"
    )


def _repair_attempts_status(repair_attempts: Any) -> str:
    if not isinstance(repair_attempts, list) or not repair_attempts:
        return ""
    details = []
    for repair_attempt in repair_attempts[-2:]:
        if not isinstance(repair_attempt, dict):
            continue
        repair_payload = {"attempt": repair_attempt}
        decision = (
            repair_attempt.get("decision")
            if isinstance(repair_attempt.get("decision"), dict)
            else {}
        )
        command_result = (
            repair_attempt.get("command_result")
            if isinstance(repair_attempt.get("command_result"), dict)
            else {}
        )
        patch_result = (
            repair_attempt.get("patch_result")
            if isinstance(repair_attempt.get("patch_result"), dict)
            else None
        )
        score_payload = repair_attempt.get("score_payload")
        details.append(
            "repair("
            f"class={_step_failure_class(repair_payload)}, "
            f"family={_step_transform_family(repair_payload)}, "
            f"{_command_status(command_result)}, "
            f"{_patch_status(patch_result)}, "
            f"{_score_status(score_payload)}, "
            f"command={_shorten(str(decision.get('next_command') or '<missing command>'), 100)}"
            ")"
        )
    if not details:
        return ""
    return "; repair_details=" + " | ".join(details)


def _planning_failure_status(payload: dict[str, Any]) -> str:
    detail = _planning_failure_detail(payload)
    if not detail:
        return ""
    return f"; planning_feedback={_shorten(detail, 260)}"


def _step_failure_class(payload: dict[str, Any]) -> str:
    if _step_payload_accepted(payload):
        return "accepted"
    attempt = payload.get("attempt") if isinstance(payload.get("attempt"), dict) else {}
    decision = attempt.get("decision") if isinstance(attempt.get("decision"), dict) else {}
    command_result = (
        attempt.get("command_result") if isinstance(attempt.get("command_result"), dict) else {}
    )
    patch_result = (
        attempt.get("patch_result") if isinstance(attempt.get("patch_result"), dict) else None
    )
    gate_decision = payload.get("gate_decision")
    score_payload = attempt.get("score_payload")
    if isinstance(patch_result, dict) and patch_result.get("ok") is False:
        return _classify_patch_failure(patch_result)
    planning_detail = _result_detail(command_result).lower()
    if "agent planning failed validation" in planning_detail:
        return _classify_planning_failure(planning_detail)
    if command_result.get("timed_out"):
        return "timeout"
    returncode = command_result.get("returncode")
    next_command = str(decision.get("next_command") or "")
    command_text = " ".join(str(item) for item in command_result.get("command") or [])
    detail = _result_detail(command_result).lower()
    if next_command.startswith("avo profile "):
        if "profiler_unsupported_runtime" in detail:
            return "profiler_unsupported_runtime"
        if "no_kernels_profiled" in detail or "no kernels were profiled" in detail:
            return "profiler_no_kernels"
        if '"error": "timeout"' in detail:
            return "profiler_timeout"
        if '"error": "profiler_permission"' in detail or "err_nvgpuctrperm" in detail:
            return "profiler_permission"
    if returncode not in (0, None):
        if _looks_like_worker_crash(returncode, detail):
            return "worker_crash"
        if _command_or_detail_looks_like_compile_failure(next_command, command_text, detail):
            return _classify_compile_failure(detail)
        if "correctness" in detail or "max_abs_error" in detail:
            return "correctness_failed"
        return "command_failed"
    if isinstance(score_payload, dict) and score_payload.get("all_correct") is False:
        return _classify_score_failure(score_payload)
    if isinstance(gate_decision, dict) and gate_decision.get("accepted") is False:
        reason = str(gate_decision.get("reason") or "").lower()
        if "correct" in reason or "error" in reason:
            return "correctness_failed"
        return "throughput_regression"
    if returncode == 0 and "avo compile" in next_command and not isinstance(score_payload, dict):
        return "compile_only_diagnostic"
    if returncode == 0:
        return "unaccepted_success"
    return "unknown"


def _classify_planning_failure(detail: str) -> str:
    if _is_planner_provider_error(detail):
        return "planner_provider_error"
    if "candidate_patch and candidate_transform are mutually exclusive" in detail:
        return "planning_edit_channel"
    if "pending compile-only candidate_transform" in detail:
        return "planning_missing_pending_transform"
    if "candidate_transform semantic mismatch" in detail:
        return "planning_transform_semantic_mismatch"
    if "predicted_correctness_failure" in detail:
        return "planning_predicted_correctness_failure"
    if "support-only" in detail:
        return "planning_support_only_transform"
    if "candidate_transform or candidate_patch" in detail:
        return "planning_missing_edit_payload"
    if "recorded no-patch compile diagnostic" in detail:
        return "planning_no_patch_compile"
    if "candidate_transform" in detail:
        return "planning_transform_preflight"
    return "planning_validation"


def _looks_like_worker_crash(returncode: object, detail: str) -> bool:
    if isinstance(returncode, int):
        if returncode < 0:
            return True
        if returncode in {128 + 6, 128 + 9, 128 + 11}:
            return True
    return any(
        marker in detail
        for marker in (
            "segmentation fault",
            "segfault",
            "sigabrt",
            "sigkill",
            "sigsegv",
            "core dumped",
        )
    )


def _is_planner_provider_error(detail: str) -> bool:
    provider_markers = (
        "badrequesterror",
        "apierror",
        "apiconnectionerror",
        "apistatuserror",
        "apitimeouterror",
        "ratelimiterror",
        "overloadederror",
        "provider unavailable",
        "credit balance",
        "plans & billing",
        "anthropic",
    )
    return any(marker in detail for marker in provider_markers)


def _classify_patch_failure(patch_result: dict[str, Any]) -> str:
    detail = _result_detail(patch_result).lower()
    reason = str(patch_result.get("rejected_reason") or "").lower()
    combined = f"{reason} {detail}"
    structural_class = _classified_structural_preflight_failure(combined)
    if structural_class:
        return structural_class
    if "candidate transform rejected" in combined:
        return "structured_transform_preflight"
    if "git apply" in combined or "corrupt patch" in combined or "hunk" in combined:
        return "raw_diff_preflight"
    if "path" in combined or "symlink" in combined or "binary" in combined:
        return "patch_safety_preflight"
    return "patch_preflight"


def _classified_structural_preflight_failure(text: str) -> str | None:
    if "structural preflight track" not in text:
        return None
    match = re.search(r"classified as ([a-z_]+)", text)
    return match.group(1) if match else "patch_preflight"


def _classify_compile_failure(detail: str) -> str:
    if "wmma::fragment" in detail or "incomplete type" in detail or "fill_fragment" in detail:
        return "unsupported_wmma_shape"
    if any(marker in detail for marker in ASYNC_COPY_COMPILE_MARKERS):
        return "async_copy_compile_error"
    if "invalid specifier" in detail or "expected a" in detail or "syntax" in detail:
        return "cuda_syntax_error"
    if "identifier" in detail and "undefined" in detail:
        return "stale_or_undefined_symbol"
    return "compile_failed"


def _command_or_detail_looks_like_compile_failure(
    next_command: str,
    command_text: str,
    detail: str,
) -> bool:
    text = f"{next_command} {command_text}".lower()
    if "avo compile" in text or " compile" in text:
        return True
    lowered_detail = detail.lower()
    return any(marker in lowered_detail for marker in COMPILE_FAILURE_DETAIL_MARKERS)


def _classify_score_failure(score_payload: dict[str, Any]) -> str:
    error_text = _score_payload_error_text(score_payload).lower()
    if "ninja is required" in error_text or "cuda is not available" in error_text:
        return "score_environment_error"
    if _command_or_detail_looks_like_compile_failure("", "", error_text):
        return "score_time_compile_failure"
    if "non-finite" in error_text:
        return "correctness_nonfinite_output"
    return "correctness_failed"


def _score_payload_error_text(score_payload: dict[str, Any]) -> str:
    cases = score_payload.get("cases")
    if not isinstance(cases, list):
        return ""
    errors = [
        str(case.get("error") or "")
        for case in cases
        if isinstance(case, dict) and case.get("error")
    ]
    return " ".join(errors)


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
    advisories = patch_result.get("advisories")
    advisory = ""
    if isinstance(advisories, list) and advisories:
        advisory = f" advisories={_shorten('; '.join(str(item) for item in advisories), 180)}"
    if ok:
        return f"patch ok paths={paths}{advisory}"
    detail = _result_detail(patch_result)
    if detail:
        return f"patch rejected reason={reason} detail={detail}"
    return f"patch rejected reason={reason}"


def _patch_cleanup_status(cleanup_result: Any) -> str:
    if not isinstance(cleanup_result, dict):
        return "no patch cleanup"
    ok = cleanup_result.get("ok")
    reason = _shorten(str(cleanup_result.get("rejected_reason") or ""), 120)
    if ok:
        return "patch cleanup ok"
    detail = _result_detail(cleanup_result)
    if detail:
        return f"patch cleanup failed reason={reason} detail={detail}"
    return f"patch cleanup failed reason={reason}"


def _result_detail(result: dict[str, Any]) -> str:
    detail = str(result.get("stderr_tail") or result.get("stdout_tail") or "").strip()
    if not detail:
        return ""
    return _shorten(" ".join(detail.split()), 180)


def _command_result_text(result: CommandResult) -> str:
    return "\n".join(part for part in (result.stderr_tail, result.stdout_tail) if part)


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
    error_text = _score_payload_error_text(score_payload)
    if correct is False and error_text:
        return (
            f"score all_correct={correct} geomean_tflops={geomean} "
            f"first_error={_shorten(error_text, 100)}"
        )
    return f"score all_correct={correct} geomean_tflops={geomean}"


def _shorten(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:]
