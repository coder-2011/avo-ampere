from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agent import VariationDecision
from .isolation import RESULT_PREFIX
from .lineage import GateDecision, commit_score

DEFAULT_ALLOWED_SUBCOMMANDS = frozenset({"env", "compile", "score"})
SHELL_TOKENS = frozenset({"&&", "||", ";", "|", ">", ">>", "<", "`"})


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
class VariationAttempt:
    decision: VariationDecision
    command_result: CommandResult
    started_at: str
    completed_at: str
    score_payload: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.as_dict(),
            "command_result": self.command_result.as_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "score_payload": self.score_payload,
        }


@dataclass(frozen=True)
class EvolutionStep:
    attempt: VariationAttempt
    gate_decision: GateDecision | None

    @property
    def accepted(self) -> bool:
        return self.gate_decision is not None and self.gate_decision.accepted

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt.as_dict(),
            "gate_decision": (
                self.gate_decision.as_dict() if self.gate_decision is not None else None
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
    )


def finalize_attempt(
    lineage: Path,
    attempt: VariationAttempt,
    *,
    message: str = "evolve: accept candidate",
) -> EvolutionStep:
    gate_decision = None
    if attempt.score_payload is not None:
        gate_decision = commit_score(lineage, attempt.score_payload, message=message)
    return EvolutionStep(attempt=attempt, gate_decision=gate_decision)


def write_attempt(path: Path, attempt: VariationAttempt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(attempt.as_dict(), indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def write_step(path: Path, step: EvolutionStep) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(step.as_dict(), indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:]
