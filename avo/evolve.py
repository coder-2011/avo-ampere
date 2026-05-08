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

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.as_dict(),
            "command_result": self.command_result.as_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
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
    except subprocess.TimeoutExpired as exc:
        result = CommandResult(
            command=command,
            returncode=None,
            timed_out=True,
            stdout_tail=_tail(exc.stdout or ""),
            stderr_tail=_tail(exc.stderr or ""),
        )
    return VariationAttempt(
        decision=decision,
        command_result=result,
        started_at=started_at,
        completed_at=_utc_now(),
    )


def write_attempt(path: Path, attempt: VariationAttempt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(attempt.as_dict(), indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:]
