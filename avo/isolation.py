from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RESULT_PREFIX = "AVO_RESULT_JSON="


@dataclass(frozen=True)
class IsolatedResult:
    returncode: int | None
    timed_out: bool
    payload: dict[str, Any] | None
    stdout_tail: str
    stderr_tail: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.payload is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "payload": self.payload,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


def run_json_worker(
    args: list[str],
    *,
    timeout_s: int,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> IsolatedResult:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return IsolatedResult(
            returncode=None,
            timed_out=True,
            payload=None,
            stdout_tail=_tail(exc.stdout or ""),
            stderr_tail=_tail(exc.stderr or ""),
        )

    return IsolatedResult(
        returncode=completed.returncode,
        timed_out=False,
        payload=_extract_payload(completed.stdout),
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
    )


def module_worker_args(*args: str) -> list[str]:
    return [sys.executable, "-m", "avo", *args]


def print_result(payload: dict[str, Any]) -> None:
    print(f"{RESULT_PREFIX}{json.dumps(payload, sort_keys=True)}", flush=True)


def _extract_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if not line.startswith(RESULT_PREFIX):
            continue
        try:
            parsed = json.loads(line.removeprefix(RESULT_PREFIX))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:]
