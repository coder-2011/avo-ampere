import sys
from pathlib import Path

from avo.isolation import module_worker_args, run_json_worker


def test_json_worker_success() -> None:
    result = run_json_worker(
        module_worker_args("worker-sleep", "--seconds", "0"),
        timeout_s=5,
        cwd=Path.cwd(),
    )
    assert result.ok
    assert result.payload == {"slept": 0.0}


def test_json_worker_timeout() -> None:
    result = run_json_worker(
        module_worker_args("worker-sleep", "--seconds", "2"),
        timeout_s=1,
        cwd=Path.cwd(),
    )
    assert not result.ok
    assert result.timed_out
    assert result.returncode is None


def test_json_worker_contains_child_crash() -> None:
    result = run_json_worker(
        [sys.executable, "-c", "import os; os._exit(139)"],
        timeout_s=5,
        cwd=Path.cwd(),
    )
    assert not result.ok
    assert result.returncode == 139
