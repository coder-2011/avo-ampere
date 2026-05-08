import json
import sys
from pathlib import Path

import pytest

from avo.agent import VariationDecision
from avo.evolve import (
    CommandResult,
    VariationAttempt,
    _extract_score_payload,
    command_from_decision,
    finalize_attempt,
    run_decision_command,
    write_attempt,
    write_step,
)
from avo.lineage import best_geomean


def decision(next_command: str) -> VariationDecision:
    return VariationDecision(
        hypothesis="validate the execution substrate",
        files_to_inspect=["avo/evolve.py"],
        candidate_edit="run a bounded command",
        expected_effect="records an attempt without shell execution",
        risk="command may fail",
        next_command=next_command,
    )


def test_command_from_decision_rewrites_avo_to_module() -> None:
    command = command_from_decision(decision("avo score --backend torch-sdpa"))

    assert command[:3] == [sys.executable, "-m", "avo"]
    assert command[3:] == ["score", "--backend", "torch-sdpa"]


def test_command_from_decision_rejects_shell() -> None:
    with pytest.raises(ValueError, match="must start with 'avo'"):
        command_from_decision(decision("rm -rf /"))

    with pytest.raises(ValueError, match="shell control"):
        command_from_decision(decision("avo env && rm -rf /"))


def test_command_from_decision_rejects_unsupported_subcommand() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        command_from_decision(decision("avo commit-score lineage score.json"))


def test_run_decision_command_executes_allowed_command() -> None:
    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0"),
        cwd=Path.cwd(),
        timeout_s=10,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )

    assert attempt.command_result.ok
    assert "AVO_RESULT_JSON" in attempt.command_result.stdout_tail


def test_write_attempt_records_json(tmp_path: Path) -> None:
    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0"),
        cwd=Path.cwd(),
        timeout_s=10,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )
    path = tmp_path / "attempt.json"

    write_attempt(path, attempt)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["decision"]["next_command"] == "avo worker-sleep --seconds 0"
    assert payload["command_result"]["ok"] is True


def test_extract_score_payload_from_score_wrapper_json() -> None:
    score = {"backend": "mock", "all_correct": True, "geomean_tflops": 3.0, "cases": []}
    stdout = json.dumps({"ok": True, "payload": score})

    assert _extract_score_payload(stdout) == score


def test_extract_score_payload_from_worker_result_line() -> None:
    score = {"backend": "mock", "all_correct": True, "geomean_tflops": 3.0, "cases": []}
    stdout = f"noise\nAVO_RESULT_JSON={json.dumps(score)}\n"

    assert _extract_score_payload(stdout) == score


def test_finalize_attempt_commits_score_payload(tmp_path: Path) -> None:
    score = {"backend": "mock", "all_correct": True, "geomean_tflops": 3.0, "cases": []}
    attempt = VariationAttempt(
        decision=decision("avo score --backend torch-sdpa"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        score_payload=score,
    )

    step = finalize_attempt(tmp_path / "lineage", attempt)

    assert step.accepted
    assert best_geomean(tmp_path / "lineage") == 3.0


def test_finalize_attempt_without_score_payload_does_not_commit(tmp_path: Path) -> None:
    attempt = VariationAttempt(
        decision=decision("avo env"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "env"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )

    step = finalize_attempt(tmp_path / "lineage", attempt)

    assert step.gate_decision is None
    assert best_geomean(tmp_path / "lineage") == 0.0


def test_write_step_records_gate_decision(tmp_path: Path) -> None:
    score = {"backend": "mock", "all_correct": True, "geomean_tflops": 3.0, "cases": []}
    attempt = VariationAttempt(
        decision=decision("avo score --backend torch-sdpa"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        score_payload=score,
    )
    step = finalize_attempt(tmp_path / "lineage", attempt)
    path = tmp_path / "step.json"

    write_step(path, step)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["gate_decision"]["accepted"] is True
