import json
import sys
from pathlib import Path

import pytest

from avo.agent import VariationDecision
from avo.evolve import command_from_decision, run_decision_command, write_attempt


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
