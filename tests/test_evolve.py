import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from avo.agent import VariationDecision
from avo.evolve import (
    CommandResult,
    EvolutionStep,
    PatchResult,
    VariationAttempt,
    _extract_score_payload,
    apply_candidate_patch,
    cleanup_rejected_candidate_patch,
    command_from_decision,
    finalize_attempt,
    paths_from_unified_diff,
    revert_candidate_patch,
    run_decision_command,
    summarize_attempt_history,
    write_attempt,
    write_step,
    write_step_record,
)
from avo.lineage import best_geomean


def decision(next_command: str, *, candidate_patch: str = "") -> VariationDecision:
    return VariationDecision(
        hypothesis="validate the execution substrate",
        files_to_inspect=["avo/evolve.py"],
        candidate_edit="run a bounded command",
        expected_effect="records an attempt without shell execution",
        risk="command may fail",
        next_command=next_command,
        candidate_patch=candidate_patch,
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


def test_paths_from_unified_diff_extracts_candidate_paths() -> None:
    patch = candidate_value_patch()

    assert paths_from_unified_diff(patch) == ["candidates/seed.py"]


def test_apply_candidate_patch_dry_run_does_not_modify_file(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    patch = candidate_value_patch()

    result = apply_candidate_patch(patch, cwd=tmp_path, dry_run=True)

    assert result.ok
    assert result.patch_paths == ["candidates/seed.py"]
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_apply_candidate_patch_updates_candidate_file(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)

    result = apply_candidate_patch(candidate_value_patch(), cwd=tmp_path)

    assert result.ok
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_apply_candidate_patch_recounts_llm_hunk_lengths(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    patch = dedent(
        """\
        diff --git a/candidates/seed.py b/candidates/seed.py
        --- a/candidates/seed.py
        +++ b/candidates/seed.py
        @@ -1,3 +1,3 @@
        -VALUE = 1
        +VALUE = 2
        """
    )

    result = apply_candidate_patch(patch, cwd=tmp_path)

    assert result.ok
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_revert_candidate_patch_restores_candidate_file(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    apply_result = apply_candidate_patch(candidate_value_patch(), cwd=tmp_path)

    revert_result = revert_candidate_patch(candidate_value_patch(), cwd=tmp_path)

    assert apply_result.ok
    assert revert_result.ok
    assert revert_result.patch_paths == ["candidates/seed.py"]
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_apply_candidate_patch_rejects_non_candidate_path() -> None:
    result = apply_candidate_patch(
        dedent(
            """\
            diff --git a/README.md b/README.md
            --- a/README.md
            +++ b/README.md
            @@ -1 +1 @@
            -old
            +new
            """
        ),
        cwd=Path.cwd(),
    )

    assert_patch_rejected(result, "under: candidates")


def test_apply_candidate_patch_rejects_path_traversal() -> None:
    result = apply_candidate_patch(
        dedent(
            """\
            diff --git a/candidates/../evil.py b/candidates/../evil.py
            --- a/candidates/../evil.py
            +++ b/candidates/../evil.py
            @@ -1 +1 @@
            -old
            +new
            """
        ),
        cwd=Path.cwd(),
    )

    assert_patch_rejected(result, "path traversal")


def test_apply_candidate_patch_rejects_symlink_patch() -> None:
    result = apply_candidate_patch(
        dedent(
            """\
            diff --git a/candidates/link b/candidates/link
            new file mode 120000
            index 0000000..e69de29
            --- /dev/null
            +++ b/candidates/link
            @@ -0,0 +1 @@
            +../outside
            """
        ),
        cwd=Path.cwd(),
    )

    assert_patch_rejected(result, "unsupported patch marker")


def test_apply_candidate_patch_rejects_existing_symlink_path(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (candidate_dir / "link").symlink_to(outside, target_is_directory=True)

    result = apply_candidate_patch(
        dedent(
            """\
            diff --git a/candidates/link/seed.py b/candidates/link/seed.py
            --- a/candidates/link/seed.py
            +++ b/candidates/link/seed.py
            @@ -1 +1 @@
            -old
            +new
            """
        ),
        cwd=tmp_path,
    )

    assert_patch_rejected(result, "existing symlink")


def test_apply_candidate_patch_rejects_binary_patch() -> None:
    result = apply_candidate_patch(
        dedent(
            """\
            diff --git a/candidates/blob.bin b/candidates/blob.bin
            new file mode 100644
            index 0000000..1234567
            GIT binary patch
            literal 0
            HcmV?d00001
            """
        ),
        cwd=Path.cwd(),
    )

    assert_patch_rejected(result, "unsupported patch marker")


def test_apply_candidate_patch_rejects_delete_patch() -> None:
    result = apply_candidate_patch(
        dedent(
            """\
            diff --git a/candidates/seed.py b/candidates/seed.py
            deleted file mode 100644
            index 1234567..0000000
            --- a/candidates/seed.py
            +++ /dev/null
            @@ -1 +0,0 @@
            -VALUE = 1
            """
        ),
        cwd=Path.cwd(),
    )

    assert_patch_rejected(result, "unsupported patch marker")


def test_apply_candidate_patch_rejects_empty_patch() -> None:
    result = apply_candidate_patch("", cwd=Path.cwd())

    assert_patch_rejected(result, "at least one diff")


def test_apply_candidate_patch_rejects_non_git_unified_diff() -> None:
    result = apply_candidate_patch(
        dedent(
            """\
            --- a/candidates/seed.py
            +++ b/candidates/seed.py
            @@ -1 +1 @@
            -VALUE = 1
            +VALUE = 2
            """
        ),
        cwd=Path.cwd(),
    )

    assert_patch_rejected(result, "diff --git")


def test_run_decision_command_executes_allowed_command() -> None:
    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0"),
        cwd=Path.cwd(),
        timeout_s=10,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )

    assert attempt.command_result.ok
    assert "AVO_RESULT_JSON" in attempt.command_result.stdout_tail
    assert attempt.patch_result is None


def test_run_decision_command_applies_candidate_patch_before_command(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0", candidate_patch=candidate_value_patch()),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )

    assert attempt.patch_result is not None
    assert attempt.patch_result.ok
    assert attempt.command_result.ok
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_run_decision_command_stops_when_candidate_patch_is_rejected(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    attempt = run_decision_command(
        decision(
            "avo worker-sleep --seconds 0",
            candidate_patch=dedent(
                """\
                diff --git a/README.md b/README.md
                --- a/README.md
                +++ b/README.md
                @@ -1 +1 @@
                -old
                +new
                """
            ),
        ),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )

    assert attempt.patch_result is not None
    assert not attempt.patch_result.ok
    assert not attempt.command_result.ok
    assert attempt.command_result.returncode is None
    assert "candidate patch rejected" in attempt.command_result.stderr_tail


def test_cleanup_rejected_candidate_patch_reverts_nonaccepted_patch(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0", candidate_patch=candidate_value_patch()),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )
    step = EvolutionStep(attempt=attempt, gate_decision=None)

    cleaned = cleanup_rejected_candidate_patch(step, cwd=tmp_path)

    assert cleaned.patch_cleanup_result is not None
    assert cleaned.patch_cleanup_result.ok
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_cleanup_rejected_candidate_patch_rejects_dirty_patch_path(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    seed.write_text("VALUE = 1\nOTHER = 1\n", encoding="utf-8")
    init_git_repo(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    patch = candidate_value_patch_with_context()
    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0", candidate_patch=patch),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )
    seed.write_text("VALUE = 2\nOTHER = 1\nEXTRA = 3\n", encoding="utf-8")
    step = EvolutionStep(attempt=attempt, gate_decision=None)

    cleaned = cleanup_rejected_candidate_patch(step, cwd=tmp_path)

    assert cleaned.patch_cleanup_result is not None
    assert not cleaned.patch_cleanup_result.ok
    assert cleaned.patch_cleanup_result.rejected_reason is not None
    assert "left paths dirty" in cleaned.patch_cleanup_result.rejected_reason
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\nOTHER = 1\nEXTRA = 3\n"


def test_cleanup_rejected_candidate_patch_keeps_accepted_patch(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    score = {"backend": "mock", "all_correct": True, "geomean_tflops": 3.0, "cases": [{}]}
    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0", candidate_patch=candidate_value_patch()),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )
    accepted_attempt = VariationAttempt(
        decision=attempt.decision,
        command_result=attempt.command_result,
        started_at=attempt.started_at,
        completed_at=attempt.completed_at,
        score_payload=score,
        patch_result=attempt.patch_result,
    )
    step = finalize_attempt(tmp_path / "lineage", accepted_attempt)

    cleaned = cleanup_rejected_candidate_patch(step, cwd=tmp_path)

    assert cleaned.accepted
    assert cleaned.patch_cleanup_result is None
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_finalize_attempt_snapshots_accepted_patch_sources(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0", candidate_patch=candidate_value_patch()),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )
    accepted_attempt = VariationAttempt(
        decision=attempt.decision,
        command_result=attempt.command_result,
        started_at=attempt.started_at,
        completed_at=attempt.completed_at,
        score_payload={
            "backend": "mock",
            "all_correct": True,
            "geomean_tflops": 3.0,
            "cases": [{}],
        },
        patch_result=attempt.patch_result,
    )

    step = finalize_attempt(tmp_path / "lineage", accepted_attempt, source_root=tmp_path)

    assert step.accepted
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (tmp_path / "lineage" / "sources" / "latest" / "candidates" / "seed.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 2\n"
    assert (tmp_path / "lineage" / "patches" / "latest.patch").read_text(
        encoding="utf-8"
    ) == candidate_value_patch()


def test_finalize_attempt_snapshots_scored_candidate_sources_without_patch(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates"
    companion_dir = candidate_dir / "cuda_demo"
    companion_dir.mkdir(parents=True)
    seed = candidate_dir / "cuda_demo_seed.py"
    seed.write_text("from candidates.cuda_demo import attention\n", encoding="utf-8")
    (companion_dir / "attention.cpp").write_text("// cpp binding\n", encoding="utf-8")
    (companion_dir / "attention_kernel.cu").write_text("// cuda kernel\n", encoding="utf-8")
    (companion_dir / "compiled.so").write_bytes(b"not source")
    (candidate_dir / "__pycache__").mkdir()
    (candidate_dir / "__pycache__" / "cuda_demo_seed.pyc").write_bytes(b"cache")
    attempt = VariationAttempt(
        decision=decision("avo score --backend candidate --candidate candidates/cuda_demo_seed.py"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        score_payload={
            "backend": "candidate",
            "candidate_path": "candidates/cuda_demo_seed.py",
            "all_correct": True,
            "geomean_tflops": 3.0,
            "cases": [{}],
        },
    )

    step = finalize_attempt(tmp_path / "lineage", attempt, source_root=tmp_path)

    assert step.accepted
    source_root = tmp_path / "lineage" / "sources" / "latest"
    assert (source_root / "candidates" / "cuda_demo_seed.py").read_text(
        encoding="utf-8"
    ) == seed.read_text(encoding="utf-8")
    assert (source_root / "candidates" / "cuda_demo" / "attention.cpp").read_text(
        encoding="utf-8"
    ) == "// cpp binding\n"
    assert (source_root / "candidates" / "cuda_demo" / "attention_kernel.cu").read_text(
        encoding="utf-8"
    ) == "// cuda kernel\n"
    assert not (source_root / "candidates" / "cuda_demo" / "compiled.so").exists()
    assert not (source_root / "candidates" / "__pycache__").exists()
    assert not (tmp_path / "lineage" / "patches" / "latest.patch").exists()


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
    score = {"backend": "mock", "all_correct": True, "geomean_tflops": 3.0, "cases": [{}]}
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
    score = {"backend": "mock", "all_correct": True, "geomean_tflops": 3.0, "cases": [{}]}
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
    assert payload["patch_cleanup_result"] is None


def test_write_step_record_uses_timestamped_file(tmp_path: Path) -> None:
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
    step = EvolutionStep(attempt=attempt, gate_decision=None)

    first = write_step_record(tmp_path / "attempts", step)
    second = write_step_record(tmp_path / "attempts", step)

    assert first.exists()
    assert second.exists()
    assert first != second
    assert first.name.startswith("2026-05-08T00-00-01-00-00")


def test_summarize_attempt_history_reports_recent_steps(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    accepted_score = {"backend": "mock", "all_correct": True, "geomean_tflops": 3.0, "cases": [{}]}
    accepted_attempt = VariationAttempt(
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
        score_payload=accepted_score,
    )
    rejected_attempt = VariationAttempt(
        decision=decision("avo score --backend candidate"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=2,
            timed_out=False,
            stdout_tail="",
            stderr_tail="failed",
        ),
        started_at="2026-05-08T00:00:02+00:00",
        completed_at="2026-05-08T00:00:03+00:00",
    )
    patch_cleanup = PatchResult(
        ok=True,
        patch_paths=["candidates/seed.py"],
        returncode=0,
        stdout_tail="",
        stderr_tail="",
    )

    write_step_record(attempts, finalize_attempt(tmp_path / "lineage", accepted_attempt))
    write_step_record(
        attempts,
        EvolutionStep(
            attempt=rejected_attempt,
            gate_decision=None,
            patch_cleanup_result=patch_cleanup,
        ),
    )
    (attempts / "bad.json").write_text("{not-json", encoding="utf-8")

    summary = summarize_attempt_history(attempts, limit=5)

    assert "Recent attempts" in summary
    assert "avo score --backend torch-sdpa" in summary
    assert "gate accepted=True" in summary
    assert "geomean_tflops=3.0" in summary
    assert "command returncode=2" in summary
    assert "patch cleanup ok" in summary
    assert "bad.json" not in summary


def test_summarize_attempt_history_flags_repeated_unaccepted_attempts(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    for index in range(3):
        attempt = VariationAttempt(
            decision=decision("avo score --backend candidate"),
            command_result=CommandResult(
                command=[sys.executable, "-m", "avo", "score"],
                returncode=2,
                timed_out=False,
                stdout_tail="",
                stderr_tail="failed",
            ),
            started_at=f"2026-05-08T00:00:0{index}+00:00",
            completed_at=f"2026-05-08T00:00:0{index + 1}+00:00",
        )
        write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "Supervisor signal" in summary
    assert "share command/edit fingerprint" in summary
    assert "materially different optimization direction" in summary


def test_summarize_attempt_history_normalizes_compile_out_dir(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    source = "candidates/cuda_tiled_attention/attention_kernel.cu"
    for index in range(3):
        attempt = VariationAttempt(
            decision=decision(f"avo compile --source {source} --out-dir build/tiled_{index}"),
            command_result=CommandResult(
                command=[sys.executable, "-m", "avo", "compile"],
                returncode=0,
                timed_out=False,
                stdout_tail="",
                stderr_tail="",
            ),
            started_at=f"2026-05-08T00:00:0{index}+00:00",
            completed_at=f"2026-05-08T00:00:0{index + 1}+00:00",
        )
        write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "Supervisor signal" in summary
    assert "share command/edit fingerprint" in summary


def test_summarize_attempt_history_flags_unaccepted_exhaustion(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    for index in range(5):
        attempt = VariationAttempt(
            decision=decision(
                f"avo score --backend candidate --candidate candidates/seed_{index}.py"
            ),
            command_result=CommandResult(
                command=[sys.executable, "-m", "avo", "score"],
                returncode=2,
                timed_out=False,
                stdout_tail="",
                stderr_tail="failed",
            ),
            started_at=f"2026-05-08T00:00:0{index}+00:00",
            completed_at=f"2026-05-08T00:00:0{index + 1}+00:00",
        )
        write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "Supervisor signal" in summary
    assert "last 5 attempts produced no accepted candidate" in summary
    assert "reset strategy" in summary


def write_seed_candidate(root: Path) -> Path:
    candidate_dir = root / "candidates"
    candidate_dir.mkdir()
    seed = candidate_dir / "seed.py"
    seed.write_text("VALUE = 1\n", encoding="utf-8")
    return seed


def init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "candidates/seed.py"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AVO Test",
            "-c",
            "user.email=avo-test@example.com",
            "commit",
            "-m",
            "seed candidate",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


def candidate_value_patch() -> str:
    return dedent(
        """\
        diff --git a/candidates/seed.py b/candidates/seed.py
        --- a/candidates/seed.py
        +++ b/candidates/seed.py
        @@ -1 +1 @@
        -VALUE = 1
        +VALUE = 2
        """
    )


def candidate_value_patch_with_context() -> str:
    return dedent(
        """\
        diff --git a/candidates/seed.py b/candidates/seed.py
        --- a/candidates/seed.py
        +++ b/candidates/seed.py
        @@ -1,2 +1,2 @@
        -VALUE = 1
        +VALUE = 2
         OTHER = 1
        """
    )


def assert_patch_rejected(result: PatchResult, reason: str) -> None:
    assert not result.ok
    assert result.rejected_reason is not None
    assert reason in result.rejected_reason
