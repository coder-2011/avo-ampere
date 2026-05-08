import json
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

from avo.agent import VariationDecision
from avo.cli import (
    _agent_status,
    _baseline_build_env,
    _baseline_build_status,
    _cuda_build_compatibility,
    _evolve_loop,
    _evolve_once,
    _nvcc_path_from_env,
    _parse_nvcc_release,
    _score,
    _seed_baseline,
    main,
)
from avo.evolve import CommandResult, VariationAttempt, apply_candidate_patch


def test_agent_status_reports_missing_key_without_secret(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    status = _agent_status(None)

    assert status["anthropic_api_key_present"] is False
    assert status["env_file"] is None
    assert status["env_file_loaded"] is False
    assert "ANTHROPIC_API_KEY" not in repr(status)


def test_agent_status_loads_env_file_without_printing_value(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text("ANTHROPIC_API_KEY=test-secret\n", encoding="utf-8")

    status = _agent_status(env_file)

    assert status["anthropic_api_key_present"] is True
    assert status["env_file"] == str(env_file)
    assert status["env_file_loaded"] is True
    assert "test-secret" not in repr(status)


def test_baseline_build_env_targets_flash_attn_ampere() -> None:
    env = {"FLASH_ATTN_CUDA_ARCHS": "90;100", "OTHER": "keep-me", "PATH": "/bin"}
    updated = _baseline_build_env(env)

    assert updated["FLASH_ATTN_CUDA_ARCHS"] == "80"
    assert updated["MAX_JOBS"] == "1"
    assert updated["NVCC_THREADS"] == "1"
    assert updated["OTHER"] == "keep-me"
    assert updated["PATH"] == "/bin"


def test_baseline_build_env_preserves_explicit_parallelism_limits() -> None:
    env = {"MAX_JOBS": "2", "NVCC_THREADS": "2"}
    updated = _baseline_build_env(env)

    assert updated["FLASH_ATTN_CUDA_ARCHS"] == "80"
    assert updated["MAX_JOBS"] == "2"
    assert updated["NVCC_THREADS"] == "2"


def test_parse_nvcc_release() -> None:
    output = "Cuda compilation tools, release 12.9, V12.9.86"

    assert _parse_nvcc_release(output) == "12.9"
    assert _parse_nvcc_release("not nvcc output") is None


def test_cuda_build_compatibility() -> None:
    assert _cuda_build_compatibility("13.0", "13.0") == ("exact", None)

    minor_status, minor_warning = _cuda_build_compatibility("13.0", "13.1")
    assert minor_status == "minor_mismatch"
    assert "may warn" in minor_warning

    major_status, major_warning = _cuda_build_compatibility("13.0", "12.9")
    assert major_status == "major_mismatch"
    assert "will fail" in major_warning


def test_nvcc_path_prefers_cuda_home() -> None:
    assert _nvcc_path_from_env({"CUDA_HOME": "/opt/cuda", "PATH": "/bin"}) == (
        "/opt/cuda/bin/nvcc"
    )


def test_baseline_build_status_reports_cuda_mismatch(monkeypatch) -> None:
    monkeypatch.setattr("avo.cli._nvcc_path_from_env", lambda env: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr("avo.cli._nvcc_cuda_version", lambda nvcc_path, env: ("12.9", None))
    monkeypatch.setattr("avo.cli.importlib.util.find_spec", lambda name: None)

    status = _baseline_build_status({"PATH": "/usr/bin"}, torch_cuda="13.0")

    assert status["flash_attn_installed"] is False
    assert status["settings"]["FLASH_ATTN_CUDA_ARCHS"] == "80"
    assert status["compatibility"] == "major_mismatch"
    assert status["ok_for_torch_extension_build"] is False
    assert "torch was compiled with CUDA 13.0" in status["warning"]


def test_score_command_forwards_trial_count(monkeypatch) -> None:
    captured = {}

    class FakeResult:
        ok = True

        def as_dict(self):
            return {"ok": True, "payload": {"all_correct": True}}

    def fake_run_json_worker(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeResult()

    monkeypatch.setattr("avo.cli.run_json_worker", fake_run_json_worker)

    exit_code = _score(
        SimpleNamespace(
            backend="candidate",
            candidate=Path("candidates/cuda_warp_rows_attention_seed.py"),
            seq_lens="16",
            causal="both",
            head_dim=16,
            num_heads=1,
            total_tokens=16,
            dtype="bf16",
            warmup=1,
            repeats=2,
            trials=3,
            timeout_s=300,
        )
    )

    assert exit_code == 0
    assert "--trials" in captured["args"]
    assert captured["args"][captured["args"].index("--trials") + 1] == "3"


def test_seed_baseline_rejects_missing_flash_attn_when_cuda_build_blocked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("avo.cli.importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr(
        "avo.cli._baseline_build_status",
        lambda env: {
            "ok_for_torch_extension_build": False,
            "warning": "PyTorch extension builds will fail",
        },
    )

    def fail_run_json_worker(*args, **kwargs):
        raise AssertionError("score worker should not run")

    monkeypatch.setattr("avo.cli.run_json_worker", fail_run_json_worker)

    with pytest.raises(RuntimeError, match="baseline source-build environment is not ready"):
        _seed_baseline(
            SimpleNamespace(
                backend="flash-attn",
                candidate=None,
                seq_lens="16",
                causal="false",
                head_dim=16,
                num_heads=1,
                total_tokens=16,
                dtype="bf16",
                warmup=1,
                repeats=1,
                trials=1,
                timeout_s=10,
                path=tmp_path / "lineage",
                message="seed baseline",
                force=False,
            )
        )


def test_apply_patch_command_reports_dry_run_result(tmp_path: Path, capsys) -> None:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    seed = candidate_dir / "seed.py"
    seed.write_text("VALUE = 1\n", encoding="utf-8")
    patch = tmp_path / "candidate.patch"
    patch.write_text(
        dedent(
            """\
            diff --git a/candidates/seed.py b/candidates/seed.py
            --- a/candidates/seed.py
            +++ b/candidates/seed.py
            @@ -1 +1 @@
            -VALUE = 1
            +VALUE = 2
            """
        ),
        encoding="utf-8",
    )

    exit_code = main(["apply-patch", str(patch), "--cwd", str(tmp_path), "--dry-run"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"ok": true' in output
    assert "candidates/seed.py" in output
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_evolve_once_cleans_up_nonaccepted_candidate_patch(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    seed = candidate_dir / "seed.py"
    seed.write_text("VALUE = 1\n", encoding="utf-8")
    knowledge = tmp_path / "knowledge.md"
    knowledge.write_text("Ampere only.", encoding="utf-8")
    decision = VariationDecision(
        hypothesis="test patch cleanup",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="change value",
        expected_effect="exercise cleanup",
        risk="command has no score payload",
        next_command="avo env",
        candidate_patch=dedent(
            """\
            diff --git a/candidates/seed.py b/candidates/seed.py
            --- a/candidates/seed.py
            +++ b/candidates/seed.py
            @@ -1 +1 @@
            -VALUE = 1
            +VALUE = 2
            """
        ),
    )

    def fake_request_variation_decision(**kwargs):
        return decision

    def fake_run_decision_command(decision, *, cwd, timeout_s, env):
        patch_result = apply_candidate_patch(decision.candidate_patch, cwd=cwd)
        return VariationAttempt(
            decision=decision,
            command_result=CommandResult(
                command=["python", "-m", "avo", "env"],
                returncode=0,
                timed_out=False,
                stdout_tail="{}",
                stderr_tail="",
            ),
            started_at="2026-05-08T00:00:00+00:00",
            completed_at="2026-05-08T00:00:01+00:00",
            patch_result=patch_result,
        )

    monkeypatch.setattr("avo.cli.request_variation_decision", fake_request_variation_decision)
    monkeypatch.setattr("avo.cli.run_decision_command", fake_run_decision_command)

    exit_code = _evolve_once(
        SimpleNamespace(
            lineage=tmp_path / "lineage",
            knowledge=knowledge,
            cwd=tmp_path,
            timeout_s=10,
            step_json=tmp_path / "step.json",
            env_file=None,
            model="claude",
            attempts_dir=None,
            attempt_limit=5,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["patch_cleanup_result"]["ok"] is True
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_evolve_once_snapshots_accepted_candidate_patch(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    seed = candidate_dir / "seed.py"
    seed.write_text("VALUE = 1\n", encoding="utf-8")
    knowledge = tmp_path / "knowledge.md"
    knowledge.write_text("Ampere only.", encoding="utf-8")
    decision = VariationDecision(
        hypothesis="test source snapshot",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="change value",
        expected_effect="exercise snapshot",
        risk="mock score only",
        next_command="avo score --backend candidate",
        candidate_patch=dedent(
            """\
            diff --git a/candidates/seed.py b/candidates/seed.py
            --- a/candidates/seed.py
            +++ b/candidates/seed.py
            @@ -1 +1 @@
            -VALUE = 1
            +VALUE = 2
            """
        ),
    )

    def fake_request_variation_decision(**kwargs):
        return decision

    def fake_run_decision_command(decision, *, cwd, timeout_s, env):
        patch_result = apply_candidate_patch(decision.candidate_patch, cwd=cwd)
        return VariationAttempt(
            decision=decision,
            command_result=CommandResult(
                command=["python", "-m", "avo", "score"],
                returncode=0,
                timed_out=False,
                stdout_tail="{}",
                stderr_tail="",
            ),
            started_at="2026-05-08T00:00:00+00:00",
            completed_at="2026-05-08T00:00:01+00:00",
            score_payload={
                "backend": "mock",
                "all_correct": True,
                "geomean_tflops": 3.0,
                "cases": [{}],
            },
            patch_result=patch_result,
        )

    monkeypatch.setattr("avo.cli.request_variation_decision", fake_request_variation_decision)
    monkeypatch.setattr("avo.cli.run_decision_command", fake_run_decision_command)

    exit_code = _evolve_once(
        SimpleNamespace(
            lineage=tmp_path / "lineage",
            knowledge=knowledge,
            cwd=tmp_path,
            timeout_s=10,
            step_json=None,
            env_file=None,
            model="claude",
            attempts_dir=None,
            attempt_limit=5,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["gate_decision"]["accepted"] is True
    assert payload["patch_cleanup_result"] is None
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (tmp_path / "lineage" / "sources" / "latest" / "candidates" / "seed.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 2\n"


def test_evolve_loop_runs_until_accepted_and_records_attempts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    knowledge = tmp_path / "knowledge.md"
    knowledge.write_text("Ampere only.", encoding="utf-8")
    decisions = [
        loop_decision("first rejected"),
        loop_decision("second accepted"),
    ]
    seen_attempt_histories = []

    def fake_request_variation_decision(**kwargs):
        seen_attempt_histories.append(kwargs["attempt_history"])
        return decisions.pop(0)

    def fake_run_decision_command(decision, *, cwd, timeout_s, env):
        geomean = 0.0 if decision.hypothesis == "first rejected" else 3.0
        return VariationAttempt(
            decision=decision,
            command_result=CommandResult(
                command=["python", "-m", "avo", "score"],
                returncode=0,
                timed_out=False,
                stdout_tail="{}",
                stderr_tail="",
            ),
            started_at="2026-05-08T00:00:00+00:00",
            completed_at="2026-05-08T00:00:01+00:00",
            score_payload={
                "backend": "mock",
                "all_correct": True,
                "geomean_tflops": geomean,
                "cases": [{}],
            },
        )

    monkeypatch.setattr("avo.cli.request_variation_decision", fake_request_variation_decision)
    monkeypatch.setattr("avo.cli.run_decision_command", fake_run_decision_command)

    exit_code = _evolve_loop(
        SimpleNamespace(
            lineage=tmp_path / "lineage",
            knowledge=knowledge,
            cwd=tmp_path,
            timeout_s=10,
            env_file=None,
            model="claude",
            max_steps=3,
            loop_json=tmp_path / "loop.json",
            attempts_dir=tmp_path / "attempts",
            attempt_limit=5,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["accepted"] is True
    assert payload["completed_steps"] == 2
    assert payload["stopped_reason"] == "accepted"
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 2
    assert seen_attempt_histories[0] == ""
    assert "first rejected" in seen_attempt_histories[1]
    assert json.loads((tmp_path / "loop.json").read_text(encoding="utf-8"))["accepted"] is True


def test_evolve_loop_requires_attempts_dir(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge.md"
    knowledge.write_text("Ampere only.", encoding="utf-8")

    with pytest.raises(ValueError, match="attempts-dir"):
        _evolve_loop(
            SimpleNamespace(
                lineage=tmp_path / "lineage",
                knowledge=knowledge,
                cwd=tmp_path,
                timeout_s=10,
                env_file=None,
                model="claude",
                max_steps=1,
                loop_json=None,
                attempts_dir=None,
                attempt_limit=5,
            )
        )


def loop_decision(hypothesis: str) -> VariationDecision:
    return VariationDecision(
        hypothesis=hypothesis,
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="mock loop step",
        expected_effect="exercise loop control",
        risk="mock score only",
        next_command="avo score --backend candidate",
    )
