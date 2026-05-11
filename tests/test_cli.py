import json
import os
import tomllib
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

from avo.agent import VariationDecision
from avo.cli import (
    _agent_status,
    _baseline_build_env,
    _baseline_build_status,
    _evolve_loop,
    _evolve_once,
    _pending_transform_payload_normalizer,
    _planning_context,
    _profile,
    _run_compile_repair_loop,
    _score,
    _seed_baseline,
    _torch_extension_worker_env,
    _with_general_cuda_context,
    main,
)
from avo.cuda_env import (
    cuda_build_compatibility,
    nvcc_path_from_env,
    parse_nvcc_release,
    prepare_torch_extension_env,
    python_cuda_home,
)
from avo.evolve import (
    CommandResult,
    EvolutionStep,
    PatchResult,
    VariationAttempt,
    apply_candidate_patch,
    materialize_candidate_transform,
    write_step_record,
)


def test_agent_status_reports_missing_key_without_secret(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    status = _agent_status(None)

    assert status["anthropic_api_key_present"] is False
    assert status["env_file"] is None
    assert status["env_file_loaded"] is False
    assert "ANTHROPIC_API_KEY" not in repr(status)


def test_baseline_extra_does_not_auto_install_flash_attn() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    baseline_deps = pyproject["project"]["optional-dependencies"]["baseline"]

    assert all(not dependency.startswith("flash-attn") for dependency in baseline_deps)


def test_agent_status_loads_env_file_without_printing_value(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text("ANTHROPIC_API_KEY=test-secret\n", encoding="utf-8")

    status = _agent_status(env_file)

    assert status["anthropic_api_key_present"] is True
    assert status["env_file"] == str(env_file)
    assert status["env_file_loaded"] is True
    assert "test-secret" not in repr(status)


def test_agent_status_command_prints_json_without_secret(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text("ANTHROPIC_API_KEY=test-secret\n", encoding="utf-8")

    assert main(["agent-status", "--env-file", str(env_file)]) == 0

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["anthropic_api_key_present"] is True
    assert payload["env_file"] == str(env_file)
    assert payload["env_file_loaded"] is True
    assert "test-secret" not in output


def test_lineage_summary_command_prints_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "avo.cli._lineage_summary",
        lambda path: json.dumps(
            {
                "latest": {"geomean_tflops": 9.0},
                "baseline_comparisons": [{"candidate_vs_baseline": 0.1}],
            }
        ),
    )

    assert main(["lineage-summary", "lineage"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["latest"]["geomean_tflops"] == 9.0
    assert payload["baseline_comparisons"][0]["candidate_vs_baseline"] == 0.1


def test_baseline_build_env_targets_flash_attn_ampere(monkeypatch) -> None:
    monkeypatch.setattr("avo.cuda_env.compatible_python_cuda_home", lambda env: None)
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


def test_baseline_env_command_prints_shell_exports(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "avo.cli._baseline_build_env",
        lambda env: {
            "FLASH_ATTN_CUDA_ARCHS": "80",
            "MAX_JOBS": "3",
            "NVCC_THREADS": "1",
            "CUDA_HOME": "/venv/nvidia/cu13",
            "LIBRARY_PATH": "/tmp/cuda links/lib:/venv/nvidia/cu13/lib",
        },
    )

    assert main(["baseline-env"]) == 0

    output = capsys.readouterr().out
    assert "export FLASH_ATTN_CUDA_ARCHS=80" in output
    assert "export MAX_JOBS=3" in output
    assert "export CUDA_HOME=/venv/nvidia/cu13" in output
    assert "export LIBRARY_PATH='/tmp/cuda links/lib:/venv/nvidia/cu13/lib'" in output


def test_baseline_env_command_prints_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "avo.cli._baseline_build_env",
        lambda env: {"FLASH_ATTN_CUDA_ARCHS": "80", "MAX_JOBS": "1"},
    )

    assert main(["baseline-env", "--format", "json"]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "FLASH_ATTN_CUDA_ARCHS": "80",
        "MAX_JOBS": "1",
    }


def test_knowledge_search_command_prints_retrieved_context(tmp_path: Path, capsys) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "ampere.md").write_text(
        "Ampere cp.async uses aligned shared-memory staging.\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "knowledge-search",
            str(knowledge),
            "--query",
            "cp.async shared staging",
            "--max-chunks",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Retrieved knowledge context" in output
    assert "ampere.md#chunk-0" in output
    assert "cp.async" in output


def test_planning_context_includes_general_cuda_context(tmp_path: Path) -> None:
    lineage = tmp_path / "lineage"
    attempts = tmp_path / "attempts"
    lineage.mkdir()
    attempts.mkdir()

    args = SimpleNamespace(
        lineage=lineage,
        attempts_dir=attempts,
        attempt_limit=6,
        cwd=Path.cwd(),
        knowledge=Path("knowledge/ampere.md"),
    )

    _, _, _, knowledge = _planning_context(args)

    assert "-- b/cuda_general.md#chunk-" in knowledge
    assert "b/cuda_programming_practice.md" in knowledge
    assert "CUDA Kernel Design Practice" in knowledge
    assert "smallest coherent transformation" in knowledge


def test_general_cuda_context_supplements_missing_broad_files(tmp_path: Path) -> None:
    knowledge_root = tmp_path / "knowledge"
    broad_root = knowledge_root / "b"
    broad_root.mkdir(parents=True)
    (knowledge_root / "ampere.md").write_text(
        "# Ampere Notes\n\nLocal attention search evidence.\n",
        encoding="utf-8",
    )
    (broad_root / "cuda_general.md").write_text(
        "# General CUDA Working Knowledge\n\n"
        "## Execution Model\n\n"
        "CUDA execution model grid block thread warp SIMT divergence.\n",
        encoding="utf-8",
    )
    (broad_root / "cuda_programming_practice.md").write_text(
        "# CUDA Kernel Design Practice\n\n"
        "## Semantic Transform Guidance For The Planner\n\n"
        "Make the smallest coherent transformation that preserves invariants.\n",
        encoding="utf-8",
    )

    context = _with_general_cuda_context(
        "Retrieved knowledge context from local corpus.\n"
        "Knowledge source: knowledge/ampere.md\n"
        "-- ampere.md#chunk-0 lines 1-3 score=1.000 --\n"
        "Local attention search evidence.",
        knowledge_root / "ampere.md",
    )

    assert "Supplemental general CUDA grounding context" in context
    assert "b/cuda_general.md" in context
    assert "General CUDA Working Knowledge" in context
    assert "Supplemental broad CUDA practice context" in context
    assert "b/cuda_programming_practice.md" in context
    assert "smallest coherent transformation" in context


def test_baseline_build_env_uses_python_cuda_home(monkeypatch) -> None:
    monkeypatch.setattr("avo.cuda_env.compatible_python_cuda_home", lambda env: "/venv/nvidia/cu13")
    monkeypatch.setattr("avo.cuda_env.cuda_env_is_build_compatible", lambda env: False)

    updated = _baseline_build_env({})

    assert updated["CUDA_HOME"] == "/venv/nvidia/cu13"


def test_baseline_build_env_preserves_explicit_cuda_home(monkeypatch) -> None:
    monkeypatch.setattr("avo.cuda_env.compatible_python_cuda_home", lambda env: "/venv/nvidia/cu13")
    monkeypatch.setattr("avo.cuda_env.cuda_env_is_build_compatible", lambda env: True)

    updated = _baseline_build_env({"CUDA_HOME": "/usr/local/cuda"})

    assert updated["CUDA_HOME"] == "/usr/local/cuda"


def test_python_cuda_home_finds_single_pip_cuda_root(tmp_path: Path, monkeypatch) -> None:
    cuda_home = tmp_path / "site-packages" / "nvidia" / "cu13"
    (cuda_home / "bin").mkdir(parents=True)
    (cuda_home / "include").mkdir()
    (cuda_home / "bin" / "nvcc").write_text("", encoding="utf-8")
    (cuda_home / "include" / "cuda.h").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "avo.cuda_env.sysconfig.get_paths",
        lambda: {"purelib": str(tmp_path / "site-packages")},
    )

    assert python_cuda_home() == str(cuda_home)


def test_parse_nvcc_release() -> None:
    output = "Cuda compilation tools, release 12.9, V12.9.86"

    assert parse_nvcc_release(output) == "12.9"
    assert parse_nvcc_release("not nvcc output") is None


def test_cuda_build_compatibility() -> None:
    assert cuda_build_compatibility("13.0", "13.0") == ("exact", None)

    minor_status, minor_warning = cuda_build_compatibility("13.0", "13.1")
    assert minor_status == "minor_mismatch"
    assert "may warn" in minor_warning

    major_status, major_warning = cuda_build_compatibility("13.0", "12.9")
    assert major_status == "major_mismatch"
    assert "will fail" in major_warning


def test_nvcc_path_prefers_cuda_home() -> None:
    assert nvcc_path_from_env({"CUDA_HOME": "/opt/cuda", "PATH": "/bin"}) == (
        "/opt/cuda/bin/nvcc"
    )


def test_prepare_torch_extension_env_uses_python_cuda_home(monkeypatch) -> None:
    monkeypatch.setattr("avo.cuda_env.compatible_python_cuda_home", lambda env: "/venv/nvidia/cu13")
    monkeypatch.setattr("avo.cuda_env.cuda_env_is_build_compatible", lambda env: False)

    env: dict[str, str] = {
        "CPATH": "/usr/local/cuda-12.9/include:/opt/include",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.9/lib64:/lib",
        "PATH": "/usr/local/cuda-12.9/bin:/usr/bin",
    }
    updated = prepare_torch_extension_env(env, max_jobs="3")

    assert updated is env
    assert updated["TORCH_CUDA_ARCH_LIST"] == "8.6"
    assert updated["MAX_JOBS"] == "3"
    assert updated["CUDA_HOME"] == "/venv/nvidia/cu13"
    assert updated["CUDA_PATH"] == "/venv/nvidia/cu13"
    assert updated["CUDACXX"] == "/venv/nvidia/cu13/bin/nvcc"
    assert updated["CPATH"].split(":") == ["/venv/nvidia/cu13/include", "/opt/include"]
    assert updated["LIBRARY_PATH"].split(":") == ["/venv/nvidia/cu13/lib"]
    assert updated["LD_LIBRARY_PATH"].split(":") == ["/venv/nvidia/cu13/lib", "/lib"]
    assert updated["PATH"].split(":") == ["/venv/nvidia/cu13/bin", "/usr/bin"]


def test_prepare_torch_extension_env_adds_cudart_link_dir(tmp_path: Path, monkeypatch) -> None:
    cuda_home = tmp_path / "site-packages" / "nvidia" / "cu13"
    (cuda_home / "lib").mkdir(parents=True)
    cudart = cuda_home / "lib" / "libcudart.so.13"
    cudart.write_text("", encoding="utf-8")
    link_dir = tmp_path / "cuda-links"
    monkeypatch.setenv("AVO_CUDA_LINK_DIR", str(link_dir))
    monkeypatch.setattr("avo.cuda_env.compatible_python_cuda_home", lambda env: str(cuda_home))
    monkeypatch.setattr("avo.cuda_env.cuda_env_is_build_compatible", lambda env: False)

    env: dict[str, str] = {}
    prepare_torch_extension_env(env)

    assert (link_dir / "libcudart.so").resolve() == cudart.resolve()
    assert env["LIBRARY_PATH"].split(":")[:2] == [str(link_dir), str(cuda_home / "lib")]
    assert env["LD_LIBRARY_PATH"].split(":")[:2] == [str(link_dir), str(cuda_home / "lib")]


def test_torch_extension_worker_env_exposes_python_bin_for_ninja(
    tmp_path: Path,
    monkeypatch,
) -> None:
    python_bin = tmp_path / "venv" / "bin"
    python_bin.mkdir(parents=True)
    python_executable = python_bin / "python"
    python_executable.write_text("", encoding="utf-8")

    def fake_prepare(env, *, max_jobs):
        env["MAX_JOBS"] = max_jobs
        return env

    monkeypatch.setattr("avo.cli.prepare_torch_extension_env", fake_prepare)
    monkeypatch.setattr("avo.cli.sys.executable", str(python_executable))

    updated = _torch_extension_worker_env({"PATH": "/usr/bin"})

    assert updated["PATH"].split(os.pathsep)[0] == str(python_bin)
    assert updated["MAX_JOBS"] == "1"


def profile_args(**overrides):
    values = {
        "backend": "candidate",
        "candidate": Path("candidates/cuda_mma_attention_seed.py"),
        "seq_lens": "4096",
        "causal": "false",
        "head_dim": 128,
        "num_heads": 16,
        "total_tokens": 32768,
        "dtype": "bf16",
        "warmup": 0,
        "repeats": 1,
        "trials": 1,
        "timeout_s": 10,
        "ncu_set": "basic",
        "section": ["Occupancy"],
        "kernel_name": "regex:.*attention.*",
        "launch_count": 1,
        "launch_skip": 0,
        "page": "raw",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_profile_wraps_worker_score_with_ncu(monkeypatch, tmp_path, capsys) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr("avo.cli.THUNDER_CUDA_SHIM", tmp_path / "missing-libthunder.so")
    monkeypatch.setattr("avo.cli.shutil.which", lambda name, path=None: "/usr/bin/ncu")
    monkeypatch.setattr("avo.cli._torch_extension_worker_env", lambda env: env)

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        stdout = (
            "==PROF== Profiling \"mma_attention_kernel\"\n"
            'AVO_RESULT_JSON={"all_correct": true, "geomean_tflops": 7.5}\n'
        )
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("avo.cli.subprocess.run", fake_run)

    exit_code = _profile(profile_args())

    payload = json.loads(capsys.readouterr().out)
    command = seen["command"]
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["profiler"]["profiled"] is True
    assert payload["profiler"]["timeout_s"] == 10
    assert payload["score_payload"]["all_correct"] is True
    assert command[:2] == ["/usr/bin/ncu", "--target-processes"]
    assert "--section" in command
    assert command[-2:] == [
        "--candidate",
        "candidates/cuda_mma_attention_seed.py",
    ]


def test_profile_reports_no_kernels_profiled(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("avo.cli.THUNDER_CUDA_SHIM", tmp_path / "missing-libthunder.so")
    monkeypatch.setattr("avo.cli.shutil.which", lambda name, path=None: "/usr/bin/ncu")
    monkeypatch.setattr("avo.cli._torch_extension_worker_env", lambda env: env)
    monkeypatch.setattr(
        "avo.cli.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="==WARNING== No kernels were profiled.\n",
            stderr="",
        ),
    )

    exit_code = _profile(profile_args())

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["profiler"]["profiled"] is False
    assert payload["profiler"]["error"] == "no_kernels_profiled"


def test_profile_reports_unsupported_runtime_preflight(
    monkeypatch, tmp_path, capsys
) -> None:
    thunder_shim = tmp_path / "libthunder.so"
    thunder_shim.write_text("", encoding="utf-8")
    monkeypatch.setattr("avo.cli.THUNDER_CUDA_SHIM", thunder_shim)
    monkeypatch.setattr("avo.cli.shutil.which", lambda name, path=None: "/usr/bin/ncu")
    monkeypatch.setattr("avo.cli._torch_extension_worker_env", lambda env: env)

    def fail_run(*args, **kwargs):
        raise AssertionError("profile preflight should not launch ncu")

    monkeypatch.setattr("avo.cli.subprocess.run", fail_run)

    exit_code = _profile(profile_args(timeout_s=1800))

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["command"] == []
    assert payload["profiler"]["profiled"] is False
    assert payload["profiler"]["error"] == "profiler_unsupported_runtime"
    assert payload["profiler"]["timeout_s"] == 120
    assert payload["profiler"]["requested_timeout_s"] == 1800


def test_baseline_build_status_reports_cuda_mismatch(monkeypatch) -> None:
    monkeypatch.setattr("avo.cli.nvcc_path_from_env", lambda env: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr("avo.cli.nvcc_cuda_version", lambda nvcc_path, env: ("12.9", None))
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


def test_score_command_contains_hard_candidate_crash(tmp_path: Path, capsys) -> None:
    candidate = tmp_path / "crashing_candidate.py"
    candidate.write_text("import os\nos._exit(139)\n", encoding="utf-8")

    exit_code = _score(
        SimpleNamespace(
            backend="candidate",
            candidate=candidate,
            seq_lens="16",
            causal="false",
            head_dim=16,
            num_heads=1,
            total_tokens=16,
            dtype="bf16",
            warmup=0,
            repeats=0,
            trials=1,
            timeout_s=10,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["returncode"] == 139
    assert payload["payload"] is None
    assert payload["timed_out"] is False


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

    def fake_run_decision_command(decision, *, cwd, timeout_s, env, **kwargs):
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


def test_evolve_once_records_planner_provider_exception(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    knowledge = tmp_path / "knowledge.md"
    knowledge.write_text("Ampere only.", encoding="utf-8")
    step_json = tmp_path / "step.json"

    def fake_request_variation_decision(**kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("avo.cli.request_variation_decision", fake_request_variation_decision)

    exit_code = _evolve_once(
        SimpleNamespace(
            lineage=tmp_path / "lineage",
            knowledge=knowledge,
            cwd=tmp_path,
            timeout_s=10,
            step_json=step_json,
            env_file=None,
            model="claude",
            attempts_dir=None,
            attempt_limit=5,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    persisted = json.loads(step_json.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert persisted == payload
    assert payload["attempt"]["decision"]["hypothesis"] == "agent planning failed validation"
    assert payload["attempt"]["command_result"]["ok"] is False
    assert "RuntimeError: provider unavailable" in (
        payload["attempt"]["command_result"]["stderr_tail"]
    )


def test_evolve_once_repairs_candidate_compile_failure_before_finishing(
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
    first_decision = VariationDecision(
        hypothesis="introduce compile failure",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="change value in a way that fails compile",
        expected_effect="exercise compile repair",
        risk="mock compile failure",
        next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
        candidate_patch=dedent(
            """\
            diff --git a/candidates/seed.py b/candidates/seed.py
            --- a/candidates/seed.py
            +++ b/candidates/seed.py
            @@ -1 +1 @@
            -VALUE = 1
            +VALUE = bad
            """
        ),
    )
    repair_decision = VariationDecision(
        hypothesis="repair compile failure",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="change value with valid syntax",
        expected_effect="build should pass",
        risk="compile-only repair",
        next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
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
    decisions = [first_decision, repair_decision]
    seen_attempt_histories: list[str] = []

    def fake_request_variation_decision(**kwargs):
        seen_attempt_histories.append(kwargs["attempt_history"])
        return decisions.pop(0)

    def fake_run_decision_command(decision, *, cwd, timeout_s, env, **kwargs):
        patch_result = apply_candidate_patch(decision.candidate_patch, cwd=cwd)
        ok = decision.hypothesis == "repair compile failure"
        return VariationAttempt(
            decision=decision,
            command_result=CommandResult(
                command=["python", "-m", "avo", "compile"],
                returncode=0 if ok else 1,
                timed_out=False,
                stdout_tail="",
                stderr_tail=(
                    ""
                    if ok
                    else "attention_kernel.cu(4): error: identifier bad is undefined"
                ),
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
            step_json=None,
            env_file=None,
            model="claude",
            attempts_dir=None,
            attempt_limit=5,
            compile_repair_attempts=1,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["attempt"]["decision"]["hypothesis"] == "repair compile failure"
    assert len(payload["repair_attempts"]) == 1
    assert payload["repair_cleanup_results"][0]["ok"] is True
    assert payload["patch_cleanup_result"]["ok"] is True
    assert "Immediate compile-repair request" in seen_attempt_histories[1]
    assert "compiler_diagnostic_summary" in seen_attempt_histories[1]
    assert "locations=attention_kernel.cu:4" in seen_attempt_histories[1]
    assert "symbols=bad" in seen_attempt_histories[1]
    assert "identifier bad is undefined" in seen_attempt_histories[1]
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_compile_repair_prompt_gives_async_copy_guidance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    seed = candidate_dir / "seed.py"
    seed.write_text("VALUE = 1\n", encoding="utf-8")
    failed_decision = VariationDecision(
        hypothesis="introduce async copy compile failure",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="add async copy dataflow with a bad primitive call",
        expected_effect="exercise async copy compile repair",
        risk="mock compile failure",
        next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
        candidate_patch=dedent(
            """\
            diff --git a/candidates/seed.py b/candidates/seed.py
            --- a/candidates/seed.py
            +++ b/candidates/seed.py
            @@ -1 +1 @@
            -VALUE = 1
            +VALUE = bad
            """
        ),
    )
    failed_patch_result = apply_candidate_patch(failed_decision.candidate_patch, cwd=tmp_path)
    failed_attempt = VariationAttempt(
        decision=failed_decision,
        command_result=CommandResult(
            command=["python", "-m", "avo", "compile"],
            returncode=1,
            timed_out=False,
            stdout_tail="",
            stderr_tail=(
                "attention_kernel.cu(4): error: identifier "
                "__pipeline_memcpy_async is undefined"
            ),
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        patch_result=failed_patch_result,
    )
    repair_decision = VariationDecision(
        hypothesis="repair async copy compile failure",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="replace failed edit with a compileable source change",
        expected_effect="build should pass",
        risk="mock repaired compile",
        next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
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
    seen_attempt_histories: list[str] = []

    def fake_request_variation_decision(**kwargs):
        seen_attempt_histories.append(kwargs["attempt_history"])
        return repair_decision

    def fake_run_decision_command(decision, *, cwd, timeout_s, env, **kwargs):
        return VariationAttempt(
            decision=decision,
            command_result=CommandResult(
                command=["python", "-m", "avo", "compile"],
                returncode=0,
                timed_out=False,
                stdout_tail="",
                stderr_tail="",
            ),
            started_at="2026-05-08T00:00:02+00:00",
            completed_at="2026-05-08T00:00:03+00:00",
            patch_result=PatchResult(
                ok=True,
                patch_paths=["candidates/seed.py"],
                returncode=0,
                stdout_tail="",
                stderr_tail="",
            ),
        )

    monkeypatch.setattr("avo.cli.request_variation_decision", fake_request_variation_decision)
    monkeypatch.setattr("avo.cli.run_decision_command", fake_run_decision_command)

    result = _run_compile_repair_loop(
        SimpleNamespace(
            cwd=tmp_path,
            timeout_s=10,
            attempts_dir=None,
            compile_repair_attempts=1,
            model="claude",
        ),
        initial_attempt=failed_attempt,
        lineage_summary="{}",
        attempt_history="",
        repo_context="",
        knowledge="Ampere only.",
    )

    assert not isinstance(result, EvolutionStep)
    assert "failure_class=async_copy_compile_error" in seen_attempt_histories[0]
    assert "repair the async-copy API/include/stage/dataflow issue" in (
        seen_attempt_histories[0]
    )
    assert "do not treat copy granularity alone as a hard rejection" in (
        seen_attempt_histories[0]
    )


def test_evolve_once_rejects_repair_that_repeats_earlier_failed_payload(
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

    first_decision = VariationDecision(
        hypothesis="introduce compile failure",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="change value in a way that fails compile",
        expected_effect="exercise compile repair",
        risk="mock compile failure",
        next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
        candidate_patch=dedent(
            """\
            diff --git a/candidates/seed.py b/candidates/seed.py
            --- a/candidates/seed.py
            +++ b/candidates/seed.py
            @@ -1 +1 @@
            -VALUE = 1
            +VALUE = bad
            """
        ),
    )
    first_repair = VariationDecision(
        hypothesis="try different compile repair",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="try a different invalid value",
        expected_effect="exercise second repair request",
        risk="mock compile failure",
        next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
        candidate_patch=dedent(
            """\
            diff --git a/candidates/seed.py b/candidates/seed.py
            --- a/candidates/seed.py
            +++ b/candidates/seed.py
            @@ -1 +1 @@
            -VALUE = 1
            +VALUE = worse
            """
        ),
    )
    repeated_original = VariationDecision(
        hypothesis="repeat original compile failure",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="accidentally repeat the original failed edit",
        expected_effect="should be rejected before execution",
        risk="mock repeated repair",
        next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
        candidate_patch=first_decision.candidate_patch,
    )
    decisions = [first_decision, first_repair, repeated_original, repeated_original]
    seen_attempt_histories: list[str] = []
    executed_hypotheses: list[str] = []

    def fake_request_variation_decision(**kwargs):
        seen_attempt_histories.append(kwargs["attempt_history"])
        return decisions.pop(0)

    def fake_run_decision_command(decision, *, cwd, timeout_s, env, **kwargs):
        executed_hypotheses.append(decision.hypothesis)
        patch_result = apply_candidate_patch(decision.candidate_patch, cwd=cwd)
        return VariationAttempt(
            decision=decision,
            command_result=CommandResult(
                command=["python", "-m", "avo", "compile"],
                returncode=1,
                timed_out=False,
                stdout_tail="",
                stderr_tail=(
                    "attention_kernel.cu(4): error: identifier bad is undefined"
                ),
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
            step_json=None,
            env_file=None,
            model="claude",
            attempts_dir=None,
            attempt_limit=5,
            compile_repair_attempts=2,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert executed_hypotheses == [
        "introduce compile failure",
        "try different compile repair",
    ]
    assert "Earlier failed edit payloads in this repair episode" in seen_attempt_histories[2]
    assert "VALUE = bad" in seen_attempt_histories[2]
    assert "Repair validation feedback" in seen_attempt_histories[3]
    assert "was not executed" in seen_attempt_histories[3]
    assert "repeats an earlier failed edit payload" in payload["attempt"][
        "command_result"
    ]["stderr_tail"]
    assert len(payload["repair_attempts"]) == 2
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_evolve_once_repairs_transform_materialization_failure_before_finishing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    seed = candidate_dir / "seed.py"
    seed.write_text("VALUE = 1\nANCHOR\nVALUE = 2\nANCHOR\n", encoding="utf-8")
    knowledge = tmp_path / "knowledge.md"
    knowledge.write_text("Ampere only.", encoding="utf-8")
    first_decision = VariationDecision(
        hypothesis="introduce ambiguous transform",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="insert after an ambiguous anchor",
        expected_effect="exercise transform repair",
        risk="mock materialization failure",
        next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
        edit_mode="transform",
        candidate_transform={
            "op": "insert_after_once",
            "path": "candidates/seed.py",
            "anchor": "ANCHOR",
            "text": "VALUE = 3\n",
        },
    )
    repair_decision = VariationDecision(
        hypothesis="repair ambiguous transform",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="insert after a unique anchor span",
        expected_effect="materialization should pass",
        risk="compile-only repair",
        next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
        edit_mode="transform",
        candidate_transform={
            "op": "insert_after_once",
            "path": "candidates/seed.py",
            "anchor": "VALUE = 1\nANCHOR",
            "text": "\nVALUE = 3",
        },
    )
    decisions = [first_decision, repair_decision]
    seen_attempt_histories: list[str] = []

    def fake_request_variation_decision(**kwargs):
        seen_attempt_histories.append(kwargs["attempt_history"])
        return decisions.pop(0)

    def fake_run_decision_command(decision, *, cwd, timeout_s, env, **kwargs):
        if decision.hypothesis == "introduce ambiguous transform":
            return VariationAttempt(
                decision=decision,
                command_result=CommandResult(
                    command=["python", "-m", "avo", "compile"],
                    returncode=None,
                    timed_out=False,
                    stdout_tail="",
                    stderr_tail=(
                        "candidate patch rejected: candidate transform rejected: insert "
                        "transform expected exactly one anchor, found 2"
                    ),
                ),
                started_at="2026-05-08T00:00:00+00:00",
                completed_at="2026-05-08T00:00:01+00:00",
                patch_result=PatchResult(
                    ok=False,
                    patch_paths=[],
                    returncode=None,
                    stdout_tail="",
                    stderr_tail="",
                    rejected_reason=(
                        "candidate transform rejected: insert transform expected exactly "
                        "one anchor, found 2"
                    ),
                ),
            )
        patch_text = materialize_candidate_transform(decision.candidate_transform, cwd=cwd)
        patch_result = apply_candidate_patch(patch_text, cwd=cwd)
        return VariationAttempt(
            decision=decision,
            command_result=CommandResult(
                command=["python", "-m", "avo", "compile"],
                returncode=0,
                timed_out=False,
                stdout_tail="",
                stderr_tail="",
            ),
            started_at="2026-05-08T00:00:00+00:00",
            completed_at="2026-05-08T00:00:01+00:00",
            patch_result=patch_result,
            materialized_patch=patch_text,
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
            compile_repair_attempts=1,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["attempt"]["decision"]["hypothesis"] == "repair ambiguous transform"
    assert len(payload["repair_attempts"]) == 1
    assert payload["repair_cleanup_results"] == []
    assert payload["patch_cleanup_result"]["ok"] is True
    assert "Immediate structured-transform materialization repair request" in (
        seen_attempt_histories[1]
    )
    assert "expected exactly one anchor" in seen_attempt_histories[1]
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\nANCHOR\nVALUE = 2\nANCHOR\n"


def test_evolve_once_repairs_candidate_correctness_failure_before_finishing(
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
    first_decision = VariationDecision(
        hypothesis="introduce correctness failure",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="change value in a way that scores incorrectly",
        expected_effect="exercise correctness repair",
        risk="mock correctness failure",
        next_command="avo score --backend candidate",
        candidate_patch=dedent(
            """\
            diff --git a/candidates/seed.py b/candidates/seed.py
            --- a/candidates/seed.py
            +++ b/candidates/seed.py
            @@ -1 +1 @@
            -VALUE = 1
            +VALUE = bad
            """
        ),
    )
    repair_decision = VariationDecision(
        hypothesis="repair correctness failure",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="change value in a way that scores correctly",
        expected_effect="score should pass",
        risk="mock repaired score",
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
    decisions = [first_decision, repair_decision]
    seen_attempt_histories: list[str] = []

    def fake_request_variation_decision(**kwargs):
        seen_attempt_histories.append(kwargs["attempt_history"])
        return decisions.pop(0)

    def fake_run_decision_command(decision, *, cwd, timeout_s, env, **kwargs):
        patch_result = apply_candidate_patch(decision.candidate_patch, cwd=cwd)
        ok = decision.hypothesis == "repair correctness failure"
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
                "all_correct": ok,
                "geomean_tflops": 3.0 if ok else 0.0,
                "cases": [
                    {}
                    if ok
                    else {
                        "correct": False,
                        "error": "max_abs_error exceeded tolerance",
                    }
                ],
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
            compile_repair_attempts=1,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["attempt"]["decision"]["hypothesis"] == "repair correctness failure"
    assert payload["gate_decision"]["accepted"] is True
    assert len(payload["repair_attempts"]) == 1
    assert payload["repair_cleanup_results"][0]["ok"] is True
    assert payload["patch_cleanup_result"] is None
    assert "Immediate correctness-repair request" in seen_attempt_histories[1]
    assert "all_correct=false" in seen_attempt_histories[1]
    assert "max_abs_error exceeded tolerance" in seen_attempt_histories[1]
    assert "already restored the clean pre-edit source" in seen_attempt_histories[1]
    assert "do not return a revert-only repair" in seen_attempt_histories[1]
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_evolve_once_repairs_score_time_extension_build_failure_before_finishing(
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
    first_decision = VariationDecision(
        hypothesis="introduce extension build failure",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="change value in a way that breaks extension build",
        expected_effect="exercise score-time compile repair",
        risk="mock extension build failure",
        next_command="avo score --backend candidate",
        candidate_patch=dedent(
            """\
            diff --git a/candidates/seed.py b/candidates/seed.py
            --- a/candidates/seed.py
            +++ b/candidates/seed.py
            @@ -1 +1 @@
            -VALUE = 1
            +VALUE = bad
            """
        ),
    )
    repair_decision = VariationDecision(
        hypothesis="repair extension build failure",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="change value in a way that builds and scores",
        expected_effect="score should pass",
        risk="mock repaired extension build",
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
    decisions = [first_decision, repair_decision]
    seen_attempt_histories: list[str] = []

    def fake_request_variation_decision(**kwargs):
        seen_attempt_histories.append(kwargs["attempt_history"])
        return decisions.pop(0)

    def fake_run_decision_command(decision, *, cwd, timeout_s, env, **kwargs):
        patch_result = apply_candidate_patch(decision.candidate_patch, cwd=cwd)
        ok = decision.hypothesis == "repair extension build failure"
        score_payload = {
            "backend": "candidate",
            "all_correct": ok,
            "geomean_tflops": 3.0 if ok else 0.0,
            "cases": [{}],
        }
        if not ok:
            score_payload = {
                "backend": "candidate",
                "candidate_path": "candidates/seed.py",
                "all_correct": False,
                "geomean_tflops": 0.0,
                "candidate_source_files": [
                    "candidates/dynamic_extension/attention.cpp",
                    "candidates/dynamic_extension/attention_kernel.cu",
                ],
                "cases": [
                    {
                        "correct": False,
                        "error": (
                            "RuntimeError: Error building extension 'runtime_demo': "
                            "identifier __pipeline_memcpy_async is undefined; "
                            "ninja: build stopped: nvcc failed compiling attention_kernel.cu"
                        ),
                    }
                ],
            }
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
            score_payload=score_payload,
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
            compile_repair_attempts=1,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["attempt"]["decision"]["hypothesis"] == "repair extension build failure"
    assert payload["gate_decision"]["accepted"] is True
    assert len(payload["repair_attempts"]) == 1
    assert payload["repair_cleanup_results"][0]["ok"] is True
    assert payload["patch_cleanup_result"] is None
    assert "Immediate score-time compile-repair request" in seen_attempt_histories[1]
    assert "score_time_compile_failure" in seen_attempt_histories[1]
    assert "score_build_diagnostic_summary" in seen_attempt_histories[1]
    assert "candidates/seed.py" in seen_attempt_histories[1]
    assert "candidates/dynamic_extension/attention_kernel.cu" in seen_attempt_histories[1]
    assert "locations=attention_kernel.cu" in seen_attempt_histories[1]
    assert "symbols=__pipeline_memcpy_async" in seen_attempt_histories[1]
    assert "Error building extension" in seen_attempt_histories[1]
    assert "repair the async-copy API/include/stage/dataflow issue" in (
        seen_attempt_histories[1]
    )
    assert "previous candidate edit compiled and ran" not in seen_attempt_histories[1]
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_evolve_once_repairs_candidate_worker_crash_before_finishing(
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
    first_decision = VariationDecision(
        hypothesis="introduce worker crash",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="change value in a way that crashes the isolated score worker",
        expected_effect="exercise worker crash repair",
        risk="mock worker crash",
        next_command="avo score --backend candidate",
        candidate_patch=dedent(
            """\
            diff --git a/candidates/seed.py b/candidates/seed.py
            --- a/candidates/seed.py
            +++ b/candidates/seed.py
            @@ -1 +1 @@
            -VALUE = 1
            +VALUE = bad
            """
        ),
    )
    repair_decision = VariationDecision(
        hypothesis="repair worker crash",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="change value in a way that scores without crashing",
        expected_effect="score should pass",
        risk="mock repaired worker crash",
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
    decisions = [first_decision, repair_decision]
    seen_attempt_histories: list[str] = []

    def fake_request_variation_decision(**kwargs):
        seen_attempt_histories.append(kwargs["attempt_history"])
        return decisions.pop(0)

    def fake_run_decision_command(decision, *, cwd, timeout_s, env, **kwargs):
        patch_result = apply_candidate_patch(decision.candidate_patch, cwd=cwd)
        ok = decision.hypothesis == "repair worker crash"
        return VariationAttempt(
            decision=decision,
            command_result=CommandResult(
                command=["python", "-m", "avo", "score"],
                returncode=0 if ok else 139,
                timed_out=False,
                stdout_tail="{}" if ok else '{"ok": false, "payload": null}',
                stderr_tail="" if ok else "Segmentation fault (core dumped)",
            ),
            started_at="2026-05-08T00:00:00+00:00",
            completed_at="2026-05-08T00:00:01+00:00",
            score_payload=(
                {
                    "backend": "mock",
                    "all_correct": True,
                    "geomean_tflops": 3.0,
                    "cases": [{}],
                }
                if ok
                else None
            ),
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
            compile_repair_attempts=1,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["attempt"]["decision"]["hypothesis"] == "repair worker crash"
    assert payload["gate_decision"]["accepted"] is True
    assert len(payload["repair_attempts"]) == 1
    assert payload["repair_attempts"][0]["command_result"]["returncode"] == 139
    assert payload["repair_cleanup_results"][0]["ok"] is True
    assert payload["patch_cleanup_result"] is None
    assert "Immediate worker-crash repair request" in seen_attempt_histories[1]
    assert "worker_returncode=139" in seen_attempt_histories[1]
    assert "invalid memory access" in seen_attempt_histories[1]
    assert "Segmentation fault" in seen_attempt_histories[1]
    assert "replaying the crashed payload" in seen_attempt_histories[1]
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_repair_loop_does_not_autofill_pending_transform(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    seed = candidate_dir / "seed.py"
    seed.write_text("VALUE = bad\n", encoding="utf-8")
    attempts = tmp_path / "attempts"
    transform = {
        "op": "replace_once",
        "path": "candidates/seed.py",
        "find": "VALUE = 1",
        "replace": "VALUE = bad",
    }
    compile_decision = VariationDecision(
        hypothesis="compile pending transform",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="compile transform",
        expected_effect="compile succeeds",
        risk="mock compile",
        next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
        edit_mode="transform",
        candidate_transform=transform,
    )
    compile_attempt = VariationAttempt(
        decision=compile_decision,
        command_result=CommandResult(
            command=["python", "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/seed.py"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
        ),
    )
    write_step_record(attempts, EvolutionStep(attempt=compile_attempt, gate_decision=None))
    failed_patch = dedent(
        """\
        diff --git a/candidates/seed.py b/candidates/seed.py
        --- a/candidates/seed.py
        +++ b/candidates/seed.py
        @@ -1 +1 @@
        -VALUE = 1
        +VALUE = bad
        """
    )
    failed_decision = VariationDecision(
        hypothesis="score pending transform",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="Score the previously compiled transform.",
        expected_effect="score should pass",
        risk="mock failed score",
        next_command="avo score --backend candidate --candidate candidates/seed.py",
        edit_mode="transform",
        candidate_transform=transform,
    )
    failed_attempt = VariationAttempt(
        decision=failed_decision,
        command_result=CommandResult(
            command=["python", "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="{}",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:02+00:00",
        completed_at="2026-05-08T00:00:03+00:00",
        score_payload={
            "backend": "mock",
            "all_correct": False,
            "geomean_tflops": 0.0,
            "cases": [{"correct": False, "error": "candidate output contains non-finite values"}],
        },
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/seed.py"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
        ),
        materialized_patch=failed_patch,
    )
    normalize_payloads = []
    seen_attempt_histories: list[str] = []

    def fake_request_variation_decision(**kwargs):
        normalize_payloads.append(kwargs.get("normalize_payload"))
        seen_attempt_histories.append(kwargs["attempt_history"])
        return VariationDecision(
            hypothesis="bad correctness repair",
            files_to_inspect=["candidates/seed.py"],
            candidate_edit="No edit; score the compiled transform again.",
            expected_effect="mock retry",
            risk="mock retry",
            next_command="avo score --backend candidate --candidate candidates/seed.py",
            edit_mode="no_edit",
        )

    monkeypatch.setattr("avo.cli.request_variation_decision", fake_request_variation_decision)

    result = _run_compile_repair_loop(
        SimpleNamespace(
            cwd=tmp_path,
            timeout_s=10,
            attempts_dir=attempts,
            compile_repair_attempts=1,
            model="claude",
        ),
        initial_attempt=failed_attempt,
        lineage_summary="{}",
        attempt_history="",
        repo_context="",
        knowledge="Ampere only.",
    )

    assert isinstance(result, EvolutionStep)
    assert normalize_payloads == [None, None]
    assert "Repair validation feedback" in seen_attempt_histories[1]
    assert "invalid repair decision was not executed" in seen_attempt_histories[1]
    assert "correctness repair decision must include a revised" in (
        result.attempt.command_result.stderr_tail
    )
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_repair_loop_retries_invalid_repair_decision_before_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    seed = candidate_dir / "seed.py"
    seed.write_text("VALUE = 1\n", encoding="utf-8")
    failed_decision = VariationDecision(
        hypothesis="introduce compile failure",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="change value in a way that fails compile",
        expected_effect="exercise compile repair",
        risk="mock compile failure",
        next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
        candidate_patch=dedent(
            """\
            diff --git a/candidates/seed.py b/candidates/seed.py
            --- a/candidates/seed.py
            +++ b/candidates/seed.py
            @@ -1 +1 @@
            -VALUE = 1
            +VALUE = bad
            """
        ),
    )
    failed_patch_result = apply_candidate_patch(failed_decision.candidate_patch, cwd=tmp_path)
    failed_attempt = VariationAttempt(
        decision=failed_decision,
        command_result=CommandResult(
            command=["python", "-m", "avo", "compile"],
            returncode=1,
            timed_out=False,
            stdout_tail="",
            stderr_tail="attention_kernel.cu(4): error: identifier bad is undefined",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        patch_result=failed_patch_result,
    )
    bad_repair = VariationDecision(
        hypothesis="bad no-edit repair",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="No edit; retry compile.",
        expected_effect="mock retry",
        risk="mock retry",
        next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
        edit_mode="no_edit",
    )
    good_repair = VariationDecision(
        hypothesis="valid compile repair",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="change value with valid syntax",
        expected_effect="build should pass",
        risk="mock repaired compile",
        next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
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
    decisions = [bad_repair, good_repair]
    seen_attempt_histories: list[str] = []
    executed_hypotheses: list[str] = []

    def fake_request_variation_decision(**kwargs):
        seen_attempt_histories.append(kwargs["attempt_history"])
        return decisions.pop(0)

    def fake_run_decision_command(decision, *, cwd, timeout_s, env, **kwargs):
        executed_hypotheses.append(decision.hypothesis)
        return VariationAttempt(
            decision=decision,
            command_result=CommandResult(
                command=["python", "-m", "avo", "compile"],
                returncode=0,
                timed_out=False,
                stdout_tail="",
                stderr_tail="",
            ),
            started_at="2026-05-08T00:00:02+00:00",
            completed_at="2026-05-08T00:00:03+00:00",
            patch_result=PatchResult(
                ok=True,
                patch_paths=["candidates/seed.py"],
                returncode=0,
                stdout_tail="",
                stderr_tail="",
            ),
        )

    monkeypatch.setattr("avo.cli.request_variation_decision", fake_request_variation_decision)
    monkeypatch.setattr("avo.cli.run_decision_command", fake_run_decision_command)

    result = _run_compile_repair_loop(
        SimpleNamespace(
            cwd=tmp_path,
            timeout_s=10,
            attempts_dir=None,
            compile_repair_attempts=1,
            model="claude",
        ),
        initial_attempt=failed_attempt,
        lineage_summary="{}",
        attempt_history="",
        repo_context="",
        knowledge="Ampere only.",
    )

    assert not isinstance(result, EvolutionStep)
    repaired_attempt, repair_attempts, repair_cleanup_results = result
    assert repaired_attempt.decision.hypothesis == "valid compile repair"
    assert repair_attempts == [failed_attempt]
    assert repair_cleanup_results[0].ok is True
    assert executed_hypotheses == ["valid compile repair"]
    assert "Repair validation feedback" in seen_attempt_histories[1]
    assert "no-edit retries do not repair failed executable edits" in seen_attempt_histories[1]


def test_repair_loop_records_planner_provider_exception(
    tmp_path: Path,
    monkeypatch,
) -> None:
    failed_attempt = VariationAttempt(
        decision=VariationDecision(
            hypothesis="introduce compile failure",
            files_to_inspect=["candidates/seed.py"],
            candidate_edit="change value in a way that fails compile",
            expected_effect="exercise compile repair",
            risk="mock compile failure",
            next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch=dedent(
                """\
                diff --git a/candidates/seed.py b/candidates/seed.py
                --- a/candidates/seed.py
                +++ b/candidates/seed.py
                @@ -1 +1 @@
                -VALUE = 1
                +VALUE = bad
                """
            ),
        ),
        command_result=CommandResult(
            command=["python", "-m", "avo", "compile"],
            returncode=1,
            timed_out=False,
            stdout_tail="",
            stderr_tail="attention_kernel.cu(4): error: identifier bad is undefined",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/seed.py"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
        ),
    )

    def fake_request_variation_decision(**kwargs):
        raise RuntimeError("provider unavailable during repair")

    monkeypatch.setattr("avo.cli.request_variation_decision", fake_request_variation_decision)
    monkeypatch.setattr("avo.cli.cleanup_rejected_candidate_patch", lambda step, *, cwd: step)

    result = _run_compile_repair_loop(
        SimpleNamespace(
            cwd=tmp_path,
            timeout_s=10,
            attempts_dir=None,
            compile_repair_attempts=1,
            model="claude",
        ),
        initial_attempt=failed_attempt,
        lineage_summary="{}",
        attempt_history="",
        repo_context="",
        knowledge="Ampere only.",
    )

    assert isinstance(result, EvolutionStep)
    assert result.repair_attempts == (failed_attempt,)
    assert "RuntimeError: provider unavailable during repair" in (
        result.attempt.command_result.stderr_tail
    )


def test_repair_loop_allows_correctness_repair_after_pending_transform_score(
    tmp_path: Path,
    monkeypatch,
) -> None:
    attempts = tmp_path / "attempts"
    pending_transform = {
        "op": "replace_once",
        "path": "candidates/seed.py",
        "find": "VALUE = 1",
        "replace": "VALUE = bad",
    }
    repair_transform = {
        "op": "replace_once",
        "path": "candidates/seed.py",
        "find": "VALUE = 1",
        "replace": "VALUE = 2",
    }
    compile_attempt = VariationAttempt(
        decision=VariationDecision(
            hypothesis="compile pending transform",
            files_to_inspect=["candidates/seed.py"],
            candidate_edit="compile transform",
            expected_effect="compile succeeds",
            risk="mock compile",
            next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
            edit_mode="transform",
            candidate_transform=pending_transform,
        ),
        command_result=CommandResult(
            command=["python", "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/seed.py"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
        ),
    )
    write_step_record(attempts, EvolutionStep(attempt=compile_attempt, gate_decision=None))
    failed_score_attempt = VariationAttempt(
        decision=VariationDecision(
            hypothesis="score pending transform",
            files_to_inspect=["candidates/seed.py"],
            candidate_edit="Score the pending transform.",
            expected_effect="score should pass",
            risk="mock failed score",
            next_command="avo score --backend candidate --candidate candidates/seed.py",
            edit_mode="transform",
            candidate_transform=pending_transform,
        ),
        command_result=CommandResult(
            command=["python", "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="{}",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:02+00:00",
        completed_at="2026-05-08T00:00:03+00:00",
        score_payload={
            "backend": "mock",
            "all_correct": False,
            "geomean_tflops": 0.0,
            "cases": [{"correct": False, "error": "candidate output contains non-finite values"}],
        },
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/seed.py"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
        ),
    )
    repair_decision = VariationDecision(
        hypothesis="repair correctness failure",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="Repair the pending transform with initialized output.",
        expected_effect="score should pass",
        risk="mock repair",
        next_command="avo score --backend candidate --candidate candidates/seed.py",
        edit_mode="transform",
        candidate_transform=repair_transform,
    )

    def fake_request_variation_decision(**kwargs):
        return repair_decision

    def fake_run_decision_command(decision, *, cwd, timeout_s, env, **kwargs):
        return VariationAttempt(
            decision=decision,
            command_result=CommandResult(
                command=["python", "-m", "avo", "score"],
                returncode=0,
                timed_out=False,
                stdout_tail="{}",
                stderr_tail="",
            ),
            started_at="2026-05-08T00:00:04+00:00",
            completed_at="2026-05-08T00:00:05+00:00",
            score_payload={
                "backend": "mock",
                "all_correct": True,
                "geomean_tflops": 10.0,
                "cases": [{}],
            },
            patch_result=PatchResult(
                ok=True,
                patch_paths=["candidates/seed.py"],
                returncode=0,
                stdout_tail="",
                stderr_tail="",
            ),
        )

    monkeypatch.setattr("avo.cli.request_variation_decision", fake_request_variation_decision)
    monkeypatch.setattr("avo.cli.run_decision_command", fake_run_decision_command)
    monkeypatch.setattr("avo.cli.cleanup_rejected_candidate_patch", lambda step, *, cwd: step)

    result = _run_compile_repair_loop(
        SimpleNamespace(
            cwd=tmp_path,
            timeout_s=10,
            attempts_dir=attempts,
            compile_repair_attempts=1,
            model="claude",
        ),
        initial_attempt=failed_score_attempt,
        lineage_summary="{}",
        attempt_history="",
        repo_context="",
        knowledge="Ampere only.",
    )

    assert not isinstance(result, EvolutionStep)
    repaired_attempt, repair_attempts, _ = result
    assert repaired_attempt.decision.candidate_transform == repair_transform
    assert repair_attempts == [failed_score_attempt]


def test_pending_transform_payload_normalizer_attaches_transform_to_score(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "replace_once",
        "path": "candidates/seed.py",
        "find": "VALUE = 1",
        "replace": "VALUE = 2",
    }
    compile_decision = VariationDecision(
        hypothesis="compile transform",
        files_to_inspect=["candidates/seed.py"],
        candidate_edit="change value",
        expected_effect="compile checks the edit",
        risk="mock compile",
        next_command="avo compile --source candidates/kernel.cu --out-dir build/kernel",
        edit_mode="transform",
        candidate_transform=transform,
    )
    compile_attempt = VariationAttempt(
        decision=compile_decision,
        command_result=CommandResult(
            command=["python", "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/seed.py"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
        ),
    )
    write_step_record(attempts, EvolutionStep(attempt=compile_attempt, gate_decision=None))

    normalize = _pending_transform_payload_normalizer(attempts)

    assert normalize is not None
    payload = normalize(
        {
            "hypothesis": "score compiled transform",
            "files_to_inspect": ["candidates/seed.py"],
            "candidate_edit": "Score the previously compiled transform.",
            "candidate_patch": "",
            "edit_mode": "transform",
            "expected_effect": "records the score",
            "risk": "mock score",
            "next_command": "avo score --backend candidate --candidate candidates/seed.py",
        }
    )
    assert payload["candidate_transform"] == transform
    assert payload["candidate_patch"] == ""
    assert payload["edit_mode"] == "transform"

    prose_payload = normalize(
        {
            "hypothesis": "follow up on a compiled transform",
            "files_to_inspect": ["candidates/seed.py"],
            "candidate_edit": "Score the previously compiled transform.",
            "candidate_patch": "",
            "edit_mode": "transform",
            "expected_effect": "records the score",
            "risk": "mock score",
            "next_command": "avo env",
        }
    )
    assert prose_payload["candidate_transform"] == transform

    no_edit_payload = normalize(
        {
            "hypothesis": "score pending compiled transform",
            "files_to_inspect": ["candidates/seed.py"],
            "candidate_edit": "Score the previously compiled transform.",
            "candidate_patch": "",
            "edit_mode": "no_edit",
            "expected_effect": "records the score",
            "risk": "mock score",
            "next_command": "avo score --backend candidate --candidate candidates/seed.py",
        }
    )
    assert no_edit_payload["candidate_transform"] == transform
    assert no_edit_payload["edit_mode"] == "transform"

    malformed_payload = normalize(
        {
            "hypothesis": "score pending compiled transform",
            "files_to_inspect": ["candidates/seed.py"],
            "candidate_edit": "Score the previously compiled transform.",
            "candidate_patch": "",
            "candidate_transform": {},
            "edit_mode": "transform",
            "expected_effect": "records the score",
            "risk": "mock score",
            "next_command": "avo score --backend candidate --candidate candidates/seed.py",
        }
    )
    assert malformed_payload["candidate_transform"] == transform

    placeholder_patch_payload = normalize(
        {
            "hypothesis": "score pending compiled transform",
            "files_to_inspect": ["candidates/seed.py"],
            "candidate_edit": "Score the compiled transform that changes VALUE.",
            "candidate_patch": "reuse the compiled structured transform",
            "edit_mode": "transform",
            "expected_effect": "records the score",
            "risk": "mock score",
            "next_command": "avo score --backend candidate --candidate candidates/seed.py",
        }
    )
    assert placeholder_patch_payload["candidate_patch"] == ""
    assert placeholder_patch_payload["candidate_transform"] == transform


def test_pending_transform_payload_normalizer_rewrites_repeated_mma_compile_to_score(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "replace_once",
        "path": "candidates/cuda_mma_attention/attention_kernel.cu",
        "find": "constexpr int kThreads = 96;",
        "replace": "constexpr int kThreads = 128;",
    }
    compile_decision = VariationDecision(
        hypothesis="compile transform",
        files_to_inspect=["candidates/cuda_mma_attention/attention_kernel.cu"],
        candidate_edit="Retune kThreads.",
        expected_effect="compile checks the edit",
        risk="mock compile",
        next_command=(
            "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
            "--out-dir build/kernel"
        ),
        edit_mode="transform",
        candidate_transform=transform,
    )
    compile_attempt = VariationAttempt(
        decision=compile_decision,
        command_result=CommandResult(
            command=["python", "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/cuda_mma_attention/attention_kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
        ),
    )
    write_step_record(attempts, EvolutionStep(attempt=compile_attempt, gate_decision=None))

    normalize = _pending_transform_payload_normalizer(attempts)

    assert normalize is not None
    payload = normalize(
        {
            "hypothesis": "compile transform again",
            "files_to_inspect": ["candidates/cuda_mma_attention/attention_kernel.cu"],
            "candidate_edit": "Retune kThreads.",
            "candidate_patch": "",
            "candidate_transform": transform,
            "edit_mode": "transform",
            "expected_effect": "compile checks the edit",
            "risk": "mock compile",
            "next_command": (
                "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
                "--out-dir build/kernel"
            ),
        }
    )

    assert payload["candidate_transform"] == transform
    assert payload["candidate_patch"] == ""
    assert payload["edit_mode"] == "transform"
    assert payload["next_command"].startswith("avo score --backend candidate")
    assert "--candidate candidates/cuda_mma_attention_seed.py" in payload["next_command"]
    assert "--seq-lens 4096,8192,16384,32768" in payload["next_command"]


def test_pending_transform_payload_normalizer_forces_score_before_new_mma_compile(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    pending_transform = {
        "op": "replace_once",
        "path": "candidates/cuda_mma_attention/attention_kernel.cu",
        "find": "constexpr int kThreads = 96;",
        "replace": "constexpr int kThreads = 80;",
    }
    compile_decision = VariationDecision(
        hypothesis="compile transform",
        files_to_inspect=["candidates/cuda_mma_attention/attention_kernel.cu"],
        candidate_edit="Retune kThreads.",
        expected_effect="compile checks the edit",
        risk="mock compile",
        next_command=(
            "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
            "--out-dir build/kernel"
        ),
        edit_mode="transform",
        candidate_transform=pending_transform,
    )
    compile_attempt = VariationAttempt(
        decision=compile_decision,
        command_result=CommandResult(
            command=["python", "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/cuda_mma_attention/attention_kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
        ),
    )
    write_step_record(attempts, EvolutionStep(attempt=compile_attempt, gate_decision=None))

    normalize = _pending_transform_payload_normalizer(attempts)

    assert normalize is not None
    payload = normalize(
        {
            "hypothesis": "try a different compile-only transform",
            "files_to_inspect": ["candidates/cuda_mma_attention/attention_kernel.cu"],
            "candidate_edit": "Add K shared staging.",
            "candidate_patch": "",
            "candidate_transform": {
                "op": "insert_after_once",
                "path": "candidates/cuda_mma_attention/attention_kernel.cu",
                "anchor": "__shared__ float old_scale[kTile];",
                "text": "\n  __shared__ __nv_bfloat16 k_tile_shared[kTile * kHeadDim];",
            },
            "edit_mode": "transform",
            "expected_effect": "compile checks a different edit",
            "risk": "mock compile",
            "next_command": (
                "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
                "--out-dir build/kernel2"
            ),
        }
    )

    assert payload["candidate_transform"] == pending_transform
    assert payload["candidate_patch"] == ""
    assert payload["edit_mode"] == "transform"
    assert payload["next_command"].startswith("avo score --backend candidate")
    assert "--candidate candidates/cuda_mma_attention_seed.py" in payload["next_command"]


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

    def fake_run_decision_command(decision, *, cwd, timeout_s, env, **kwargs):
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
    seen_knowledge_contexts = []

    def fake_request_variation_decision(**kwargs):
        seen_attempt_histories.append(kwargs["attempt_history"])
        seen_knowledge_contexts.append(kwargs["knowledge"])
        return decisions.pop(0)

    def fake_run_decision_command(decision, *, cwd, timeout_s, env, **kwargs):
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
    assert "Retrieved knowledge context" in seen_knowledge_contexts[0]
    assert "Ampere only." in seen_knowledge_contexts[0]
    assert json.loads((tmp_path / "loop.json").read_text(encoding="utf-8"))["accepted"] is True


def test_evolve_loop_stops_between_steps_after_wall_time(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    knowledge = tmp_path / "knowledge.md"
    knowledge.write_text("Ampere only.", encoding="utf-8")
    decisions = [
        loop_decision("first rejected"),
        loop_decision("should not run"),
    ]
    monotonic_values = iter((0.0, 11.0, 11.0))

    def fake_request_variation_decision(**kwargs):
        return decisions.pop(0)

    def fake_run_decision_command(decision, *, cwd, timeout_s, env, **kwargs):
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
                "geomean_tflops": 0.0,
                "cases": [{}],
            },
        )

    monkeypatch.setattr("avo.cli.request_variation_decision", fake_request_variation_decision)
    monkeypatch.setattr("avo.cli.run_decision_command", fake_run_decision_command)
    monkeypatch.setattr("avo.cli.time.monotonic", lambda: next(monotonic_values))

    exit_code = _evolve_loop(
        SimpleNamespace(
            lineage=tmp_path / "lineage",
            knowledge=knowledge,
            cwd=tmp_path,
            timeout_s=10,
            env_file=None,
            model="claude",
            max_steps=5,
            max_wall_time_s=10,
            loop_json=tmp_path / "loop.json",
            attempts_dir=tmp_path / "attempts",
            attempt_limit=5,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["accepted"] is False
    assert payload["completed_steps"] == 1
    assert payload["max_steps"] == 5
    assert payload["max_wall_time_s"] == 10
    assert payload["elapsed_wall_time_s"] == 11.0
    assert payload["stopped_reason"] == "max_wall_time"
    assert [step["attempt"]["decision"]["hypothesis"] for step in payload["steps"]] == [
        "first rejected"
    ]
    assert len(decisions) == 1
    assert json.loads((tmp_path / "loop.json").read_text(encoding="utf-8")) == payload


def test_evolve_loop_scores_pending_compile_transform_at_step_limit(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    knowledge = tmp_path / "knowledge.md"
    knowledge.write_text("Ampere only.", encoding="utf-8")
    transform = {
        "op": "replace_once",
        "path": "candidates/cuda_mma_attention/attention_kernel.cu",
        "find": "constexpr int kThreads = 96;",
        "replace": "constexpr int kThreads = 128;",
    }
    decisions = [
        VariationDecision(
            hypothesis="compile structured transform",
            files_to_inspect=["candidates/cuda_mma_attention/attention_kernel.cu"],
            candidate_edit="Retune the MMA thread count.",
            expected_effect="compile checks the edit",
            risk="mock compile",
            next_command=(
                "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
                "--out-dir build/kernel"
            ),
            edit_mode="transform",
            candidate_transform=transform,
        )
    ]
    seen_commands = []
    seen_transforms = []

    def fake_request_variation_decision(**kwargs):
        return decisions.pop(0)

    def fake_run_decision_command(decision, *, cwd, timeout_s, env, **kwargs):
        seen_commands.append(decision.next_command)
        seen_transforms.append(decision.candidate_transform)
        if decision.next_command.startswith("avo compile "):
            return VariationAttempt(
                decision=decision,
                command_result=CommandResult(
                    command=["python", "-m", "avo", "compile"],
                    returncode=0,
                    timed_out=False,
                    stdout_tail="",
                    stderr_tail="",
                ),
                started_at="2026-05-08T00:00:00+00:00",
                completed_at="2026-05-08T00:00:01+00:00",
                patch_result=PatchResult(
                    ok=True,
                    patch_paths=["candidates/cuda_mma_attention/attention_kernel.cu"],
                    returncode=0,
                    stdout_tail="",
                    stderr_tail="",
                ),
            )
        return VariationAttempt(
            decision=decision,
            command_result=CommandResult(
                command=["python", "-m", "avo", "score"],
                returncode=0,
                timed_out=False,
                stdout_tail="{}",
                stderr_tail="",
            ),
            started_at="2026-05-08T00:00:02+00:00",
            completed_at="2026-05-08T00:00:03+00:00",
            score_payload={
                "backend": "mock",
                "all_correct": True,
                "geomean_tflops": 3.0,
                "cases": [{}],
            },
            patch_result=PatchResult(
                ok=True,
                patch_paths=["candidates/cuda_mma_attention/attention_kernel.cu"],
                returncode=0,
                stdout_tail="",
                stderr_tail="",
            ),
        )

    monkeypatch.setattr("avo.cli.request_variation_decision", fake_request_variation_decision)
    monkeypatch.setattr("avo.cli.run_decision_command", fake_run_decision_command)
    monkeypatch.setattr("avo.cli.cleanup_rejected_candidate_patch", lambda step, *, cwd: step)

    exit_code = _evolve_loop(
        SimpleNamespace(
            lineage=tmp_path / "lineage",
            knowledge=knowledge,
            cwd=tmp_path,
            timeout_s=10,
            env_file=None,
            model="claude",
            max_steps=1,
            loop_json=tmp_path / "loop.json",
            attempts_dir=tmp_path / "attempts",
            attempt_limit=5,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["accepted"] is True
    assert payload["completed_steps"] == 2
    assert payload["max_steps"] == 1
    assert payload["stopped_reason"] == "accepted"
    assert seen_commands[0].startswith("avo compile ")
    assert seen_commands[1].startswith("avo score ")
    assert seen_transforms == [transform, transform]
    assert not decisions
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 2
    assert json.loads((tmp_path / "loop.json").read_text(encoding="utf-8")) == payload


def test_evolve_loop_records_planning_validation_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    knowledge = tmp_path / "knowledge.md"
    knowledge.write_text("Ampere only.", encoding="utf-8")

    def fake_request_variation_decision(**kwargs):
        raise ValueError("invalid seed score shape")

    monkeypatch.setattr("avo.cli.request_variation_decision", fake_request_variation_decision)

    exit_code = _evolve_loop(
        SimpleNamespace(
            lineage=tmp_path / "lineage",
            knowledge=knowledge,
            cwd=tmp_path,
            timeout_s=10,
            env_file=None,
            model="claude",
            max_steps=1,
            loop_json=tmp_path / "loop.json",
            attempts_dir=tmp_path / "attempts",
            attempt_limit=5,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    saved_payload = json.loads((tmp_path / "loop.json").read_text(encoding="utf-8"))

    assert exit_code == 2
    assert payload["accepted"] is False
    assert payload["completed_steps"] == 1
    assert payload["steps"][0]["attempt"]["command_result"]["ok"] is False
    assert "agent planning failed validation" in payload["steps"][0]["attempt"]["command_result"][
        "stderr_tail"
    ]
    assert saved_payload == payload
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 1


def test_evolve_loop_stops_after_planner_provider_error(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    knowledge = tmp_path / "knowledge.md"
    knowledge.write_text("Ampere only.", encoding="utf-8")
    calls = 0

    def fake_request_variation_decision(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("Anthropic BadRequestError credit balance too low")

    monkeypatch.setattr("avo.cli.request_variation_decision", fake_request_variation_decision)

    exit_code = _evolve_loop(
        SimpleNamespace(
            lineage=tmp_path / "lineage",
            knowledge=knowledge,
            cwd=tmp_path,
            timeout_s=10,
            env_file=None,
            model="claude",
            max_steps=5,
            loop_json=tmp_path / "loop.json",
            attempts_dir=tmp_path / "attempts",
            attempt_limit=5,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    saved_payload = json.loads((tmp_path / "loop.json").read_text(encoding="utf-8"))

    assert exit_code == 2
    assert calls == 1
    assert payload["accepted"] is False
    assert payload["completed_steps"] == 1
    assert payload["max_steps"] == 5
    assert payload["stopped_reason"] == "planner_provider_error"
    assert "BadRequestError" in payload["steps"][0]["attempt"]["command_result"]["stderr_tail"]
    assert saved_payload == payload
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 1


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
