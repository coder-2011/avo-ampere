from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sysconfig
from pathlib import Path

from .agent import (
    DEFAULT_AGENT_MODEL,
    VariationDecision,
    build_repo_context,
    load_env_file,
    request_variation_decision,
)
from .benchmark import score_backend, sleep_score
from .compile import compile_cuda_source
from .config import AMPERE_A6000, cases_from_cli
from .evolve import (
    DEFAULT_ATTEMPT_HISTORY_LIMIT,
    EvolutionStep,
    apply_candidate_patch,
    cleanup_rejected_candidate_patch,
    finalize_attempt,
    run_decision_command,
    summarize_attempt_history,
    write_attempt,
    write_step,
    write_step_record,
)
from .isolation import module_worker_args, print_result, run_json_worker
from .lineage import commit_score, init_lineage_repo, seed_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("env")
    env_parser.add_argument("--env-file", type=Path, default=None)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--source", type=Path, required=True)
    compile_parser.add_argument("--out-dir", type=Path, required=True)
    compile_parser.add_argument("--timeout-s", type=int, default=120)

    score_parser = subparsers.add_parser("score")
    add_score_args(score_parser)
    score_parser.add_argument("--timeout-s", type=int, default=900)

    worker_parser = subparsers.add_parser("worker-score")
    add_score_args(worker_parser)

    sleep_parser = subparsers.add_parser("worker-sleep")
    sleep_parser.add_argument("--seconds", type=float, required=True)

    init_parser = subparsers.add_parser("init-lineage")
    init_parser.add_argument("path", type=Path)

    baseline_parser = subparsers.add_parser("seed-baseline")
    baseline_parser.add_argument("path", type=Path)
    add_score_args(baseline_parser)
    baseline_parser.add_argument("--message", default="chore: seed baseline")
    baseline_parser.add_argument("--timeout-s", type=int, default=900)
    baseline_parser.add_argument("--force", action="store_true")
    baseline_parser.set_defaults(backend="flash-attn")

    commit_parser = subparsers.add_parser("commit-score")
    commit_parser.add_argument("lineage", type=Path)
    commit_parser.add_argument("score_json", type=Path)
    commit_parser.add_argument("--message", default="evolve: accept candidate")

    agent_parser = subparsers.add_parser("agent-plan")
    agent_parser.add_argument("--lineage", type=Path, required=True)
    agent_parser.add_argument("--knowledge", type=Path, required=True)
    agent_parser.add_argument("--cwd", type=Path, default=Path.cwd())
    agent_parser.add_argument("--env-file", type=Path, default=None)
    agent_parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    add_attempt_history_args(agent_parser)

    run_decision_parser = subparsers.add_parser("run-decision")
    run_decision_parser.add_argument("decision_json", type=Path)
    run_decision_parser.add_argument("--cwd", type=Path, default=Path.cwd())
    run_decision_parser.add_argument("--timeout-s", type=int, default=900)
    run_decision_parser.add_argument("--attempt-json", type=Path, default=None)

    apply_patch_parser = subparsers.add_parser("apply-patch")
    apply_patch_parser.add_argument("patch", type=Path)
    apply_patch_parser.add_argument("--cwd", type=Path, default=Path.cwd())
    apply_patch_parser.add_argument("--dry-run", action="store_true")

    evolve_parser = subparsers.add_parser("evolve-once")
    evolve_parser.add_argument("--lineage", type=Path, required=True)
    evolve_parser.add_argument("--knowledge", type=Path, required=True)
    evolve_parser.add_argument("--cwd", type=Path, default=Path.cwd())
    evolve_parser.add_argument("--timeout-s", type=int, default=900)
    evolve_parser.add_argument("--step-json", type=Path, default=None)
    evolve_parser.add_argument("--env-file", type=Path, default=None)
    evolve_parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    add_attempt_history_args(evolve_parser)

    loop_parser = subparsers.add_parser("evolve-loop")
    loop_parser.add_argument("--lineage", type=Path, required=True)
    loop_parser.add_argument("--knowledge", type=Path, required=True)
    loop_parser.add_argument("--cwd", type=Path, default=Path.cwd())
    loop_parser.add_argument("--timeout-s", type=int, default=900)
    loop_parser.add_argument("--env-file", type=Path, default=None)
    loop_parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    loop_parser.add_argument("--max-steps", type=int, default=3)
    loop_parser.add_argument("--loop-json", type=Path, default=None)
    add_attempt_history_args(loop_parser)

    args = parser.parse_args(argv)

    if args.command == "env":
        return _env(args)
    if args.command == "compile":
        return _compile(args)
    if args.command == "score":
        return _score(args)
    if args.command == "worker-score":
        return _worker_score(args)
    if args.command == "worker-sleep":
        print_result(sleep_score(args.seconds))
        return 0
    if args.command == "init-lineage":
        init_lineage_repo(args.path)
        return 0
    if args.command == "seed-baseline":
        return _seed_baseline(args)
    if args.command == "commit-score":
        payload = json.loads(args.score_json.read_text(encoding="utf-8"))
        decision = commit_score(args.lineage, payload, args.message)
        print(json.dumps(decision.as_dict(), indent=2, sort_keys=True))
        return 0 if decision.accepted else 2
    if args.command == "agent-plan":
        return _agent_plan(args)
    if args.command == "run-decision":
        return _run_decision(args)
    if args.command == "apply-patch":
        return _apply_patch(args)
    if args.command == "evolve-once":
        return _evolve_once(args)
    if args.command == "evolve-loop":
        return _evolve_loop(args)
    raise AssertionError(args.command)


def add_score_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=["torch-sdpa", "flash-attn", "candidate"],
        required=True,
    )
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--seq-lens", default="4096,8192,16384,32768")
    parser.add_argument("--causal", choices=["true", "false", "both"], default="both")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--total-tokens", type=int, default=32768)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1)


def add_attempt_history_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--attempts-dir", type=Path, default=None)
    parser.add_argument("--attempt-limit", type=int, default=DEFAULT_ATTEMPT_HISTORY_LIMIT)


def _env(args: argparse.Namespace) -> int:
    torch_cuda: str | None = None
    payload = {
        "target": {
            "name": AMPERE_A6000.name,
            "compute_capability": AMPERE_A6000.compute_capability,
            "compute": AMPERE_A6000.compute,
            "sm": AMPERE_A6000.sm,
            "nvcc_gencode": AMPERE_A6000.nvcc_gencode,
        },
        "agent": _agent_status(args.env_file),
    }
    try:
        import torch

        payload["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
        torch_cuda = torch.version.cuda
        if torch.cuda.is_available():
            payload["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "compute_capability": torch.cuda.get_device_capability(0),
            }
    except Exception as exc:
        payload["torch_error"] = f"{type(exc).__name__}: {exc}"
    payload["baseline_build"] = _baseline_build_status(
        os.environ.copy(),
        torch_cuda=torch_cuda,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _agent_status(env_file: Path | None) -> dict[str, object]:
    env_file_loaded = False
    if env_file is not None:
        env_file_loaded = env_file.exists()
        load_env_file(env_file)

    payload: dict[str, object] = {
        "anthropic_api_key_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "env_file": str(env_file) if env_file is not None else None,
        "env_file_loaded": env_file_loaded,
    }
    try:
        import anthropic
    except ModuleNotFoundError as exc:
        payload["anthropic_installed"] = False
        payload["anthropic_error"] = f"{type(exc).__name__}: {exc}"
    else:
        payload["anthropic_installed"] = True
        payload["anthropic_version"] = getattr(anthropic, "__version__", "unknown")
    return payload


def _compile(args: argparse.Namespace) -> int:
    result = compile_cuda_source(
        args.source,
        args.out_dir,
        timeout_s=args.timeout_s,
        env=os.environ.copy(),
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 2


def _score(args: argparse.Namespace) -> int:
    worker_args = module_worker_args(
        "worker-score",
        "--backend",
        args.backend,
        "--seq-lens",
        args.seq_lens,
        "--causal",
        args.causal,
        "--head-dim",
        str(args.head_dim),
        "--num-heads",
        str(args.num_heads),
        "--total-tokens",
        str(args.total_tokens),
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--trials",
        str(args.trials),
    )
    if args.candidate:
        worker_args.extend(["--candidate", str(args.candidate)])
    result = run_json_worker(
        worker_args,
        timeout_s=args.timeout_s,
        cwd=Path.cwd(),
        env=os.environ.copy(),
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 2


def _seed_baseline(args: argparse.Namespace) -> int:
    worker_args = module_worker_args(
        "worker-score",
        "--backend",
        args.backend,
        "--seq-lens",
        args.seq_lens,
        "--causal",
        args.causal,
        "--head-dim",
        str(args.head_dim),
        "--num-heads",
        str(args.num_heads),
        "--total-tokens",
        str(args.total_tokens),
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--trials",
        str(args.trials),
    )
    if args.candidate:
        worker_args.extend(["--candidate", str(args.candidate)])
    baseline_env = _baseline_build_env(os.environ.copy())
    if args.backend == "flash-attn" and importlib.util.find_spec("flash_attn") is None:
        build_status = _baseline_build_status(baseline_env)
        if not build_status["ok_for_torch_extension_build"]:
            raise RuntimeError(
                "flash-attn is not installed and the baseline source-build "
                f"environment is not ready: {build_status['warning']}"
            )
    result = run_json_worker(
        worker_args,
        timeout_s=args.timeout_s,
        cwd=Path.cwd(),
        env=baseline_env,
    )
    if not result.ok or result.payload is None:
        raise RuntimeError(f"baseline score failed: {result.stderr_tail}")
    seed = seed_baseline(args.path, result.payload, message=args.message, force=args.force)
    print(json.dumps(seed, indent=2, sort_keys=True))
    return 0


def _worker_score(args: argparse.Namespace) -> int:
    cases = cases_from_cli(
        args.seq_lens,
        args.causal,
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        total_tokens=args.total_tokens,
        dtype=args.dtype,
    )
    payload = score_backend(
        args.backend,
        cases,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
        candidate=args.candidate,
    )
    print_result(payload)
    return 0


def _agent_plan(args: argparse.Namespace) -> int:
    if args.env_file:
        load_env_file(args.env_file)
    knowledge = args.knowledge.read_text(encoding="utf-8")
    lineage_summary = _lineage_summary(args.lineage)
    decision = request_variation_decision(
        lineage_summary=lineage_summary,
        knowledge=knowledge,
        attempt_history=summarize_attempt_history(args.attempts_dir, limit=args.attempt_limit),
        repo_context=build_repo_context(args.cwd),
        model=args.model,
    )
    print(json.dumps(decision.as_dict(), indent=2, sort_keys=True))
    return 0


def _run_decision(args: argparse.Namespace) -> int:
    payload = json.loads(args.decision_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("decision JSON must be an object")
    decision = VariationDecision.from_mapping(payload)
    attempt = run_decision_command(
        decision,
        cwd=args.cwd,
        timeout_s=args.timeout_s,
        env=os.environ.copy(),
    )
    if args.attempt_json:
        write_attempt(args.attempt_json, attempt)
    print(json.dumps(attempt.as_dict(), indent=2, sort_keys=True))
    return 0 if attempt.command_result.ok else 2


def _apply_patch(args: argparse.Namespace) -> int:
    result = apply_candidate_patch(
        args.patch.read_text(encoding="utf-8"),
        cwd=args.cwd,
        dry_run=args.dry_run,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 2


def _evolve_once(args: argparse.Namespace) -> int:
    if args.env_file:
        load_env_file(args.env_file)
    knowledge = args.knowledge.read_text(encoding="utf-8")
    step = _run_evolve_step(args, knowledge)
    if args.step_json:
        write_step(args.step_json, step)
    if args.attempts_dir:
        write_step_record(args.attempts_dir, step)
    print(json.dumps(step.as_dict(), indent=2, sort_keys=True))
    return _step_exit_code(step)


def _evolve_loop(args: argparse.Namespace) -> int:
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.attempts_dir is None:
        raise ValueError("evolve-loop requires --attempts-dir for cross-step memory")
    if args.env_file:
        load_env_file(args.env_file)
    knowledge = args.knowledge.read_text(encoding="utf-8")

    steps: list[EvolutionStep] = []
    stopped_reason = "max_steps"
    for _ in range(args.max_steps):
        step = _run_evolve_step(args, knowledge)
        steps.append(step)
        write_step_record(args.attempts_dir, step)
        if step.accepted:
            stopped_reason = "accepted"
            break
        if step.patch_cleanup_result is not None and not step.patch_cleanup_result.ok:
            stopped_reason = "cleanup_failed"
            break

    payload = {
        "accepted": any(step.accepted for step in steps),
        "completed_steps": len(steps),
        "max_steps": args.max_steps,
        "stopped_reason": stopped_reason,
        "steps": [step.as_dict() for step in steps],
    }
    if args.loop_json:
        args.loop_json.parent.mkdir(parents=True, exist_ok=True)
        args.loop_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["accepted"] else 2


def _run_evolve_step(args: argparse.Namespace, knowledge: str) -> EvolutionStep:
    decision = request_variation_decision(
        lineage_summary=_lineage_summary(args.lineage),
        knowledge=knowledge,
        attempt_history=summarize_attempt_history(args.attempts_dir, limit=args.attempt_limit),
        repo_context=build_repo_context(args.cwd),
        model=args.model,
    )
    attempt = run_decision_command(
        decision,
        cwd=args.cwd,
        timeout_s=args.timeout_s,
        env=os.environ.copy(),
    )
    step = finalize_attempt(args.lineage, attempt, source_root=args.cwd)
    return cleanup_rejected_candidate_patch(step, cwd=args.cwd)


def _step_exit_code(step: EvolutionStep) -> int:
    if step.patch_cleanup_result is not None and not step.patch_cleanup_result.ok:
        return 2
    if not step.attempt.command_result.ok:
        return 2
    if step.gate_decision is not None and not step.gate_decision.accepted:
        return 2
    return 0


def _lineage_summary(path: Path) -> str:
    latest = path / "scores" / "latest.json"
    if not latest.exists():
        return "No accepted candidates yet."
    return latest.read_text(encoding="utf-8")


def _baseline_build_env(env: dict[str, str]) -> dict[str, str]:
    # Build FlashAttention-2 for Ampere family members only on this system.
    # The upstream setup script does not expose a 86-specific token; `80` maps to
    # the Ampere family path used by the project for sm80+ targets.
    env["FLASH_ATTN_CUDA_ARCHS"] = "80"
    # This A6000 pod has limited host RAM. Keep CUDA compilation conservative by
    # default while still allowing an explicit caller override.
    env.setdefault("MAX_JOBS", "1")
    env.setdefault("NVCC_THREADS", "1")
    python_cuda_home = _compatible_python_cuda_home(env)
    if python_cuda_home is not None and not _cuda_env_is_build_compatible(env):
        env["CUDA_HOME"] = python_cuda_home
    return env


def _baseline_build_status(
    env: dict[str, str],
    *,
    torch_cuda: str | None = None,
) -> dict[str, object]:
    baseline_env = _baseline_build_env(dict(env))
    if torch_cuda is None:
        try:
            import torch
        except Exception:
            torch_cuda = None
        else:
            torch_cuda = torch.version.cuda
    nvcc_path = _nvcc_path_from_env(baseline_env)
    nvcc_cuda, nvcc_error = _nvcc_cuda_version(nvcc_path, baseline_env)
    compatibility, warning = _cuda_build_compatibility(torch_cuda, nvcc_cuda)
    return {
        "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None,
        "settings": {
            "FLASH_ATTN_CUDA_ARCHS": baseline_env["FLASH_ATTN_CUDA_ARCHS"],
            "MAX_JOBS": baseline_env["MAX_JOBS"],
            "NVCC_THREADS": baseline_env["NVCC_THREADS"],
            "CUDA_HOME": baseline_env.get("CUDA_HOME"),
            "CUDA_PATH": baseline_env.get("CUDA_PATH"),
        },
        "torch_cuda": torch_cuda,
        "nvcc_path": nvcc_path,
        "nvcc_cuda": nvcc_cuda,
        "nvcc_error": nvcc_error,
        "compatibility": compatibility,
        "ok_for_torch_extension_build": compatibility in {"exact", "minor_mismatch"},
        "warning": warning,
    }


def _nvcc_path_from_env(env: dict[str, str]) -> str | None:
    cuda_home = env.get("CUDA_HOME") or env.get("CUDA_PATH")
    if cuda_home:
        executable = "nvcc.exe" if os.name == "nt" else "nvcc"
        return str(Path(cuda_home) / "bin" / executable)
    return shutil.which("nvcc", path=env.get("PATH"))


def _python_cuda_home() -> str | None:
    try:
        purelib = Path(sysconfig.get_paths()["purelib"])
    except Exception:
        return None
    candidates = []
    for nvcc in purelib.glob("nvidia/cu*/bin/nvcc"):
        cuda_home = nvcc.parent.parent
        if (cuda_home / "include" / "cuda.h").exists():
            candidates.append(cuda_home)
    if len(candidates) != 1:
        return None
    return str(candidates[0])


def _compatible_python_cuda_home(env: dict[str, str]) -> str | None:
    cuda_home = _python_cuda_home()
    if cuda_home is None:
        return None
    nvcc_path = str(Path(cuda_home) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc"))
    nvcc_cuda, _ = _nvcc_cuda_version(nvcc_path, env)
    compatibility, _ = _cuda_build_compatibility(_torch_cuda_version(), nvcc_cuda)
    if compatibility not in {"exact", "minor_mismatch"}:
        return None
    return cuda_home


def _cuda_env_is_build_compatible(env: dict[str, str]) -> bool:
    nvcc_path = _nvcc_path_from_env(env)
    nvcc_cuda, _ = _nvcc_cuda_version(nvcc_path, env)
    compatibility, _ = _cuda_build_compatibility(_torch_cuda_version(), nvcc_cuda)
    return compatibility in {"exact", "minor_mismatch"}


def _torch_cuda_version() -> str | None:
    try:
        import torch
    except Exception:
        return None
    return torch.version.cuda


def _nvcc_cuda_version(
    nvcc_path: str | None,
    env: dict[str, str],
) -> tuple[str | None, str | None]:
    if nvcc_path is None:
        return None, "nvcc was not found in CUDA_HOME, CUDA_PATH, or PATH"
    try:
        completed = subprocess.run(
            [nvcc_path, "--version"],
            check=True,
            capture_output=True,
            env=env,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return None, f"nvcc was not found at {nvcc_path}"
    except subprocess.TimeoutExpired:
        return None, f"{nvcc_path} --version timed out"
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or exc.stderr or "").strip()
        suffix = f": {output}" if output else ""
        return None, f"{nvcc_path} --version failed{suffix}"
    output = f"{completed.stdout}\n{completed.stderr}"
    version = _parse_nvcc_release(output)
    if version is None:
        return None, f"could not parse CUDA release from {nvcc_path} --version"
    return version, None


def _parse_nvcc_release(output: str) -> str | None:
    match = re.search(r"release\s+(\d+\.\d+)", output)
    if match is None:
        return None
    return match.group(1)


def _cuda_build_compatibility(
    torch_cuda: str | None,
    nvcc_cuda: str | None,
) -> tuple[str, str | None]:
    if torch_cuda is None:
        return "missing_torch_cuda", "torch.version.cuda is unavailable"
    if nvcc_cuda is None:
        return "missing_nvcc", "nvcc CUDA version is unavailable"
    torch_version = _cuda_major_minor(torch_cuda)
    nvcc_version = _cuda_major_minor(nvcc_cuda)
    if torch_version is None:
        return "unparseable_torch_cuda", f"could not parse torch CUDA version {torch_cuda!r}"
    if nvcc_version is None:
        return "unparseable_nvcc_cuda", f"could not parse nvcc CUDA version {nvcc_cuda!r}"
    if torch_version == nvcc_version:
        return "exact", None
    if torch_version[0] == nvcc_version[0]:
        return (
            "minor_mismatch",
            "PyTorch extension builds may warn: "
            f"nvcc reports CUDA {nvcc_cuda} but torch was compiled with CUDA {torch_cuda}",
        )
    return (
        "major_mismatch",
        "PyTorch extension builds will fail: "
        f"nvcc reports CUDA {nvcc_cuda} but torch was compiled with CUDA {torch_cuda}",
    )


def _cuda_major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"^\s*(\d+)\.(\d+)", version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))
