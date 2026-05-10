from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

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
from .cuda_env import (
    baseline_build_env,
    cuda_build_compatibility,
    nvcc_cuda_version,
    nvcc_path_from_env,
    prepare_torch_extension_env,
)
from .evolve import (
    DEFAULT_ATTEMPT_HISTORY_LIMIT,
    EvolutionStep,
    VariationAttempt,
    apply_candidate_patch,
    attempt_has_repairable_compile_failure,
    attempt_has_repairable_transform_materialization_failure,
    cleanup_rejected_candidate_patch,
    compile_failure_class_for_attempt,
    finalize_attempt,
    load_promoted_preflight_classes,
    pending_compile_only_transform,
    planning_failure_step,
    run_decision_command,
    summarize_attempt_history,
    update_promoted_preflight_tracks,
    validate_decision_against_attempt_history,
    write_attempt,
    write_step,
    write_step_record,
)
from .isolation import RESULT_PREFIX, module_worker_args, print_result, run_json_worker
from .knowledge import build_knowledge_context
from .lineage import commit_score, init_lineage_repo, lineage_score_summary, seed_baseline

GENERAL_CUDA_PRACTICE_QUERY = (
    "CUDA Kernel Design Practice Basic Mental Model Decomposing Work Indexing "
    "Data Layout Memory Movement Synchronization Communication Tiling Pattern "
    "Tensor Cores Launch Configuration Profiling Optimization Workflow Semantic "
    "Transform Guidance"
)
GENERAL_CUDA_PRACTICE_MAX_CHARS = 8_000
GENERAL_CUDA_PRACTICE_MAX_CHUNKS = 4
DEFAULT_COMPILE_REPAIR_ATTEMPTS = 2
PROFILE_TIMEOUT_CAP_S = 120
THUNDER_CUDA_SHIM = Path("/etc/thunder/libthunder.so")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("env")
    env_parser.add_argument("--env-file", type=Path, default=None)

    baseline_env_parser = subparsers.add_parser("baseline-env")
    baseline_env_parser.add_argument("--format", choices=["shell", "json"], default="shell")

    knowledge_parser = subparsers.add_parser("knowledge-search")
    knowledge_parser.add_argument("path", type=Path)
    knowledge_parser.add_argument("--query", required=True)
    knowledge_parser.add_argument("--max-chunks", type=int, default=10)
    knowledge_parser.add_argument("--max-chars", type=int, default=24_000)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--source", type=Path, required=True)
    compile_parser.add_argument("--out-dir", type=Path, required=True)
    compile_parser.add_argument("--timeout-s", type=int, default=120)

    score_parser = subparsers.add_parser("score")
    add_score_args(score_parser)
    score_parser.add_argument("--timeout-s", type=int, default=900)

    profile_parser = subparsers.add_parser("profile")
    add_score_args(profile_parser)
    profile_parser.add_argument("--timeout-s", type=int, default=900)
    profile_parser.add_argument("--ncu-set", default="basic")
    profile_parser.add_argument("--section", action="append", default=[])
    profile_parser.add_argument("--kernel-name", default="regex:.*attention.*")
    profile_parser.add_argument("--launch-count", type=int, default=1)
    profile_parser.add_argument("--launch-skip", type=int, default=0)
    profile_parser.add_argument("--page", choices=["details", "raw", "source"], default="raw")

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
    evolve_parser.add_argument(
        "--compile-repair-attempts",
        type=int,
        default=DEFAULT_COMPILE_REPAIR_ATTEMPTS,
    )
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
    loop_parser.add_argument(
        "--compile-repair-attempts",
        type=int,
        default=DEFAULT_COMPILE_REPAIR_ATTEMPTS,
    )
    add_attempt_history_args(loop_parser)

    args = parser.parse_args(argv)

    if args.command == "env":
        return _env(args)
    if args.command == "baseline-env":
        return _baseline_env(args)
    if args.command == "knowledge-search":
        return _knowledge_search(args)
    if args.command == "compile":
        return _compile(args)
    if args.command == "score":
        return _score(args)
    if args.command == "profile":
        return _profile(args)
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


def _knowledge_search(args: argparse.Namespace) -> int:
    print(
        build_knowledge_context(
            args.path,
            query=args.query,
            max_chunks=args.max_chunks,
            max_chars=args.max_chars,
        )
    )
    return 0


BASELINE_ENV_EXPORT_KEYS = (
    "FLASH_ATTN_CUDA_ARCHS",
    "MAX_JOBS",
    "NVCC_THREADS",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDACXX",
    "PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
)


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


def _baseline_env(args: argparse.Namespace) -> int:
    baseline_env = _baseline_build_env(os.environ.copy())
    payload = {
        key: baseline_env[key]
        for key in BASELINE_ENV_EXPORT_KEYS
        if baseline_env.get(key)
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    for key, value in payload.items():
        print(f"export {key}={shlex.quote(value)}")
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
    prepare_torch_extension_env(os.environ, max_jobs="1")
    result = compile_cuda_source(
        args.source,
        args.out_dir,
        timeout_s=args.timeout_s,
        env=os.environ.copy(),
        include_dirs=_torch_cuda_include_dirs(),
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 2


def _torch_cuda_include_dirs() -> list[Path]:
    include_dirs: list[Path] = []
    try:
        from torch.utils.cpp_extension import include_paths
    except Exception:
        return include_dirs
    try:
        torch_include_paths = include_paths(cuda=True)
    except TypeError:
        torch_include_paths = include_paths(device_type="cuda")
    include_dirs.extend(Path(path) for path in torch_include_paths)
    python_include = sysconfig.get_paths().get("include")
    if python_include:
        include_dirs.append(Path(python_include))
    return include_dirs


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
        env=_torch_extension_worker_env(os.environ.copy()),
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 2


def _profile(args: argparse.Namespace) -> int:
    if args.backend != "candidate":
        raise ValueError("profile currently supports --backend candidate only")
    if args.candidate is None:
        raise ValueError("profile requires --candidate")
    if args.timeout_s <= 0:
        raise ValueError("profile requires --timeout-s to be positive")
    if args.launch_count <= 0:
        raise ValueError("profile requires --launch-count to be positive")
    if args.launch_skip < 0:
        raise ValueError("profile requires --launch-skip to be non-negative")

    env = _torch_extension_worker_env(os.environ.copy())
    profile_timeout_s = min(args.timeout_s, PROFILE_TIMEOUT_CAP_S)
    ncu_path = shutil.which("ncu", path=env.get("PATH")) or shutil.which("ncu")
    if ncu_path is None:
        payload = _profile_payload(
            command=[],
            returncode=None,
            timed_out=False,
            stdout="",
            stderr="",
            score_payload=None,
            profiler_error="ncu_not_found",
            settings=_profile_settings(args, timeout_s=profile_timeout_s),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    unavailable_error = _profile_unavailable_error()
    if unavailable_error is not None:
        payload = _profile_payload(
            command=[],
            returncode=None,
            timed_out=False,
            stdout="",
            stderr=(
                "Nsight Compute profiling is unavailable in this runtime before launch: "
                f"{unavailable_error}"
            ),
            score_payload=None,
            profiler_error=unavailable_error,
            settings=_profile_settings(args, timeout_s=profile_timeout_s),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

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
        "--candidate",
        str(args.candidate),
    )
    ncu_args = [
        ncu_path,
        "--target-processes",
        "all",
        "--set",
        args.ncu_set,
        "--kernel-name",
        args.kernel_name,
        "--launch-count",
        str(args.launch_count),
        "--launch-skip",
        str(args.launch_skip),
        "--csv",
        "--page",
        args.page,
    ]
    for section in args.section:
        ncu_args.extend(["--section", section])
    command = [*ncu_args, *worker_args]

    try:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=profile_timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        payload = _profile_payload(
            command=command,
            returncode=None,
            timed_out=True,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            score_payload=_extract_worker_payload(exc.stdout or ""),
            profiler_error="timeout",
            settings=_profile_settings(args, timeout_s=profile_timeout_s),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    score_payload = _extract_worker_payload(completed.stdout)
    profiler_error = _classify_profile_error(
        completed.stdout,
        completed.stderr,
        returncode=completed.returncode,
    )
    payload = _profile_payload(
        command=command,
        returncode=completed.returncode,
        timed_out=False,
        stdout=completed.stdout,
        stderr=completed.stderr,
        score_payload=score_payload,
        profiler_error=profiler_error,
        settings=_profile_settings(args, timeout_s=profile_timeout_s),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


def _profile_unavailable_error() -> str | None:
    if THUNDER_CUDA_SHIM.exists():
        return "profiler_unsupported_runtime"
    return None


def _profile_settings(args: argparse.Namespace, *, timeout_s: int) -> dict[str, object]:
    return {
        "ncu_set": args.ncu_set,
        "sections": list(args.section),
        "kernel_name": args.kernel_name,
        "launch_count": args.launch_count,
        "launch_skip": args.launch_skip,
        "page": args.page,
        "timeout_s": timeout_s,
        "requested_timeout_s": args.timeout_s,
    }


def _profile_payload(
    *,
    command: list[str],
    returncode: int | None,
    timed_out: bool,
    stdout: str,
    stderr: str,
    score_payload: dict[str, Any] | None,
    profiler_error: str | None,
    settings: dict[str, object],
) -> dict[str, object]:
    profiled = profiler_error is None and "==PROF==" in stdout
    ok = returncode == 0 and not timed_out and profiled and score_payload is not None
    return {
        "ok": ok,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "profiler": {
            "tool": "ncu",
            "profiled": profiled,
            "error": profiler_error,
            **settings,
        },
        "score_payload": score_payload,
        "stdout_tail": _short_text(stdout, 4000),
        "stderr_tail": _short_text(stderr, 4000),
    }


def _extract_worker_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if not line.startswith(RESULT_PREFIX):
            continue
        try:
            parsed = json.loads(line.removeprefix(RESULT_PREFIX))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _classify_profile_error(stdout: str, stderr: str, *, returncode: int | None) -> str | None:
    text = f"{stdout}\n{stderr}".lower()
    if "no kernels were profiled" in text:
        return "no_kernels_profiled"
    if "err_nvgpuctrperm" in text or (
        "permission" in text and "performance counter" in text
    ):
        return "profiler_permission"
    if (
        "unimplemented cuda export table function" in text
        or "not yet supported on thunder" in text
        or "thunder_aborted" in text
    ):
        return "profiler_unsupported_runtime"
    if returncode not in {0, None}:
        return "profiler_failed"
    if RESULT_PREFIX not in stdout:
        return "target_payload_missing"
    if "==prof==" not in text:
        return "no_profiler_output"
    return None


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
    _prepend_env_path(baseline_env, str(Path(sys.executable).parent))
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


def _torch_extension_worker_env(env: dict[str, str]) -> dict[str, str]:
    updated = prepare_torch_extension_env(env, max_jobs="1")
    _prepend_env_path(updated, str(Path(sys.executable).parent))
    return updated


def _prepend_env_path(env: dict[str, str], path: str) -> None:
    existing = [part for part in env.get("PATH", "").split(os.pathsep) if part]
    env["PATH"] = os.pathsep.join([path, *(part for part in existing if part != path)])


def _agent_plan(args: argparse.Namespace) -> int:
    if args.env_file:
        load_env_file(args.env_file)
    lineage_summary, attempt_history, repo_context, knowledge = _planning_context(args)
    decision = request_variation_decision(
        lineage_summary=lineage_summary,
        knowledge=knowledge,
        attempt_history=attempt_history,
        repo_context=repo_context,
        model=args.model,
        normalize_payload=_pending_transform_payload_normalizer(args.attempts_dir),
    )
    validate_decision_against_attempt_history(decision, args.attempts_dir)
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
    step = _run_evolve_step(args)
    if args.step_json:
        write_step(args.step_json, step)
    if args.attempts_dir:
        write_step_record(args.attempts_dir, step)
        update_promoted_preflight_tracks(args.attempts_dir)
    print(json.dumps(step.as_dict(), indent=2, sort_keys=True))
    return _step_exit_code(step)


def _evolve_loop(args: argparse.Namespace) -> int:
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.attempts_dir is None:
        raise ValueError("evolve-loop requires --attempts-dir for cross-step memory")
    if args.env_file:
        load_env_file(args.env_file)

    steps: list[EvolutionStep] = []
    stopped_reason = "max_steps"
    update_promoted_preflight_tracks(args.attempts_dir)
    for _ in range(args.max_steps):
        step = _run_evolve_step(args)
        steps.append(step)
        write_step_record(args.attempts_dir, step)
        update_promoted_preflight_tracks(args.attempts_dir)
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


def _run_evolve_step(args: argparse.Namespace) -> EvolutionStep:
    try:
        lineage_summary, attempt_history, repo_context, knowledge = _planning_context(args)
        decision = request_variation_decision(
            lineage_summary=lineage_summary,
            knowledge=knowledge,
            attempt_history=attempt_history,
            repo_context=repo_context,
            model=args.model,
            normalize_payload=_pending_transform_payload_normalizer(args.attempts_dir),
        )
        validate_decision_against_attempt_history(decision, args.attempts_dir)
    except ValueError as exc:
        return planning_failure_step(exc)
    attempt = run_decision_command(
        decision,
        cwd=args.cwd,
        timeout_s=args.timeout_s,
        env=os.environ.copy(),
        promoted_preflight_classes=load_promoted_preflight_classes(args.attempts_dir),
    )
    repaired = _run_compile_repair_loop(
        args,
        initial_attempt=attempt,
        lineage_summary=lineage_summary,
        attempt_history=attempt_history,
        repo_context=repo_context,
        knowledge=knowledge,
    )
    if isinstance(repaired, EvolutionStep):
        return repaired
    attempt, repair_attempts, repair_cleanup_results = repaired
    step = finalize_attempt(
        args.lineage,
        attempt,
        source_root=args.cwd,
        repair_attempts=tuple(repair_attempts),
        repair_cleanup_results=tuple(repair_cleanup_results),
    )
    return cleanup_rejected_candidate_patch(step, cwd=args.cwd)


def _run_compile_repair_loop(
    args: argparse.Namespace,
    *,
    initial_attempt: VariationAttempt,
    lineage_summary: str,
    attempt_history: str,
    repo_context: str,
    knowledge: str,
) -> EvolutionStep | tuple[VariationAttempt, list[VariationAttempt], list[Any]]:
    repair_limit = max(
        0,
        int(getattr(args, "compile_repair_attempts", DEFAULT_COMPILE_REPAIR_ATTEMPTS)),
    )
    repair_attempts: list[VariationAttempt] = []
    repair_cleanup_results = []
    current_attempt = initial_attempt

    for repair_index in range(1, repair_limit + 1):
        repair_kind = _repair_kind_for_attempt(current_attempt)
        if repair_kind is None:
            break
        cleanup_step = cleanup_rejected_candidate_patch(
            EvolutionStep(attempt=current_attempt, gate_decision=None),
            cwd=args.cwd,
        )
        if cleanup_step.patch_cleanup_result is not None:
            repair_cleanup_results.append(cleanup_step.patch_cleanup_result)
            if not cleanup_step.patch_cleanup_result.ok:
                return EvolutionStep(
                    attempt=current_attempt,
                    gate_decision=None,
                    patch_cleanup_result=cleanup_step.patch_cleanup_result,
                    repair_attempts=tuple(repair_attempts),
                    repair_cleanup_results=tuple(repair_cleanup_results),
                )
        repair_attempts.append(current_attempt)
        repair_history = _edit_repair_attempt_history(
            attempt_history,
            failed_attempt=current_attempt,
            cleanup_result=cleanup_step.patch_cleanup_result,
            repair_index=repair_index,
            repair_kind=repair_kind,
        )
        try:
            repair_decision = request_variation_decision(
                lineage_summary=lineage_summary,
                knowledge=knowledge,
                attempt_history=repair_history,
                repo_context=repo_context,
                model=args.model,
                normalize_payload=_pending_transform_payload_normalizer(args.attempts_dir),
            )
            _validate_edit_repair_decision(repair_decision, current_attempt, repair_kind)
            validate_decision_against_attempt_history(repair_decision, args.attempts_dir)
        except ValueError as exc:
            return planning_failure_step(
                exc,
                repair_attempts=tuple(repair_attempts),
                repair_cleanup_results=tuple(repair_cleanup_results),
            )
        current_attempt = run_decision_command(
            repair_decision,
            cwd=args.cwd,
            timeout_s=args.timeout_s,
            env=os.environ.copy(),
            promoted_preflight_classes=load_promoted_preflight_classes(args.attempts_dir),
        )

    return current_attempt, repair_attempts, repair_cleanup_results


def _repair_kind_for_attempt(attempt: VariationAttempt) -> str | None:
    if attempt_has_repairable_compile_failure(attempt):
        return "compile"
    if attempt_has_repairable_transform_materialization_failure(attempt):
        return "structured-transform-materialization"
    return None


def _edit_repair_attempt_history(
    attempt_history: str,
    *,
    failed_attempt: VariationAttempt,
    cleanup_result: Any,
    repair_index: int,
    repair_kind: str,
) -> str:
    cleanup_status = "not needed"
    if cleanup_result is not None:
        cleanup_status = "ok" if cleanup_result.ok else "failed"
    if repair_kind == "compile":
        request = (
            "Immediate compile-repair request:\n"
            f"- repair_attempt={repair_index}\n"
            f"- failure_class={compile_failure_class_for_attempt(failed_attempt)}\n"
            f"- failed_command={failed_attempt.decision.next_command}\n"
            f"- worktree_cleanup_before_repair={cleanup_status}\n"
            "- The previous candidate edit was applied and the CUDA build failed. "
            "The failed edit has been reverted before this repair request. Return a "
            "revised executable edit against the current source, not a no-edit retry. "
            "Use candidate_transform when repairing CUDA sources, keep candidate_patch "
            "empty in transform mode, and make the smallest coherent semantic repair "
            "that addresses the compiler output.\n"
            f"- failed_edit_payload={_attempt_edit_payload_summary(failed_attempt)}\n"
            f"- compiler_stderr_tail:\n{failed_attempt.command_result.stderr_tail or '<empty>'}\n"
            f"- compiler_stdout_tail:\n{failed_attempt.command_result.stdout_tail or '<empty>'}"
        )
    else:
        patch_result = failed_attempt.patch_result
        materialization_error = ""
        if patch_result is not None:
            materialization_error = patch_result.rejected_reason or patch_result.stderr_tail
        request = (
            "Immediate structured-transform materialization repair request:\n"
            f"- repair_attempt={repair_index}\n"
            f"- failed_command={failed_attempt.decision.next_command}\n"
            f"- worktree_cleanup_before_repair={cleanup_status}\n"
            "- The previous candidate_transform failed before command execution because "
            "one or more anchors/matches did not select exactly one source span. Return "
            "a revised candidate_transform with larger unique surrounding-code snippets, "
            "not a no-edit retry and not prose-only CUDA edits. Keep candidate_patch "
            "empty in transform mode and make the smallest coherent semantic repair "
            "that preserves the candidate's intended invariant.\n"
            f"- failed_edit_payload={_attempt_edit_payload_summary(failed_attempt)}\n"
            f"- materialization_error:\n{materialization_error or '<empty>'}"
        )
    sections = [
        attempt_history.strip(),
        request,
    ]
    return "\n\n".join(section for section in sections if section)


def _validate_edit_repair_decision(
    repair_decision: VariationDecision,
    failed_attempt: VariationAttempt,
    repair_kind: str,
) -> None:
    if repair_decision.candidate_transform is None and not repair_decision.candidate_patch.strip():
        raise ValueError(
            f"{repair_kind} repair decision must include a revised candidate_transform "
            "or candidate_patch; no-edit retries do not repair failed executable edits"
        )
    if (
        repair_kind == "structured-transform-materialization"
        and failed_attempt.decision.candidate_transform is not None
        and repair_decision.candidate_transform is None
    ):
        raise ValueError(
            "structured-transform materialization repair must return a revised "
            "candidate_transform; prose-only retries and raw CUDA patches are not a "
            "valid repair for failed transform materialization"
        )
    if _same_edit_payload(repair_decision, failed_attempt.decision):
        raise ValueError(
            f"{repair_kind} repair decision repeats the failed edit payload unchanged; "
            "revise the structured transform or patch to address the failure"
        )


def _same_edit_payload(left: VariationDecision, right: VariationDecision) -> bool:
    return (
        left.candidate_transform == right.candidate_transform
        and left.candidate_patch.strip() == right.candidate_patch.strip()
    )


def _attempt_edit_payload_summary(attempt: VariationAttempt) -> str:
    if attempt.decision.candidate_transform is not None:
        return json.dumps(attempt.decision.candidate_transform, sort_keys=True)
    patch_text = attempt.materialized_patch or attempt.decision.candidate_patch
    if patch_text.strip():
        return _short_text(patch_text, 3000)
    return "<no edit payload>"


def _short_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _pending_transform_payload_normalizer(attempts_dir: Path | None):
    pending_transform = pending_compile_only_transform(attempts_dir)
    if pending_transform is None:
        return None

    def normalize(payload: dict[str, Any]) -> dict[str, Any]:
        candidate_patch = str(payload.get("candidate_patch") or "")
        if _payload_candidate_patch_has_diff(candidate_patch):
            return payload
        edit_mode = str(payload.get("edit_mode") or "")
        if edit_mode not in {"", "transform", "no_edit"}:
            return payload
        if not _payload_wants_pending_transform_score(payload):
            return payload
        updated = dict(payload)
        updated["edit_mode"] = "transform"
        updated["candidate_patch"] = ""
        updated["candidate_transform"] = pending_transform
        return updated

    return normalize


def _payload_candidate_patch_has_diff(candidate_patch: str) -> bool:
    return any(line.startswith("diff --git ") for line in candidate_patch.splitlines())


def _payload_wants_pending_transform_score(payload: dict[str, Any]) -> bool:
    if _payload_subcommand(payload) == "score":
        return True
    text = " ".join(
        str(payload.get(key) or "")
        for key in ("hypothesis", "candidate_edit", "expected_effect", "risk", "next_command")
    ).lower()
    return "score" in text and "compiled" in text and "transform" in text


def _payload_subcommand(payload: dict[str, Any]) -> str:
    try:
        parts = shlex.split(str(payload.get("next_command") or ""))
    except ValueError:
        return ""
    if len(parts) < 2 or parts[0] != "avo":
        return ""
    return parts[1]


def _planning_context(args: argparse.Namespace) -> tuple[str, str, str, str]:
    lineage_summary = _lineage_summary(args.lineage)
    attempt_history = summarize_attempt_history(args.attempts_dir, limit=args.attempt_limit)
    repo_context = build_repo_context(args.cwd)
    knowledge_query = _knowledge_query(
        lineage_summary=lineage_summary,
        attempt_history=attempt_history,
        repo_context=repo_context,
    )
    knowledge = build_knowledge_context(
        args.knowledge,
        query=knowledge_query,
    )
    knowledge = _with_general_cuda_practice_context(knowledge, args.knowledge)
    return lineage_summary, attempt_history, repo_context, knowledge


def _with_general_cuda_practice_context(knowledge: str, source: Path) -> str:
    if "b/cuda_programming_practice.md" in knowledge:
        return knowledge
    supplemental = build_knowledge_context(
        source,
        query=GENERAL_CUDA_PRACTICE_QUERY,
        max_chunks=GENERAL_CUDA_PRACTICE_MAX_CHUNKS,
        max_chars=GENERAL_CUDA_PRACTICE_MAX_CHARS,
    )
    if (
        "b/cuda_programming_practice.md" not in supplemental
        or "No supported knowledge files were found" in supplemental
    ):
        return knowledge
    return f"{knowledge}\n\nSupplemental broad CUDA practice context:\n{supplemental}"


def _knowledge_query(
    *,
    lineage_summary: str,
    attempt_history: str,
    repo_context: str,
) -> str:
    return "\n\n".join(
        part
        for part in (
            "Ampere sm86 FlashAttention-2 CUDA attention kernel evolution",
            GENERAL_CUDA_PRACTICE_QUERY,
            lineage_summary,
            attempt_history,
            repo_context[:12_000],
        )
        if part.strip()
    )


def _step_exit_code(step: EvolutionStep) -> int:
    if step.patch_cleanup_result is not None and not step.patch_cleanup_result.ok:
        return 2
    if not step.attempt.command_result.ok:
        return 2
    if step.gate_decision is not None and not step.gate_decision.accepted:
        return 2
    return 0


def _lineage_summary(path: Path) -> str:
    return lineage_score_summary(path)


def _baseline_build_env(env: dict[str, str]) -> dict[str, str]:
    return baseline_build_env(env)


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
    nvcc_path = nvcc_path_from_env(baseline_env)
    nvcc_cuda, nvcc_error = nvcc_cuda_version(nvcc_path, baseline_env)
    compatibility, warning = cuda_build_compatibility(torch_cuda, nvcc_cuda)
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
