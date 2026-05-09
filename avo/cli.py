from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
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
    apply_candidate_patch,
    cleanup_rejected_candidate_patch,
    finalize_attempt,
    load_promoted_preflight_classes,
    planning_failure_step,
    run_decision_command,
    summarize_attempt_history,
    update_promoted_preflight_tracks,
    validate_decision_against_attempt_history,
    write_attempt,
    write_step,
    write_step_record,
)
from .isolation import module_worker_args, print_result, run_json_worker
from .knowledge import build_knowledge_context
from .lineage import commit_score, init_lineage_repo, lineage_score_summary, seed_baseline


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
    if args.command == "baseline-env":
        return _baseline_env(args)
    if args.command == "knowledge-search":
        return _knowledge_search(args)
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
    lineage_summary, attempt_history, repo_context, knowledge = _planning_context(args)
    decision = request_variation_decision(
        lineage_summary=lineage_summary,
        knowledge=knowledge,
        attempt_history=attempt_history,
        repo_context=repo_context,
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
    step = finalize_attempt(args.lineage, attempt, source_root=args.cwd)
    return cleanup_rejected_candidate_patch(step, cwd=args.cwd)


def _planning_context(args: argparse.Namespace) -> tuple[str, str, str, str]:
    lineage_summary = _lineage_summary(args.lineage)
    attempt_history = summarize_attempt_history(args.attempts_dir, limit=args.attempt_limit)
    repo_context = build_repo_context(args.cwd)
    knowledge = build_knowledge_context(
        args.knowledge,
        query=_knowledge_query(
            lineage_summary=lineage_summary,
            attempt_history=attempt_history,
            repo_context=repo_context,
        ),
    )
    return lineage_summary, attempt_history, repo_context, knowledge


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
