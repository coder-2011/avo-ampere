from __future__ import annotations

import argparse
import json
import os
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

    evolve_parser = subparsers.add_parser("evolve-once")
    evolve_parser.add_argument("--lineage", type=Path, required=True)
    evolve_parser.add_argument("--knowledge", type=Path, required=True)
    evolve_parser.add_argument("--cwd", type=Path, default=Path.cwd())
    evolve_parser.add_argument("--timeout-s", type=int, default=900)
    evolve_parser.add_argument("--step-json", type=Path, default=None)
    evolve_parser.add_argument("--env-file", type=Path, default=None)
    evolve_parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    add_attempt_history_args(evolve_parser)

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
    if args.command == "evolve-once":
        return _evolve_once(args)
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
        if torch.cuda.is_available():
            payload["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "compute_capability": torch.cuda.get_device_capability(0),
            }
    except Exception as exc:
        payload["torch_error"] = f"{type(exc).__name__}: {exc}"
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


def _evolve_once(args: argparse.Namespace) -> int:
    if args.env_file:
        load_env_file(args.env_file)
    knowledge = args.knowledge.read_text(encoding="utf-8")
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
    step = finalize_attempt(args.lineage, attempt)
    if args.step_json:
        write_step(args.step_json, step)
    if args.attempts_dir:
        write_step_record(args.attempts_dir, step)
    print(json.dumps(step.as_dict(), indent=2, sort_keys=True))
    if not attempt.command_result.ok:
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
    return env
