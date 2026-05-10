from __future__ import annotations

import importlib.util
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from .config import AMPERE_A6000, AttentionCase

BackendName = Literal["torch-sdpa", "flash-attn", "candidate"]
CANDIDATE_SOURCE_FILE_ATTRIBUTES = ("AVO_SOURCE_FILES", "__avo_source_files__")
CANDIDATE_RUNTIME_SOURCE_SUFFIXES = frozenset({".py", ".cpp", ".cu", ".cuh", ".h", ".hpp"})
SKIPPED_RUNTIME_SOURCE_PARTS = frozenset({"__pycache__"})


@dataclass(frozen=True)
class CaseScore:
    backend: str
    case: AttentionCase
    correct: bool
    milliseconds: float | None
    tflops: float
    max_abs_error: float | None
    error: str | None = None
    timing_samples_ms: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "case": self.case.as_dict(),
            "correct": self.correct,
            "milliseconds": self.milliseconds,
            "tflops": self.tflops,
            "max_abs_error": self.max_abs_error,
            "error": self.error,
            "timing": timing_summary(self.timing_samples_ms),
        }


def attention_forward_flops(case: AttentionCase) -> int:
    dense = 4 * case.batch_size * case.num_heads * case.seq_len * case.seq_len * case.head_dim
    return dense // 2 if case.causal else dense


def tflops_from_ms(case: AttentionCase, milliseconds: float) -> float:
    if milliseconds <= 0:
        return math.inf
    return attention_forward_flops(case) / (milliseconds / 1000.0) / 1e12


def timing_summary(samples_ms: Iterable[float]) -> dict[str, Any]:
    samples = tuple(float(sample) for sample in samples_ms)
    if not samples:
        return {
            "samples_ms": [],
            "trials": 0,
            "min_ms": None,
            "median_ms": None,
            "mean_ms": None,
            "cv": None,
        }
    mean = statistics.fmean(samples)
    return {
        "samples_ms": list(samples),
        "trials": len(samples),
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": mean,
        "cv": statistics.pstdev(samples) / mean if mean > 0 and len(samples) > 1 else 0.0,
    }


def geometric_mean(values: Iterable[float]) -> float:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def score_summary(backend: str, scores: list[CaseScore]) -> dict[str, Any]:
    return {
        "backend": backend,
        "all_correct": bool(scores) and all(score.correct for score in scores),
        "geomean_tflops": geometric_mean(score.tflops for score in scores if score.correct),
        "cases": [score.as_dict() for score in scores],
    }


def score_backend(
    backend: BackendName,
    cases: list[AttentionCase],
    warmup: int,
    repeats: int,
    trials: int = 1,
    seed: int = 0,
    candidate: Path | None = None,
) -> dict[str, Any]:
    if backend == "torch-sdpa":
        scores = [
            _score_torch_sdpa(case, warmup=warmup, repeats=repeats, trials=trials, seed=seed)
            for case in cases
        ]
    elif backend == "flash-attn":
        scores = [
            _score_flash_attn(case, warmup=warmup, repeats=repeats, trials=trials, seed=seed)
            for case in cases
        ]
    elif backend == "candidate":
        if candidate is None:
            raise ValueError("candidate backend requires --candidate")
        candidate_source_files: tuple[str, ...] = ()
        try:
            module = _load_candidate(candidate)
            candidate_source_files = _candidate_declared_source_files(module, candidate)
        except Exception as exc:
            scores = [
                _failed_candidate_score(case, f"{type(exc).__name__}: {exc}") for case in cases
            ]
        else:
            scores = [
                _score_candidate(
                    module,
                    candidate,
                    case,
                    warmup=warmup,
                    repeats=repeats,
                    trials=trials,
                    seed=seed,
                )
                for case in cases
            ]
            candidate_source_files = _dedupe_source_files(
                (
                    *candidate_source_files,
                    *_candidate_runtime_imported_source_files(candidate),
                )
            )
    else:
        raise ValueError(f"unsupported backend: {backend}")
    summary = score_summary(backend, scores)
    summary["benchmark"] = benchmark_metadata(
        cases=cases,
        warmup=warmup,
        repeats=repeats,
        trials=trials,
        seed=seed,
    )
    if backend == "candidate" and candidate is not None:
        summary["candidate_path"] = str(candidate)
        if candidate_source_files:
            summary["candidate_source_files"] = list(candidate_source_files)
    return summary


def benchmark_metadata(
    *,
    cases: list[AttentionCase],
    warmup: int,
    repeats: int,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "measured_at": datetime.now(UTC).isoformat(),
        "settings": {
            "warmup": warmup,
            "repeats": repeats,
            "trials": trials,
            "seed": seed,
            "case_count": len(cases),
        },
        "target": {
            "name": AMPERE_A6000.name,
            "compute_capability": list(AMPERE_A6000.compute_capability),
            "compute": AMPERE_A6000.compute,
            "sm": AMPERE_A6000.sm,
            "nvcc_gencode": AMPERE_A6000.nvcc_gencode,
        },
        "environment": _benchmark_environment(),
    }


def _require_torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 6):
        raise RuntimeError(f"expected sm_86/A6000-like GPU, got compute capability {capability}")
    return torch


def _benchmark_environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
    except ModuleNotFoundError as exc:
        payload["torch"] = {
            "installed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        return payload

    torch_payload: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    payload["torch"] = torch_payload
    if not torch.cuda.is_available():
        return payload

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    payload["gpu"] = {
        "index": device_index,
        "name": torch.cuda.get_device_name(device_index),
        "compute_capability": list(torch.cuda.get_device_capability(device_index)),
        "total_memory_bytes": props.total_memory,
        "multi_processor_count": props.multi_processor_count,
    }
    if hasattr(torch.cuda, "is_bf16_supported"):
        payload["gpu"]["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    return payload


def _make_inputs(case: AttentionCase, seed: int):
    torch = _require_torch()
    if case.dtype == "bf16":
        dtype = torch.bfloat16
    elif case.dtype == "fp16":
        dtype = torch.float16
    elif case.dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError("dtype must be one of: bf16, fp16, fp32")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    shape = (case.batch_size, case.num_heads, case.seq_len, case.head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    return q, k, v


def _reference_sdpa(q, k, v, causal: bool):
    import torch.nn.functional as F

    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)


def _time_cuda(fn: Callable[[], object], warmup: int, repeats: int) -> float:
    return _time_cuda_samples(fn, warmup=warmup, repeats=repeats, trials=1)[0]


def _time_cuda_samples(
    fn: Callable[[], object],
    warmup: int,
    repeats: int,
    trials: int,
) -> tuple[float, ...]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if trials <= 0:
        raise ValueError("trials must be positive")
    torch = _require_torch()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)) / repeats)
    return tuple(samples)


def _score_torch_sdpa(
    case: AttentionCase,
    warmup: int,
    repeats: int,
    trials: int,
    seed: int,
) -> CaseScore:
    try:
        q, k, v = _make_inputs(case, seed)
        output = _reference_sdpa(q, k, v, case.causal)
        if not output.isfinite().all().item():
            raise RuntimeError("reference output contains non-finite values")
        timing_samples = _time_cuda_samples(
            lambda: _reference_sdpa(q, k, v, case.causal),
            warmup=warmup,
            repeats=repeats,
            trials=trials,
        )
        milliseconds = statistics.median(timing_samples)
        return CaseScore(
            backend="torch-sdpa",
            case=case,
            correct=True,
            milliseconds=milliseconds,
            tflops=tflops_from_ms(case, milliseconds),
            max_abs_error=0.0,
            timing_samples_ms=timing_samples,
        )
    except Exception as exc:
        return CaseScore(
            backend="torch-sdpa",
            case=case,
            correct=False,
            milliseconds=None,
            tflops=0.0,
            max_abs_error=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _score_flash_attn(
    case: AttentionCase,
    warmup: int,
    repeats: int,
    trials: int,
    seed: int,
) -> CaseScore:
    try:
        torch = _require_torch()
        from flash_attn import flash_attn_func

        q_bhsd, k_bhsd, v_bhsd = _make_inputs(case, seed)
        q = q_bhsd.transpose(1, 2).contiguous()
        k = k_bhsd.transpose(1, 2).contiguous()
        v = v_bhsd.transpose(1, 2).contiguous()
        reference = (
            _reference_sdpa(q_bhsd, k_bhsd, v_bhsd, case.causal)
            .transpose(1, 2)
            .contiguous()
        )
        candidate = flash_attn_func(q, k, v, dropout_p=0.0, causal=case.causal)
        max_abs_error = float((candidate.float() - reference.float()).abs().max().item())
        correct = bool(torch.allclose(candidate.float(), reference.float(), atol=3e-2, rtol=3e-2))
        timing_samples = _time_cuda_samples(
            lambda: flash_attn_func(q, k, v, dropout_p=0.0, causal=case.causal),
            warmup=warmup,
            repeats=repeats,
            trials=trials,
        )
        milliseconds = statistics.median(timing_samples)
        return CaseScore(
            backend="flash-attn",
            case=case,
            correct=correct,
            milliseconds=milliseconds,
            tflops=tflops_from_ms(case, milliseconds) if correct else 0.0,
            max_abs_error=max_abs_error,
            timing_samples_ms=timing_samples,
            error=None if correct else "flash-attn output failed tolerance check",
        )
    except Exception as exc:
        return CaseScore(
            backend="flash-attn",
            case=case,
            correct=False,
            milliseconds=None,
            tflops=0.0,
            max_abs_error=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _load_candidate(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    module_name = f"avo_candidate_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load candidate module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    attention = getattr(module, "attention", None)
    if not callable(attention):
        raise ValueError("candidate module must define callable attention(q, k, v, causal)")
    return module


def _candidate_declared_source_files(module: Any, candidate_path: Path) -> tuple[str, ...]:
    paths: list[str] = []
    for attribute in CANDIDATE_SOURCE_FILE_ATTRIBUTES:
        raw_files = getattr(module, attribute, None)
        if raw_files is None:
            continue
        for raw_file in _candidate_source_file_values(raw_files, attribute=attribute):
            paths.append(_candidate_source_file_path(candidate_path, raw_file))
    return tuple(dict.fromkeys(paths))


def _candidate_source_file_values(
    value: Any,
    *,
    attribute: str,
) -> tuple[str | os.PathLike[str], ...]:
    if isinstance(value, (str, os.PathLike)):
        return (value,)
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{attribute} must be a path or iterable of paths") from exc
    for item in values:
        if not isinstance(item, (str, os.PathLike)):
            raise ValueError(f"{attribute} entries must be paths")
    return values


def _candidate_source_file_path(candidate_path: Path, raw_path: str | os.PathLike[str]) -> str:
    path = Path(os.fspath(raw_path))
    if path.is_absolute():
        return path.as_posix()
    if path.as_posix().startswith("candidates/"):
        return path.as_posix()
    return (candidate_path.parent / path).as_posix()


def _candidate_runtime_imported_source_files(candidate_path: Path) -> tuple[str, ...]:
    source_root = _candidate_source_root(candidate_path)
    if source_root is None:
        return ()
    paths = []
    for module in tuple(sys.modules.values()):
        raw_file = getattr(module, "__file__", None)
        if not isinstance(raw_file, str):
            continue
        normalized = _candidate_runtime_source_path(source_root, Path(raw_file))
        if normalized is not None:
            paths.append(normalized)
    return _dedupe_source_files(paths)


def _candidate_source_root(candidate_path: Path) -> Path | None:
    resolved = candidate_path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if parent.name == "candidates":
            return parent.parent
    return None


def _candidate_runtime_source_path(source_root: Path, raw_path: Path) -> str | None:
    try:
        relative = raw_path.resolve().relative_to((source_root / "candidates").resolve())
    except (OSError, ValueError):
        return None
    if (
        raw_path.suffix not in CANDIDATE_RUNTIME_SOURCE_SUFFIXES
        or any(part in SKIPPED_RUNTIME_SOURCE_PARTS for part in relative.parts)
    ):
        return None
    return PurePosixPath("candidates", *relative.parts).as_posix()


def _dedupe_source_files(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(paths))


def _score_candidate(
    module: Any,
    candidate_path: Path,
    case: AttentionCase,
    warmup: int,
    repeats: int,
    trials: int,
    seed: int,
) -> CaseScore:
    try:
        torch = _require_torch()
        q, k, v = _make_inputs(case, seed)
        reference = _reference_sdpa(q, k, v, case.causal)
        candidate = module.attention(q, k, v, case.causal)
        if tuple(candidate.shape) != tuple(reference.shape):
            candidate_shape = tuple(candidate.shape)
            reference_shape = tuple(reference.shape)
            raise RuntimeError(
                f"candidate output shape {candidate_shape} != reference {reference_shape}"
            )
        if not candidate.isfinite().all().item():
            raise RuntimeError("candidate output contains non-finite values")
        max_abs_error = float((candidate.float() - reference.float()).abs().max().item())
        correct = bool(torch.allclose(candidate.float(), reference.float(), atol=3e-2, rtol=3e-2))
        timing_samples = _time_cuda_samples(
            lambda: module.attention(q, k, v, case.causal),
            warmup=warmup,
            repeats=repeats,
            trials=trials,
        )
        milliseconds = statistics.median(timing_samples)
        return CaseScore(
            backend="candidate",
            case=case,
            correct=correct,
            milliseconds=milliseconds,
            tflops=tflops_from_ms(case, milliseconds) if correct else 0.0,
            max_abs_error=max_abs_error,
            timing_samples_ms=timing_samples,
            error=None if correct else f"{candidate_path} failed tolerance check",
        )
    except Exception as exc:
        return CaseScore(
            backend="candidate",
            case=case,
            correct=False,
            milliseconds=None,
            tflops=0.0,
            max_abs_error=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _failed_candidate_score(case: AttentionCase, error: str) -> CaseScore:
    return CaseScore(
        backend="candidate",
        case=case,
        correct=False,
        milliseconds=None,
        tflops=0.0,
        max_abs_error=None,
        error=error,
    )


def sleep_score(seconds: float) -> dict[str, Any]:
    time.sleep(seconds)
    return {"slept": seconds}
