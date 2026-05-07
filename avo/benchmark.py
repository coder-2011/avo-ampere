from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from .config import AttentionCase

BackendName = Literal["torch-sdpa", "flash-attn"]


@dataclass(frozen=True)
class CaseScore:
    backend: str
    case: AttentionCase
    correct: bool
    milliseconds: float | None
    tflops: float
    max_abs_error: float | None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "case": self.case.as_dict(),
            "correct": self.correct,
            "milliseconds": self.milliseconds,
            "tflops": self.tflops,
            "max_abs_error": self.max_abs_error,
            "error": self.error,
        }


def attention_forward_flops(case: AttentionCase) -> int:
    dense = 4 * case.batch_size * case.num_heads * case.seq_len * case.seq_len * case.head_dim
    return dense // 2 if case.causal else dense


def tflops_from_ms(case: AttentionCase, milliseconds: float) -> float:
    if milliseconds <= 0:
        return math.inf
    return attention_forward_flops(case) / (milliseconds / 1000.0) / 1e12


def geometric_mean(values: Iterable[float]) -> float:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def score_summary(backend: str, scores: list[CaseScore]) -> dict[str, Any]:
    return {
        "backend": backend,
        "all_correct": all(score.correct for score in scores),
        "geomean_tflops": geometric_mean(score.tflops for score in scores if score.correct),
        "cases": [score.as_dict() for score in scores],
    }


def score_backend(
    backend: BackendName,
    cases: list[AttentionCase],
    warmup: int,
    repeats: int,
    seed: int = 0,
) -> dict[str, Any]:
    if backend == "torch-sdpa":
        scores = [
            _score_torch_sdpa(case, warmup=warmup, repeats=repeats, seed=seed) for case in cases
        ]
    elif backend == "flash-attn":
        scores = [
            _score_flash_attn(case, warmup=warmup, repeats=repeats, seed=seed) for case in cases
        ]
    else:
        raise ValueError(f"unsupported backend: {backend}")
    return score_summary(backend, scores)


def _require_torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 6):
        raise RuntimeError(f"expected sm_86/A6000-like GPU, got compute capability {capability}")
    return torch


def _make_inputs(case: AttentionCase, seed: int):
    torch = _require_torch()
    dtype = torch.bfloat16 if case.dtype == "bf16" else torch.float16
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
    torch = _require_torch()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / repeats


def _score_torch_sdpa(case: AttentionCase, warmup: int, repeats: int, seed: int) -> CaseScore:
    try:
        q, k, v = _make_inputs(case, seed)
        output = _reference_sdpa(q, k, v, case.causal)
        if not output.isfinite().all().item():
            raise RuntimeError("reference output contains non-finite values")
        milliseconds = _time_cuda(lambda: _reference_sdpa(q, k, v, case.causal), warmup, repeats)
        return CaseScore(
            backend="torch-sdpa",
            case=case,
            correct=True,
            milliseconds=milliseconds,
            tflops=tflops_from_ms(case, milliseconds),
            max_abs_error=0.0,
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


def _score_flash_attn(case: AttentionCase, warmup: int, repeats: int, seed: int) -> CaseScore:
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
        milliseconds = _time_cuda(
            lambda: flash_attn_func(q, k, v, dropout_p=0.0, causal=case.causal),
            warmup,
            repeats,
        )
        return CaseScore(
            backend="flash-attn",
            case=case,
            correct=correct,
            milliseconds=milliseconds,
            tflops=tflops_from_ms(case, milliseconds) if correct else 0.0,
            max_abs_error=max_abs_error,
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


def sleep_score(seconds: float) -> dict[str, Any]:
    time.sleep(seconds)
    return {"slept": seconds}
