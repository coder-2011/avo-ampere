import pytest

from avo.benchmark import (
    CaseScore,
    attention_forward_flops,
    benchmark_metadata,
    geometric_mean,
    score_summary,
    tflops_from_ms,
    timing_summary,
)
from avo.config import AttentionCase


def test_attention_flops_dense_and_causal() -> None:
    dense = AttentionCase(seq_len=4096, causal=False)
    causal = AttentionCase(seq_len=4096, causal=True)
    assert attention_forward_flops(dense) == 4 * 8 * 16 * 4096 * 4096 * 128
    assert attention_forward_flops(causal) == attention_forward_flops(dense) // 2


def test_tflops_from_ms() -> None:
    case = AttentionCase(seq_len=4096, causal=False)
    assert tflops_from_ms(case, 1.0) == attention_forward_flops(case) / 1e-3 / 1e12


def test_geometric_mean_ignores_zero_failed_scores() -> None:
    assert geometric_mean([2.0, 8.0, 0.0]) == 4.0
    assert geometric_mean([0.0]) == 0.0


def test_score_summary_treats_empty_case_set_as_not_correct() -> None:
    summary = score_summary("candidate", [])

    assert summary["all_correct"] is False
    assert summary["geomean_tflops"] == 0.0
    assert summary["cases"] == []


def test_timing_summary_reports_replicate_statistics() -> None:
    summary = timing_summary([3.0, 1.0, 2.0])

    assert summary["samples_ms"] == [3.0, 1.0, 2.0]
    assert summary["trials"] == 3
    assert summary["min_ms"] == 1.0
    assert summary["median_ms"] == 2.0
    assert summary["mean_ms"] == 2.0
    assert summary["cv"] == pytest.approx(0.40824829046)


def test_case_score_serializes_empty_timing_stats() -> None:
    payload = CaseScore(
        backend="mock",
        case=AttentionCase(seq_len=16, causal=False),
        correct=False,
        milliseconds=None,
        tflops=0.0,
        max_abs_error=None,
        error="failed",
    ).as_dict()

    assert payload["timing"] == {
        "samples_ms": [],
        "trials": 0,
        "min_ms": None,
        "median_ms": None,
        "mean_ms": None,
        "cv": None,
    }


def test_benchmark_metadata_records_settings_and_target() -> None:
    metadata = benchmark_metadata(
        cases=[AttentionCase(seq_len=16, causal=False)],
        warmup=1,
        repeats=2,
        trials=3,
        seed=4,
    )

    assert metadata["settings"] == {
        "warmup": 1,
        "repeats": 2,
        "trials": 3,
        "seed": 4,
        "case_count": 1,
    }
    assert metadata["target"]["sm"] == "sm_86"
    assert metadata["target"]["nvcc_gencode"] == "-gencode=arch=compute_86,code=sm_86"
    assert "measured_at" in metadata
    assert "environment" in metadata
