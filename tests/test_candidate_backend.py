import re
from pathlib import Path

import pytest

from avo.benchmark import _load_candidate, score_backend
from avo.config import AttentionCase


def test_candidate_backend_requires_path() -> None:
    with pytest.raises(ValueError, match="requires --candidate"):
        score_backend("candidate", [], warmup=0, repeats=0)


def test_candidate_backend_loads_attention_function(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        "def attention(q, k, v, causal):\n"
        "    return q\n",
        encoding="utf-8",
    )

    summary = score_backend("candidate", [], warmup=0, repeats=0, candidate=candidate)

    assert summary["backend"] == "candidate"
    assert summary["candidate_path"] == str(candidate)
    assert summary["benchmark"]["settings"]["warmup"] == 0
    assert summary["benchmark"]["settings"]["repeats"] == 0
    assert summary["benchmark"]["settings"]["trials"] == 1
    assert summary["benchmark"]["target"]["sm"] == "sm_86"


def test_candidate_backend_rejects_missing_attention(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must define callable attention"):
        _load_candidate(candidate)


def test_candidate_backend_reports_load_failure_as_failed_score(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text("VALUE = 1\n", encoding="utf-8")

    summary = score_backend(
        "candidate",
        [AttentionCase(seq_len=4096, causal=False)],
        warmup=0,
        repeats=0,
        candidate=candidate,
    )

    assert summary["all_correct"] is False
    assert summary["cases"][0]["correct"] is False
    assert "must define callable attention" in summary["cases"][0]["error"]


def test_mma_wrapper_sequences_are_supported_by_kernel_guard() -> None:
    wrapper = Path("candidates/cuda_mma_attention_seed.py").read_text(encoding="utf-8")
    kernel = Path("candidates/cuda_mma_attention/attention_kernel.cu").read_text(
        encoding="utf-8",
    )

    sequences_match = re.search(r"SMOKE_SEQUENCES\s*=\s*\{(?P<body>[^}]*)\}", wrapper)
    max_seq_match = re.search(r"constexpr\s+int\s+kMaxSeqLen\s*=\s*(?P<value>\d+);", kernel)
    assert sequences_match is not None
    assert max_seq_match is not None
    sequences = {
        int(item.strip())
        for item in sequences_match.group("body").split(",")
        if item.strip()
    }
    max_seq_len = int(max_seq_match.group("value"))

    assert sequences
    assert all(seq_len % 16 == 0 and seq_len <= max_seq_len for seq_len in sequences)
    assert "seq_len <= kMaxSeqLen" in kernel
    assert "seq_len % kTile == 0" in kernel
