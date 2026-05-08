import json
import math
import subprocess

from avo.lineage import (
    best_geomean,
    commit_score,
    decide_gate,
    init_lineage_repo,
    seed_baseline,
)


def test_gate_rejects_incorrect_candidate() -> None:
    decision = decide_gate({"all_correct": False, "geomean_tflops": 10.0}, best_geomean=1.0)
    assert not decision.accepted
    assert "correctness" in decision.reason


def test_gate_rejects_regression() -> None:
    decision = decide_gate(
        {"all_correct": True, "geomean_tflops": 0.5, "cases": [{}]},
        best_geomean=1.0,
    )
    assert not decision.accepted
    assert "regressed" in decision.reason


def test_gate_rejects_empty_cases() -> None:
    decision = decide_gate({"all_correct": True, "geomean_tflops": 1.0, "cases": []}, 0.0)

    assert not decision.accepted
    assert "no scored cases" in decision.reason


def test_gate_rejects_benchmark_case_mismatch() -> None:
    best = score_payload(seq_len=128, geomean=1.0)
    candidate = score_payload(seq_len=256, geomean=2.0)

    decision = decide_gate(candidate, 1.0, best_payload=best)

    assert not decision.accepted
    assert "benchmark cases differ" in decision.reason


def test_gate_rejects_non_positive_or_non_finite_geomean() -> None:
    zero = decide_gate({"all_correct": True, "geomean_tflops": 0.0, "cases": [{}]}, 0.0)
    infinite = decide_gate(
        {"all_correct": True, "geomean_tflops": math.inf, "cases": [{}]},
        0.0,
    )

    assert not zero.accepted
    assert "non-positive" in zero.reason
    assert not infinite.accepted
    assert "non-positive" in infinite.reason


def test_commit_score_records_payload(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    candidate = {"backend": "mock", "all_correct": True, "geomean_tflops": 12.5, "cases": [{}]}
    decision = commit_score(repo, candidate)
    assert decision.accepted
    assert best_geomean(repo) == 12.5
    latest = subprocess.check_output(
        ["git", "show", "HEAD:scores/latest.json"],
        cwd=repo,
        text=True,
    )
    assert '"geomean_tflops": 12.5' in latest


def test_commit_score_rejects_changed_benchmark_shape(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    baseline = score_payload(seq_len=128, geomean=1.0)
    changed_shape = score_payload(seq_len=256, geomean=2.0)
    commit_score(repo, baseline)
    head_before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)

    decision = commit_score(repo, changed_shape)

    head_after = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)
    assert not decision.accepted
    assert "benchmark cases differ" in decision.reason
    assert head_after == head_before
    assert best_geomean(repo) == 1.0


def test_commit_score_rejects_unchanged_source_rerun(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    source = {"candidates/seed.py": "VALUE = 1\n"}
    commit_score(
        repo,
        score_payload(seq_len=128, geomean=1.0),
        source_files=source,
        candidate_patch="diff --git a/candidates/seed.py b/candidates/seed.py\n",
    )
    head_before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)

    decision = commit_score(
        repo,
        score_payload(seq_len=128, geomean=2.0),
        source_files=source,
    )

    head_after = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)
    assert not decision.accepted
    assert "source is unchanged" in decision.reason
    assert head_after == head_before
    assert best_geomean(repo) == 1.0


def test_commit_score_records_accepted_source_artifacts(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    candidate = {"backend": "mock", "all_correct": True, "geomean_tflops": 12.5, "cases": [{}]}
    patch = "diff --git a/candidates/seed.py b/candidates/seed.py\n"

    decision = commit_score(
        repo,
        candidate,
        source_files={"candidates/seed.py": "VALUE = 2\n"},
        candidate_patch=patch,
    )

    assert decision.accepted
    source = subprocess.check_output(
        ["git", "show", "HEAD:sources/latest/candidates/seed.py"],
        cwd=repo,
        text=True,
    )
    stored_patch = subprocess.check_output(
        ["git", "show", "HEAD:patches/latest.patch"],
        cwd=repo,
        text=True,
    )
    assert source == "VALUE = 2\n"
    assert stored_patch == patch


def test_commit_score_does_not_record_source_for_rejected_candidate(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    baseline = {"backend": "mock", "all_correct": True, "geomean_tflops": 12.5, "cases": [{}]}
    regression = {"backend": "mock", "all_correct": True, "geomean_tflops": 1.0, "cases": [{}]}
    commit_score(repo, baseline)
    head_before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)

    decision = commit_score(
        repo,
        regression,
        source_files={"candidates/seed.py": "VALUE = 2\n"},
        candidate_patch="diff --git a/candidates/seed.py b/candidates/seed.py\n",
    )

    head_after = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)
    assert not decision.accepted
    assert head_after == head_before
    assert not (repo / "sources").exists()


def test_seed_baseline_records_json(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    payload = {
        "backend": "flash-attn",
        "all_correct": True,
        "geomean_tflops": 25.0,
        "cases": [{}],
    }
    seeded = seed_baseline(repo, payload, force=True)
    assert seeded["role"] == "baseline"
    baseline = json.loads((repo / "scores" / "baseline.json").read_text(encoding="utf-8"))
    latest = json.loads((repo / "scores" / "latest.json").read_text(encoding="utf-8"))
    assert baseline == latest


def score_payload(*, seq_len: int, geomean: float) -> dict:
    return {
        "backend": "candidate",
        "all_correct": True,
        "geomean_tflops": geomean,
        "cases": [
            {
                "case": {
                    "causal": False,
                    "dtype": "bf16",
                    "head_dim": 128,
                    "num_heads": 4,
                    "seq_len": seq_len,
                    "total_tokens": seq_len * 4,
                },
                "correct": True,
                "tflops": geomean,
            }
        ],
    }
