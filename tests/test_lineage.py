import hashlib
import json
import math
import subprocess

from avo.lineage import (
    best_geomean,
    best_score_payload_for_signature,
    commit_score,
    decide_gate,
    init_lineage_repo,
    lineage_score_summary,
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


def test_gate_rejects_one_shot_score_against_existing_best() -> None:
    best = score_payload(seq_len=128, geomean=100.0, repeats=3, warmup=2)
    candidate = score_payload(seq_len=128, geomean=125.0, repeats=1, warmup=1)

    decision = decide_gate(candidate, 100.0, best_payload=best)

    assert not decision.accepted
    assert "one-shot score must be confirmed" in decision.reason
    assert "repeats>=3/warmup>=2" in decision.reason


def test_gate_accepts_initial_one_shot_score_without_existing_best() -> None:
    candidate = score_payload(seq_len=128, geomean=100.6, repeats=1, warmup=1)

    decision = decide_gate(candidate, 0.0, best_payload=None)

    assert decision.accepted


def test_gate_accepts_confirmed_low_margin_score() -> None:
    best = score_payload(seq_len=128, geomean=100.0, repeats=3, warmup=2)
    candidate = score_payload(seq_len=128, geomean=100.4, repeats=3, warmup=2)

    decision = decide_gate(candidate, 100.0, best_payload=best)

    assert decision.accepted


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


def test_commit_score_accepts_new_benchmark_shape_lane(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    baseline = score_payload(seq_len=128, geomean=1.0)
    changed_shape = score_payload(seq_len=256, geomean=2.0)
    commit_score(repo, baseline)

    decision = commit_score(repo, changed_shape)

    assert decision.accepted
    assert "established benchmark case set" in decision.reason
    assert best_geomean(repo) == 2.0
    assert best_score_payload_for_signature(repo, baseline)["geomean_tflops"] == 1.0
    assert best_score_payload_for_signature(repo, changed_shape)["geomean_tflops"] == 2.0


def test_commit_score_accepts_new_benchmark_shape_for_unchanged_source(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    source = {"candidates/seed.py": "VALUE = 1\n"}
    commit_score(
        repo,
        score_payload(seq_len=128, geomean=1.0),
        source_files=source,
        candidate_patch="diff --git a/candidates/seed.py b/candidates/seed.py\n",
    )

    decision = commit_score(
        repo,
        score_payload(seq_len=256, geomean=2.0),
        source_files=source,
    )

    assert decision.accepted
    assert "established benchmark case set" in decision.reason


def test_commit_score_rejects_same_benchmark_shape_regression(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    baseline = score_payload(seq_len=128, geomean=2.0)
    regression = score_payload(seq_len=128, geomean=1.0)
    commit_score(repo, baseline)
    head_before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)

    decision = commit_score(repo, regression)

    head_after = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)
    assert not decision.accepted
    assert "regressed" in decision.reason
    assert head_after == head_before
    assert best_geomean(repo) == 2.0


def test_lineage_score_summary_lists_benchmark_lanes(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    commit_score(repo, score_payload(seq_len=128, geomean=1.0))
    commit_score(repo, score_payload(seq_len=256, geomean=2.0))

    summary = json.loads(lineage_score_summary(repo))

    lanes = summary["benchmark_lanes"]
    assert [lane["geomean_tflops"] for lane in lanes] == [2.0, 1.0]
    assert {lane["cases"][0]["seq_len"] for lane in lanes} == {128, 256}
    assert summary["latest"]["cases"][0]["seq_len"] == 256


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
    manifest = json.loads(
        subprocess.check_output(
            ["git", "show", "HEAD:sources/latest/manifest.json"],
            cwd=repo,
            text=True,
        )
    )
    assert source == "VALUE = 2\n"
    assert stored_patch == patch
    assert manifest["files"] == [
        {
            "bytes": len(b"VALUE = 2\n"),
            "path": "candidates/seed.py",
            "sha256": hashlib.sha256(b"VALUE = 2\n").hexdigest(),
        }
    ]


def test_commit_score_does_not_record_source_for_rejected_candidate(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    baseline = score_payload(seq_len=128, geomean=12.5)
    regression = score_payload(seq_len=128, geomean=1.0)
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


def test_baseline_does_not_block_candidate_lineage_progress(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    baseline = score_payload(seq_len=128, geomean=25.0)
    baseline["backend"] = "flash-attn"
    seeded = seed_baseline(repo, baseline, force=True)
    candidate = score_payload(seq_len=128, geomean=5.0)

    decision = commit_score(repo, candidate)

    assert seeded["role"] == "baseline"
    assert decision.accepted
    assert "established benchmark case set" in decision.reason
    assert best_score_payload_for_signature(repo, candidate)["geomean_tflops"] == 5.0
    assert json.loads((repo / "scores" / "baseline.json").read_text(encoding="utf-8"))[
        "geomean_tflops"
    ] == 25.0


def test_baseline_rejects_candidate_with_different_benchmark_signature(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    baseline = score_payload(seq_len=128, geomean=25.0)
    baseline["backend"] = "flash-attn"
    seed_baseline(repo, baseline, force=True)
    head_before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)

    decision = commit_score(repo, score_payload(seq_len=256, geomean=5.0))

    head_after = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)
    assert not decision.accepted
    assert "baseline target suite" in decision.reason
    assert head_after == head_before
    assert best_score_payload_for_signature(repo, score_payload(seq_len=256, geomean=5.0)) is None


def test_candidate_regression_rejects_against_prior_candidate_not_baseline(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    baseline = score_payload(seq_len=128, geomean=25.0)
    baseline["backend"] = "flash-attn"
    seed_baseline(repo, baseline, force=True)
    commit_score(repo, score_payload(seq_len=128, geomean=5.0))
    head_before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)

    decision = commit_score(repo, score_payload(seq_len=128, geomean=4.0))

    head_after = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)
    assert not decision.accepted
    assert "regressed" in decision.reason
    assert decision.best_geomean == 5.0
    assert head_after == head_before


def test_lineage_summary_keeps_baseline_and_candidate_lanes_for_same_signature(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    baseline = score_payload(seq_len=128, geomean=25.0)
    baseline["backend"] = "flash-attn"
    seed_baseline(repo, baseline, force=True)
    commit_score(repo, score_payload(seq_len=128, geomean=5.0))

    summary = json.loads(lineage_score_summary(repo))

    lanes = summary["benchmark_lanes"]
    assert [lane["geomean_tflops"] for lane in lanes] == [25.0, 5.0]
    assert [lane.get("role") for lane in lanes] == ["baseline", None]
    comparison = summary["baseline_comparisons"][0]
    assert comparison["candidate_geomean_tflops"] == 5.0
    assert comparison["baseline_geomean_tflops"] == 25.0
    assert comparison["candidate_vs_baseline"] == 0.2
    assert comparison["baseline_vs_candidate"] == 5.0
    assert comparison["gap_tflops"] == -20.0


def score_payload(
    *,
    seq_len: int,
    geomean: float,
    repeats: int | None = None,
    warmup: int | None = None,
) -> dict:
    payload = {
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
    if repeats is not None or warmup is not None:
        payload["benchmark"] = {
            "settings": {
                "repeats": 0 if repeats is None else repeats,
                "warmup": 0 if warmup is None else warmup,
            }
        }
    return payload
