import subprocess

from avo.lineage import best_geomean, commit_score, decide_gate, init_lineage_repo


def test_gate_rejects_incorrect_candidate() -> None:
    decision = decide_gate({"all_correct": False, "geomean_tflops": 10.0}, best_geomean=1.0)
    assert not decision.accepted
    assert "correctness" in decision.reason


def test_gate_rejects_regression() -> None:
    decision = decide_gate({"all_correct": True, "geomean_tflops": 0.5}, best_geomean=1.0)
    assert not decision.accepted
    assert "regressed" in decision.reason


def test_commit_score_records_payload(tmp_path) -> None:
    repo = tmp_path / "lineage"
    init_lineage_repo(repo)
    candidate = {"backend": "mock", "all_correct": True, "geomean_tflops": 12.5, "cases": []}
    decision = commit_score(repo, candidate)
    assert decision.accepted
    assert best_geomean(repo) == 12.5
    latest = subprocess.check_output(
        ["git", "show", "HEAD:scores/latest.json"],
        cwd=repo,
        text=True,
    )
    assert '"geomean_tflops": 12.5' in latest
