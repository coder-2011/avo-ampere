from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    candidate_geomean: float
    best_geomean: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "candidate_geomean": self.candidate_geomean,
            "best_geomean": self.best_geomean,
        }


def init_lineage_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-b", "main")
    _ensure_git_identity(path)
    (path / "scores").mkdir(exist_ok=True)
    readme = path / "README.md"
    if not readme.exists():
        readme.write_text(
            "# AVO Lineage\n\n"
            "Each accepted commit records one correctness-gated candidate score.\n",
            encoding="utf-8",
        )
    _git(path, "add", "README.md", "scores")
    if not _has_commits(path):
        _git(path, "commit", "-m", "chore: initialize AVO lineage")


def seed_baseline(
    path: Path,
    payload: dict[str, Any],
    message: str = "chore: seed baseline",
    force: bool = False,
) -> dict[str, Any]:
    init_lineage_repo(path)
    if not payload.get("all_correct"):
        raise ValueError("baseline candidate failed correctness gate")
    path.joinpath("scores").mkdir(exist_ok=True)
    baseline_path = path / "scores" / "baseline.json"
    latest_path = path / "scores" / "latest.json"
    if baseline_path.exists() and not force:
        raise FileExistsError(
            f"baseline already exists at {baseline_path}. Use force=True to overwrite."
        )

    seeded = dict(payload)
    seeded.setdefault("source", "flash-attn")
    seeded.setdefault("role", "baseline")
    seeded.setdefault("backend", "flash-attn")

    baseline_path.write_text(json.dumps(seeded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_path.write_text(json.dumps(seeded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _git(path, "add", "scores/baseline.json", "scores/latest.json")
    _git(path, "commit", "-m", message)
    return seeded


def decide_gate(candidate: dict[str, Any], best_geomean: float) -> GateDecision:
    candidate_geomean = float(candidate.get("geomean_tflops") or 0.0)
    if not candidate.get("all_correct"):
        return GateDecision(False, "candidate failed correctness", candidate_geomean, best_geomean)
    if candidate_geomean + 1e-9 < best_geomean:
        return GateDecision(
            False,
            "candidate regressed geomean throughput",
            candidate_geomean,
            best_geomean,
        )
    return GateDecision(
        True,
        "candidate passed correctness and throughput gate",
        candidate_geomean,
        best_geomean,
    )


def commit_score(
    path: Path,
    candidate: dict[str, Any],
    message: str = "evolve: accept candidate",
) -> GateDecision:
    init_lineage_repo(path)
    best = best_geomean(path)
    decision = decide_gate(candidate, best)
    if not decision.accepted:
        return decision

    score_path = path / "scores" / "latest.json"
    score_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _git(path, "add", "scores/latest.json")
    commit_message = f"{message}\n\nAVO-Score: {json.dumps(decision.as_dict(), sort_keys=True)}"
    _git(path, "commit", "-m", commit_message)
    return decision


def best_geomean(path: Path) -> float:
    if not (path / ".git").exists() or not _has_commits(path):
        return 0.0
    try:
        raw = _git_capture(path, "show", "HEAD:scores/latest.json")
    except subprocess.CalledProcessError:
        return 0.0
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0.0
    return float(payload.get("geomean_tflops") or 0.0)


def _has_commits(path: Path) -> bool:
    return subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True)


def _git_capture(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=path, text=True)


def _ensure_git_identity(path: Path) -> None:
    if _git_config(path, "user.email") and _git_config(path, "user.name"):
        return
    _git(path, "config", "user.email", "avo-agent@example.invalid")
    _git(path, "config", "user.name", "AVO Agent")


def _git_config(path: Path, key: str) -> str:
    result = subprocess.run(
        ["git", "config", "--get", key],
        cwd=path,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""
