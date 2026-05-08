from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
    if not (path / ".git").exists():
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


def decide_gate(
    candidate: dict[str, Any],
    best_geomean: float,
    *,
    best_payload: dict[str, Any] | None = None,
) -> GateDecision:
    candidate_geomean = float(candidate.get("geomean_tflops") or 0.0)
    cases = candidate.get("cases")
    if not candidate.get("all_correct"):
        return GateDecision(False, "candidate failed correctness", candidate_geomean, best_geomean)
    if not isinstance(cases, list) or not cases:
        return GateDecision(False, "candidate has no scored cases", candidate_geomean, best_geomean)
    if not math.isfinite(candidate_geomean) or candidate_geomean <= 0.0:
        return GateDecision(
            False,
            "candidate has non-positive or non-finite geomean throughput",
            candidate_geomean,
            best_geomean,
        )
    if best_payload is not None and _benchmark_signature(candidate) != _benchmark_signature(
        best_payload
    ):
        return GateDecision(
            False,
            "candidate benchmark cases differ from current best",
            candidate_geomean,
            best_geomean,
        )
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
    *,
    source_files: Mapping[str, str] | None = None,
    candidate_patch: str | None = None,
) -> GateDecision:
    init_lineage_repo(path)
    best_payload = latest_score_payload(path)
    best = _score_geomean(best_payload)
    decision = decide_gate(candidate, best, best_payload=best_payload)
    if not decision.accepted:
        return decision
    if not (candidate_patch or "").strip() and _source_snapshot_matches_latest(path, source_files):
        return GateDecision(
            False,
            "candidate source is unchanged from current best",
            decision.candidate_geomean,
            decision.best_geomean,
        )

    score_path = path / "scores" / "latest.json"
    score_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_source_artifacts(
        path,
        source_files=source_files,
        candidate_patch=candidate_patch,
    )
    _git(path, "add", "-A")
    commit_message = f"{message}\n\nAVO-Score: {json.dumps(decision.as_dict(), sort_keys=True)}"
    _git(path, "commit", "-m", commit_message)
    return decision


def best_geomean(path: Path) -> float:
    return _score_geomean(latest_score_payload(path))


def latest_score_payload(path: Path) -> dict[str, Any] | None:
    if not (path / ".git").exists() or not _has_commits(path):
        return None
    try:
        raw = _git_capture(path, "show", "HEAD:scores/latest.json")
    except subprocess.CalledProcessError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _score_geomean(payload: dict[str, Any] | None) -> float:
    if payload is None:
        return 0.0
    return float(payload.get("geomean_tflops") or 0.0)


def _benchmark_signature(payload: dict[str, Any]) -> tuple[str, ...]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return ()
    signature: list[str] = []
    for case_payload in cases:
        if not isinstance(case_payload, dict):
            continue
        case = case_payload.get("case")
        if isinstance(case, dict):
            signature.append(json.dumps(case, sort_keys=True, separators=(",", ":")))
    return tuple(sorted(signature))


def _has_commits(path: Path) -> bool:
    return subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _write_source_artifacts(
    path: Path,
    *,
    source_files: Mapping[str, str] | None,
    candidate_patch: str | None,
) -> None:
    source_root = path / "sources" / "latest"
    patch_path = path / "patches" / "latest.patch"
    if source_root.exists():
        shutil.rmtree(source_root)
    if patch_path.exists():
        patch_path.unlink()

    if source_files:
        for relative, content in sorted(source_files.items()):
            dest = source_root / _validate_candidate_source_path(relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

    if candidate_patch:
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(candidate_patch, encoding="utf-8")


def _source_snapshot_matches_latest(
    path: Path,
    source_files: Mapping[str, str] | None,
) -> bool:
    if not source_files:
        return False
    try:
        tracked = _git_capture(path, "ls-tree", "-r", "--name-only", "HEAD", "sources/latest")
    except subprocess.CalledProcessError:
        return False
    latest_paths = sorted(
        line.removeprefix("sources/latest/")
        for line in tracked.splitlines()
        if line.startswith("sources/latest/")
    )
    if latest_paths != sorted(source_files):
        return False
    for relative, content in source_files.items():
        try:
            latest_content = _git_capture(path, "show", f"HEAD:sources/latest/{relative}")
        except subprocess.CalledProcessError:
            return False
        if latest_content != content:
            return False
    return True


def _validate_candidate_source_path(path: str) -> PurePosixPath:
    if not path:
        raise ValueError("source path is empty")
    if "\x00" in path or "\\" in path:
        raise ValueError("source path contains unsupported characters")
    posix_path = PurePosixPath(path)
    if posix_path.is_absolute():
        raise ValueError("absolute source paths are not supported")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise ValueError("source path traversal is not supported")
    if ".git" in posix_path.parts:
        raise ValueError("source path must not contain .git")
    if not posix_path.as_posix().startswith("candidates/"):
        raise ValueError("source paths must be under candidates/")
    return posix_path


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True)


def _git_capture(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=path, text=True, stderr=subprocess.DEVNULL)


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
