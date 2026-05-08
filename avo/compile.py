from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AMPERE_A6000


@dataclass(frozen=True)
class CompileResult:
    ok: bool
    command: list[str]
    returncode: int
    object_path: Path
    stdout_tail: str
    stderr_tail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "command": self.command,
            "returncode": self.returncode,
            "object_path": str(self.object_path),
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


def compile_cuda_source(
    source: Path,
    out_dir: Path,
    *,
    timeout_s: int = 120,
    nvcc: str | None = None,
    env: dict[str, str] | None = None,
    include_dirs: Sequence[str | Path] = (),
) -> CompileResult:
    if not source.exists():
        raise FileNotFoundError(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    object_path = out_dir / f"{source.stem}.o"
    compiler = nvcc or _default_nvcc(env or os.environ)
    command = [
        compiler,
        AMPERE_A6000.nvcc_gencode,
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "--expt-relaxed-constexpr",
        "--ptxas-options=-v",
        *[flag for include_dir in include_dirs for flag in ("-I", str(include_dir))],
        "-c",
        str(source),
        "-o",
        str(object_path),
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
        env=env,
    )
    return CompileResult(
        ok=completed.returncode == 0 and object_path.exists(),
        command=command,
        returncode=completed.returncode,
        object_path=object_path,
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
    )


def _default_nvcc(env: dict[str, str]) -> str:
    cuda_home = env.get("CUDA_HOME")
    if cuda_home:
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.exists():
            return str(candidate)
    return "nvcc"


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:]
