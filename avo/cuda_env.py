from __future__ import annotations

import os
import re
import shutil
import subprocess
import sysconfig
from collections.abc import MutableMapping
from pathlib import Path


def baseline_build_env(env: dict[str, str]) -> dict[str, str]:
    # Build FlashAttention-2 for Ampere family members only on this system.
    # The upstream setup script does not expose a 86-specific token; `80` maps to
    # the Ampere family path used by the project for sm80+ targets.
    env["FLASH_ATTN_CUDA_ARCHS"] = "80"
    # This A6000 pod has limited host RAM. Keep CUDA compilation conservative by
    # default while still allowing an explicit caller override.
    env.setdefault("MAX_JOBS", "1")
    env.setdefault("NVCC_THREADS", "1")
    python_home = compatible_python_cuda_home(env)
    if python_home is not None:
        selected_python_home = env.get("CUDA_HOME") == python_home or env.get(
            "CUDA_PATH"
        ) == python_home
        if selected_python_home or not cuda_env_is_build_compatible(env):
            apply_python_cuda_home(env, python_home)
    return env


def prepare_torch_extension_env(
    env: MutableMapping[str, str] | None = None,
    *,
    max_jobs: str = "2",
) -> MutableMapping[str, str]:
    target = os.environ if env is None else env
    target.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    target.setdefault("MAX_JOBS", max_jobs)
    python_home = compatible_python_cuda_home(dict(target))
    if python_home is not None:
        selected_python_home = target.get("CUDA_HOME") == python_home or target.get(
            "CUDA_PATH"
        ) == python_home
        if selected_python_home or not cuda_env_is_build_compatible(dict(target)):
            apply_python_cuda_home(target, python_home)
    return target


def apply_python_cuda_home(env: MutableMapping[str, str], cuda_home: str) -> None:
    cuda_root = Path(cuda_home)
    env["CUDA_HOME"] = cuda_home
    env["CUDA_PATH"] = cuda_home
    env["CUDACXX"] = str(cuda_root / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc"))
    _prepend_path(env, "PATH", cuda_root / "bin", cuda_home)
    library_dirs = _cuda_library_dirs(cuda_root)
    _prepend_paths(env, "LIBRARY_PATH", library_dirs, cuda_home)
    _prepend_paths(env, "LD_LIBRARY_PATH", library_dirs, cuda_home)
    for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        _prepend_path(env, key, cuda_root / "include", cuda_home)


def nvcc_path_from_env(env: dict[str, str]) -> str | None:
    cuda_home = env.get("CUDA_HOME") or env.get("CUDA_PATH")
    if cuda_home:
        executable = "nvcc.exe" if os.name == "nt" else "nvcc"
        return str(Path(cuda_home) / "bin" / executable)
    return shutil.which("nvcc", path=env.get("PATH"))


def python_cuda_home() -> str | None:
    try:
        purelib = Path(sysconfig.get_paths()["purelib"])
    except Exception:
        return None
    candidates = []
    for nvcc in purelib.glob("nvidia/cu*/bin/nvcc"):
        cuda_home = nvcc.parent.parent
        if (cuda_home / "include" / "cuda.h").exists():
            candidates.append(cuda_home)
    if len(candidates) != 1:
        return None
    return str(candidates[0])


def compatible_python_cuda_home(env: dict[str, str]) -> str | None:
    cuda_home = python_cuda_home()
    if cuda_home is None:
        return None
    nvcc_path = str(Path(cuda_home) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc"))
    nvcc_cuda, _ = nvcc_cuda_version(nvcc_path, env)
    compatibility, _ = cuda_build_compatibility(torch_cuda_version(), nvcc_cuda)
    if compatibility not in {"exact", "minor_mismatch"}:
        return None
    return cuda_home


def cuda_env_is_build_compatible(env: dict[str, str]) -> bool:
    nvcc_path = nvcc_path_from_env(env)
    nvcc_cuda, _ = nvcc_cuda_version(nvcc_path, env)
    compatibility, _ = cuda_build_compatibility(torch_cuda_version(), nvcc_cuda)
    return compatibility in {"exact", "minor_mismatch"}


def torch_cuda_version() -> str | None:
    try:
        import torch
    except Exception:
        return None
    return torch.version.cuda


def nvcc_cuda_version(
    nvcc_path: str | None,
    env: dict[str, str],
) -> tuple[str | None, str | None]:
    if nvcc_path is None:
        return None, "nvcc was not found in CUDA_HOME, CUDA_PATH, or PATH"
    try:
        completed = subprocess.run(
            [nvcc_path, "--version"],
            check=True,
            capture_output=True,
            env=env,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return None, f"nvcc was not found at {nvcc_path}"
    except subprocess.TimeoutExpired:
        return None, f"{nvcc_path} --version timed out"
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or exc.stderr or "").strip()
        suffix = f": {output}" if output else ""
        return None, f"{nvcc_path} --version failed{suffix}"
    output = f"{completed.stdout}\n{completed.stderr}"
    version = parse_nvcc_release(output)
    if version is None:
        return None, f"could not parse CUDA release from {nvcc_path} --version"
    return version, None


def parse_nvcc_release(output: str) -> str | None:
    match = re.search(r"release\s+(\d+\.\d+)", output)
    if match is None:
        return None
    return match.group(1)


def cuda_build_compatibility(
    torch_cuda: str | None,
    nvcc_cuda: str | None,
) -> tuple[str, str | None]:
    if torch_cuda is None:
        return "missing_torch_cuda", "torch.version.cuda is unavailable"
    if nvcc_cuda is None:
        return "missing_nvcc", "nvcc CUDA version is unavailable"
    torch_version = _cuda_major_minor(torch_cuda)
    nvcc_version = _cuda_major_minor(nvcc_cuda)
    if torch_version is None:
        return "unparseable_torch_cuda", f"could not parse torch CUDA version {torch_cuda!r}"
    if nvcc_version is None:
        return "unparseable_nvcc_cuda", f"could not parse nvcc CUDA version {nvcc_cuda!r}"
    if torch_version == nvcc_version:
        return "exact", None
    if torch_version[0] == nvcc_version[0]:
        return (
            "minor_mismatch",
            "PyTorch extension builds may warn: "
            f"nvcc reports CUDA {nvcc_cuda} but torch was compiled with CUDA {torch_cuda}",
        )
    return (
        "major_mismatch",
        "PyTorch extension builds will fail: "
        f"nvcc reports CUDA {nvcc_cuda} but torch was compiled with CUDA {torch_cuda}",
    )


def _cuda_major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"^\s*(\d+)\.(\d+)", version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _prepend_path(
    env: MutableMapping[str, str],
    key: str,
    entry: Path,
    selected_cuda_home: str,
) -> None:
    _prepend_paths(env, key, [entry], selected_cuda_home)


def _prepend_paths(
    env: MutableMapping[str, str],
    key: str,
    entries: list[Path],
    selected_cuda_home: str,
) -> None:
    entry_texts = [str(entry) for entry in entries]
    old_entries = env.get(key, "").split(os.pathsep)
    kept = [
        old_entry
        for old_entry in old_entries
        if old_entry
        and old_entry not in entry_texts
        and not _is_conflicting_cuda_path(old_entry, selected_cuda_home)
    ]
    env[key] = os.pathsep.join([*entry_texts, *kept])


def _is_conflicting_cuda_path(entry: str, selected_cuda_home: str) -> bool:
    entry_path = Path(entry).expanduser()
    selected_path = Path(selected_cuda_home).expanduser()
    try:
        if entry_path.is_relative_to(selected_path):
            return True
    except ValueError:
        pass
    normalized = entry_path.as_posix()
    return normalized == "/usr/local/cuda" or normalized.startswith("/usr/local/cuda-")


def _cuda_library_dirs(cuda_root: Path) -> list[Path]:
    lib_dir = cuda_root / "lib"
    if (lib_dir / "libcudart.so").exists():
        return [lib_dir]
    cudart = _find_versioned_cudart(lib_dir)
    if cudart is None:
        return [lib_dir]
    link_dir = Path(
        os.environ.get(
            "AVO_CUDA_LINK_DIR",
            Path.home() / ".cache" / "avo" / "cuda-links" / cuda_root.name / "lib",
        )
    )
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / "libcudart.so"
    if link.is_symlink():
        if link.resolve() != cudart.resolve():
            link.unlink()
            link.symlink_to(cudart)
    elif not link.exists():
        link.symlink_to(cudart)
    return [link_dir, lib_dir]


def _find_versioned_cudart(lib_dir: Path) -> Path | None:
    candidates = sorted(lib_dir.glob("libcudart.so.*"))
    if not candidates:
        return None
    return candidates[-1]
