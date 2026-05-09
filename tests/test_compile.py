from pathlib import Path

from avo.compile import compile_cuda_source


def test_compile_uses_sm86_gencode_with_fake_nvcc(tmp_path: Path) -> None:
    source = tmp_path / "kernel.cu"
    source.write_text("extern \"C\" __global__ void k() {}\n", encoding="utf-8")
    log = tmp_path / "args.txt"
    fake_nvcc = tmp_path / "nvcc"
    fake_nvcc.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$FAKE_NVCC_LOG\"\n"
        "while [[ $# -gt 0 ]]; do\n"
        "  if [[ \"$1\" == '-o' ]]; then shift; touch \"$1\"; exit 0; fi\n"
        "  shift\n"
        "done\n",
        encoding="utf-8",
    )
    fake_nvcc.chmod(0o755)

    result = compile_cuda_source(
        source,
        tmp_path / "build",
        nvcc=str(fake_nvcc),
        env={"FAKE_NVCC_LOG": str(log)},
    )

    assert result.ok
    args = log.read_text(encoding="utf-8").splitlines()
    assert "-gencode=arch=compute_86,code=sm_86" in args
    assert "--expt-relaxed-constexpr" in args
    assert "--ptxas-options=-v" in args
    assert "-D__CUDA_NO_HALF_OPERATORS__" in args
    assert "-D__CUDA_NO_HALF_CONVERSIONS__" in args
    assert "-D__CUDA_NO_BFLOAT16_CONVERSIONS__" in args
    assert "-D__CUDA_NO_HALF2_OPERATORS__" in args
    assert "-c" in args


def test_compile_forwards_include_dirs_to_nvcc(tmp_path: Path) -> None:
    source = tmp_path / "kernel.cu"
    source.write_text("extern \"C\" __global__ void k() {}\n", encoding="utf-8")
    include_dir = tmp_path / "include"
    include_dir.mkdir()
    log = tmp_path / "args.txt"
    fake_nvcc = tmp_path / "nvcc"
    fake_nvcc.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$FAKE_NVCC_LOG\"\n"
        "while [[ $# -gt 0 ]]; do\n"
        "  if [[ \"$1\" == '-o' ]]; then shift; touch \"$1\"; exit 0; fi\n"
        "  shift\n"
        "done\n",
        encoding="utf-8",
    )
    fake_nvcc.chmod(0o755)

    result = compile_cuda_source(
        source,
        tmp_path / "build",
        nvcc=str(fake_nvcc),
        env={"FAKE_NVCC_LOG": str(log)},
        include_dirs=[include_dir],
    )

    assert result.ok
    args = log.read_text(encoding="utf-8").splitlines()
    assert "-I" in args
    assert str(include_dir) in args
