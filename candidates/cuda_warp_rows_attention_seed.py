from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load

SOURCE_DIR = Path(__file__).resolve().parent / "cuda_warp_rows_attention"
CUDA_HOME = Path("/usr/local/cuda-12.9")
MAX_SMOKE_SEQUENCE = 128
MAX_SMOKE_HEAD_DIM = 128


@lru_cache(maxsize=1)
def _extension():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    os.environ.setdefault("MAX_JOBS", "2")
    if CUDA_HOME.exists():
        os.environ.setdefault("CUDA_HOME", str(CUDA_HOME))
    return load(
        name="avo_cuda_warp_rows_attention_seed",
        sources=[
            str(SOURCE_DIR / "attention.cpp"),
            str(SOURCE_DIR / "attention_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=bool(os.environ.get("AVO_VERBOSE_EXT_BUILD")),
    )


def attention(q, k, v, causal: bool):
    seq_len = q.shape[2]
    head_dim = q.shape[3]
    if seq_len > MAX_SMOKE_SEQUENCE or head_dim > MAX_SMOKE_HEAD_DIM:
        raise RuntimeError(
            "cuda_warp_rows_attention_seed is a tiny multi-row correctness seed; "
            f"got seq_len={seq_len}, head_dim={head_dim}"
        )
    return _extension().attention(q.contiguous(), k.contiguous(), v.contiguous(), causal)
