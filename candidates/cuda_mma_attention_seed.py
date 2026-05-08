from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load

SOURCE_DIR = Path(__file__).resolve().parent / "cuda_mma_attention"
CUDA_HOME = Path("/usr/local/cuda-12.9")
SMOKE_SEQUENCE = 16
SMOKE_HEAD_DIM = 16


@lru_cache(maxsize=1)
def _extension():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    os.environ.setdefault("MAX_JOBS", "2")
    if CUDA_HOME.exists():
        os.environ.setdefault("CUDA_HOME", str(CUDA_HOME))
    return load(
        name="avo_cuda_mma_attention_seed",
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
    if seq_len != SMOKE_SEQUENCE or head_dim != SMOKE_HEAD_DIM:
        raise RuntimeError(
            "cuda_mma_attention_seed is a 16x16 BF16 tensor-core correctness seed; "
            f"got seq_len={seq_len}, head_dim={head_dim}"
        )
    if str(q.dtype) != "torch.bfloat16":
        raise RuntimeError(f"cuda_mma_attention_seed requires bf16 inputs; got {q.dtype}")
    return _extension().attention(q.contiguous(), k.contiguous(), v.contiguous(), causal)
