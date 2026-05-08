from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch.nn.functional as F
from torch.utils.cpp_extension import load

SOURCE_DIR = Path(__file__).resolve().parent / "cuda_identity"
CUDA_HOME = Path("/usr/local/cuda-12.9")


@lru_cache(maxsize=1)
def _extension():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    os.environ.setdefault("MAX_JOBS", "2")
    if CUDA_HOME.exists():
        os.environ.setdefault("CUDA_HOME", str(CUDA_HOME))
    return load(
        name="avo_cuda_identity_seed",
        sources=[
            str(SOURCE_DIR / "identity.cpp"),
            str(SOURCE_DIR / "identity_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=bool(os.environ.get("AVO_VERBOSE_EXT_BUILD")),
    )


def attention(q, k, v, causal: bool):
    output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
    return _extension().identity(output)
