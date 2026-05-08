from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from avo.cuda_env import prepare_torch_extension_env

prepare_torch_extension_env(os.environ, max_jobs="2")

SOURCE_DIR = Path(__file__).resolve().parent / "cuda_mma_attention"
SMOKE_SEQUENCES = {16, 32, 64, 128, 256, 1024, 2048, 4096, 8192}
SMOKE_HEAD_DIM = 128


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

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
    if seq_len not in SMOKE_SEQUENCES or head_dim != SMOKE_HEAD_DIM:
        raise RuntimeError(
            "cuda_mma_attention_seed is a 16/32/64/128/256/1024/2048/4096/8192-token, "
            "head_dim128 BF16 tensor-core seed; "
            f"got seq_len={seq_len}, head_dim={head_dim}"
        )
    if str(q.dtype) != "torch.bfloat16":
        raise RuntimeError(f"cuda_mma_attention_seed requires bf16 inputs; got {q.dtype}")
    return _extension().attention(q.contiguous(), k.contiguous(), v.contiguous(), causal)
