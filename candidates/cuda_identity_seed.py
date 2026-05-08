from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch.nn.functional as F

from avo.cuda_env import prepare_torch_extension_env

prepare_torch_extension_env(os.environ, max_jobs="2")

SOURCE_DIR = Path(__file__).resolve().parent / "cuda_identity"


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

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
