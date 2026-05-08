from __future__ import annotations

import torch.nn.functional as F


def attention(q, k, v, causal: bool):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
