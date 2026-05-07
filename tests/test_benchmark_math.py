from avo.benchmark import attention_forward_flops, geometric_mean, tflops_from_ms
from avo.config import AttentionCase


def test_attention_flops_dense_and_causal() -> None:
    dense = AttentionCase(seq_len=4096, causal=False)
    causal = AttentionCase(seq_len=4096, causal=True)
    assert attention_forward_flops(dense) == 4 * 8 * 16 * 4096 * 4096 * 128
    assert attention_forward_flops(causal) == attention_forward_flops(dense) // 2


def test_tflops_from_ms() -> None:
    case = AttentionCase(seq_len=4096, causal=False)
    assert tflops_from_ms(case, 1.0) == attention_forward_flops(case) / 1e-3 / 1e12


def test_geometric_mean_ignores_zero_failed_scores() -> None:
    assert geometric_mean([2.0, 8.0, 0.0]) == 4.0
    assert geometric_mean([0.0]) == 0.0
