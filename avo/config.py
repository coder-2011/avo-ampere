from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MaskMode = Literal["causal", "noncausal"]


@dataclass(frozen=True)
class AttentionCase:
    seq_len: int
    causal: bool
    head_dim: int = 128
    num_heads: int = 16
    total_tokens: int = 32768
    dtype: str = "bf16"

    @property
    def batch_size(self) -> int:
        if self.total_tokens % self.seq_len != 0:
            msg = f"total_tokens={self.total_tokens} must be divisible by seq_len={self.seq_len}"
            raise ValueError(msg)
        return self.total_tokens // self.seq_len

    @property
    def mask_mode(self) -> MaskMode:
        return "causal" if self.causal else "noncausal"

    def as_dict(self) -> dict[str, int | bool | str]:
        return {
            "seq_len": self.seq_len,
            "causal": self.causal,
            "mask_mode": self.mask_mode,
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
            "total_tokens": self.total_tokens,
            "batch_size": self.batch_size,
            "dtype": self.dtype,
        }


DEFAULT_SEQUENCE_LENGTHS = (4096, 8192, 16384, 32768)
DEFAULT_CASES = tuple(
    AttentionCase(seq_len=seq_len, causal=causal)
    for seq_len in DEFAULT_SEQUENCE_LENGTHS
    for causal in (False, True)
)


@dataclass(frozen=True)
class AmpereTarget:
    name: str = "NVIDIA RTX A6000"
    compute_capability: tuple[int, int] = (8, 6)

    @property
    def compute(self) -> str:
        return f"compute_{self.compute_capability[0]}{self.compute_capability[1]}"

    @property
    def sm(self) -> str:
        return f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"

    @property
    def nvcc_gencode(self) -> str:
        return f"-gencode=arch={self.compute},code={self.sm}"


AMPERE_A6000 = AmpereTarget()


def cases_from_cli(
    seq_lens: str,
    causal: str,
    *,
    head_dim: int = 128,
    num_heads: int = 16,
    total_tokens: int = 32768,
    dtype: str = "bf16",
) -> list[AttentionCase]:
    lengths = [int(part) for part in seq_lens.split(",") if part.strip()]
    if not lengths:
        raise ValueError("at least one sequence length is required")
    causal_values = {
        "true": [True],
        "false": [False],
        "both": [False, True],
    }.get(causal)
    if causal_values is None:
        raise ValueError("causal must be one of: true, false, both")
    return [
        AttentionCase(
            seq_len=seq_len,
            causal=is_causal,
            head_dim=head_dim,
            num_heads=num_heads,
            total_tokens=total_tokens,
            dtype=dtype,
        )
        for seq_len in lengths
        for is_causal in causal_values
    ]
