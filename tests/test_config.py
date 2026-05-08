from avo.config import AMPERE_A6000, DEFAULT_CASES, cases_from_cli


def test_ampere_target_is_sm86() -> None:
    assert AMPERE_A6000.compute == "compute_86"
    assert AMPERE_A6000.sm == "sm_86"
    assert AMPERE_A6000.nvcc_gencode == "-gencode=arch=compute_86,code=sm_86"


def test_default_cases_match_architecture_suite() -> None:
    assert len(DEFAULT_CASES) == 8
    assert {case.seq_len for case in DEFAULT_CASES} == {4096, 8192, 16384, 32768}
    assert {case.causal for case in DEFAULT_CASES} == {False, True}
    assert {case.batch_size for case in DEFAULT_CASES} == {8, 4, 2, 1}
    assert {case.head_dim for case in DEFAULT_CASES} == {128}
    assert {case.num_heads for case in DEFAULT_CASES} == {16}
    assert {case.dtype for case in DEFAULT_CASES} == {"bf16"}


def test_cases_from_cli() -> None:
    cases = cases_from_cli("4096,8192", "both")
    assert [(case.seq_len, case.causal) for case in cases] == [
        (4096, False),
        (4096, True),
        (8192, False),
        (8192, True),
    ]


def test_cases_from_cli_accepts_smoke_shape_overrides() -> None:
    cases = cases_from_cli(
        "16",
        "false",
        head_dim=16,
        num_heads=1,
        total_tokens=16,
        dtype="fp32",
    )

    assert len(cases) == 1
    case = cases[0]
    assert case.seq_len == 16
    assert case.batch_size == 1
    assert case.head_dim == 16
    assert case.num_heads == 1
    assert case.total_tokens == 16
    assert case.dtype == "fp32"
