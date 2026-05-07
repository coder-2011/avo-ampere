from avo.cli import _baseline_build_env


def test_baseline_build_env_targets_flash_attn_ampere() -> None:
    env = {"FLASH_ATTN_CUDA_ARCHS": "90;100", "OTHER": "keep-me", "PATH": "/bin"}
    updated = _baseline_build_env(env)

    assert updated["FLASH_ATTN_CUDA_ARCHS"] == "80"
    assert updated["OTHER"] == "keep-me"
    assert updated["PATH"] == "/bin"
