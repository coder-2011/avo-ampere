from pathlib import Path
from types import SimpleNamespace

from avo.cli import _baseline_build_env, _score


def test_baseline_build_env_targets_flash_attn_ampere() -> None:
    env = {"FLASH_ATTN_CUDA_ARCHS": "90;100", "OTHER": "keep-me", "PATH": "/bin"}
    updated = _baseline_build_env(env)

    assert updated["FLASH_ATTN_CUDA_ARCHS"] == "80"
    assert updated["OTHER"] == "keep-me"
    assert updated["PATH"] == "/bin"


def test_score_command_forwards_trial_count(monkeypatch) -> None:
    captured = {}

    class FakeResult:
        ok = True

        def as_dict(self):
            return {"ok": True, "payload": {"all_correct": True}}

    def fake_run_json_worker(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeResult()

    monkeypatch.setattr("avo.cli.run_json_worker", fake_run_json_worker)

    exit_code = _score(
        SimpleNamespace(
            backend="candidate",
            candidate=Path("candidates/cuda_warp_rows_attention_seed.py"),
            seq_lens="16",
            causal="both",
            head_dim=16,
            num_heads=1,
            total_tokens=16,
            dtype="bf16",
            warmup=1,
            repeats=2,
            trials=3,
            timeout_s=300,
        )
    )

    assert exit_code == 0
    assert "--trials" in captured["args"]
    assert captured["args"][captured["args"].index("--trials") + 1] == "3"
