from pathlib import Path
from types import SimpleNamespace

from avo.cli import _agent_status, _baseline_build_env, _score


def test_agent_status_reports_missing_key_without_secret(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    status = _agent_status(None)

    assert status["anthropic_api_key_present"] is False
    assert status["env_file"] is None
    assert status["env_file_loaded"] is False
    assert "ANTHROPIC_API_KEY" not in repr(status)


def test_agent_status_loads_env_file_without_printing_value(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text("ANTHROPIC_API_KEY=test-secret\n", encoding="utf-8")

    status = _agent_status(env_file)

    assert status["anthropic_api_key_present"] is True
    assert status["env_file"] == str(env_file)
    assert status["env_file_loaded"] is True
    assert "test-secret" not in repr(status)


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
