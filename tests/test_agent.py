import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from avo.agent import (
    DECISION_TOOL_NAME,
    DEFAULT_AGENT_MODEL,
    VariationDecision,
    _request_decision_response,
    build_repo_context,
    build_variation_prompt,
    decision_tool,
    parse_decision_response,
    parse_decision_text,
)


def decision_payload() -> dict[str, object]:
    return {
        "hypothesis": "cp.async staging may hide K/V latency",
        "files_to_inspect": ["kernel.cu"],
        "candidate_edit": "inspect pipeline depth",
        "expected_effect": "better long-sequence throughput",
        "risk": "higher shared memory pressure",
        "next_command": "avo score --backend flash-attn",
    }


def test_parse_variation_decision() -> None:
    payload = decision_payload()
    decision = parse_decision_text(json.dumps(payload))
    assert isinstance(decision, VariationDecision)
    assert decision.files_to_inspect == ["kernel.cu"]


def test_parse_variation_decision_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="malformed JSON"):
        parse_decision_text("{not-json")


def test_parse_variation_decision_recovers_embedded_json() -> None:
    payload = {
        "hypothesis": "reduce shared memory bank conflicts",
        "files_to_inspect": ["kernel.cu"],
        "candidate_edit": "inspect smem layout",
        "expected_effect": "lower replay overhead",
        "risk": "layout change may break indexing",
        "next_command": "avo score --backend torch-sdpa",
    }
    decision = parse_decision_text(f"```json\n{json.dumps(payload)}\n```")
    assert decision.hypothesis == payload["hypothesis"]


def test_parse_variation_decision_rejects_missing_key() -> None:
    with pytest.raises(ValueError, match="missing required"):
        parse_decision_text(json.dumps({"hypothesis": "x"}))


def test_parse_variation_decision_rejects_unbounded_next_command() -> None:
    payload = decision_payload()
    payload["next_command"] = "cat kernel.cu | head -20"

    with pytest.raises(ValueError, match="must start with 'avo'"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_shell_next_command() -> None:
    payload = decision_payload()
    payload["next_command"] = "avo env && rm -rf /"

    with pytest.raises(ValueError, match="shell control"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_unsupported_avo_subcommand() -> None:
    payload = decision_payload()
    payload["next_command"] = "avo commit-score lineage score.json"

    with pytest.raises(ValueError, match="unsupported"):
        parse_decision_text(json.dumps(payload))


def test_decision_tool_uses_strict_schema() -> None:
    tool = decision_tool()
    assert tool["name"] == DECISION_TOOL_NAME
    assert tool["strict"] is True
    assert tool["input_schema"]["additionalProperties"] is False


def test_default_agent_model_supports_structured_outputs_family() -> None:
    assert DEFAULT_AGENT_MODEL == "claude-sonnet-4-5-20250929"


def test_build_repo_context_lists_local_candidates() -> None:
    context = build_repo_context(Path.cwd())

    assert "candidates/cuda_identity_seed.py" in context
    assert "candidates/cuda_naive_attention_seed.py" in context
    assert "candidates/cuda_tiled_attention_seed.py" in context
    assert "candidates/cuda_warp_rows_attention_seed.py" in context
    assert "candidates/torch_sdpa_seed.py" in context
    assert "candidates/cuda_identity/identity_kernel.cu" in context
    assert "--candidate candidates/cuda_warp_rows_attention_seed.py" in context
    assert "--seq-lens 16" in context
    assert "avo score --backend candidate" in context
    assert "csrc/flash_attn" not in context


def test_build_repo_context_falls_back_to_tiled_candidate(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    cuda_source = candidates / "cuda_tiled_attention"
    cuda_source.mkdir(parents=True)
    (candidates / "cuda_tiled_attention_seed.py").write_text("", encoding="utf-8")
    (cuda_source / "attention_kernel.cu").write_text("", encoding="utf-8")

    context = build_repo_context(tmp_path)

    assert "--candidate candidates/cuda_tiled_attention_seed.py" in context
    assert "--candidate candidates/cuda_warp_rows_attention_seed.py" not in context


def test_build_repo_context_falls_back_to_naive_candidate(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    cuda_source = candidates / "cuda_naive_attention"
    cuda_source.mkdir(parents=True)
    (candidates / "cuda_naive_attention_seed.py").write_text("", encoding="utf-8")
    (cuda_source / "attention_kernel.cu").write_text("", encoding="utf-8")

    context = build_repo_context(tmp_path)

    assert "--candidate candidates/cuda_naive_attention_seed.py" in context
    assert "--candidate candidates/cuda_tiled_attention_seed.py" not in context


def test_build_repo_context_falls_back_to_identity_candidate(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    cuda_source = candidates / "cuda_identity"
    cuda_source.mkdir(parents=True)
    (candidates / "cuda_identity_seed.py").write_text("", encoding="utf-8")
    (cuda_source / "identity_kernel.cu").write_text("", encoding="utf-8")

    context = build_repo_context(tmp_path)

    assert "--candidate candidates/cuda_identity_seed.py" in context
    assert "--candidate candidates/cuda_naive_attention_seed.py" not in context


def test_build_variation_prompt_includes_repo_context() -> None:
    prompt = build_variation_prompt(
        knowledge="Ampere only.",
        lineage_summary="No accepted candidates yet.",
        repo_context="Candidate modules:\n- candidates/cuda_identity_seed.py",
    )

    assert "Knowledge:\nAmpere only." in prompt
    assert "Lineage:\nNo accepted candidates yet." in prompt
    assert "Local repo context:" in prompt
    assert "candidates/cuda_identity_seed.py" in prompt


def test_build_variation_prompt_includes_attempt_history() -> None:
    prompt = build_variation_prompt(
        knowledge="Ampere only.",
        lineage_summary="No accepted candidates yet.",
        attempt_history="Recent attempts, oldest to newest:\n- command ok; gate rejected",
    )

    assert "Recent attempt history:" in prompt
    assert "gate rejected" in prompt
    assert "avoid repeating failed or regressed directions" in prompt


def test_parse_variation_decision_response_prefers_tool_use() -> None:
    payload = decision_payload()
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text='{"hypothesis": "ignored malformed fallback"'),
            SimpleNamespace(type="tool_use", name=DECISION_TOOL_NAME, input=payload),
        ]
    )

    decision = parse_decision_response(response)

    assert decision.hypothesis == payload["hypothesis"]


def test_parse_variation_decision_response_falls_back_to_text() -> None:
    payload = decision_payload()
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload))])

    decision = parse_decision_response(response)

    assert decision.next_command == payload["next_command"]


def test_parse_variation_decision_response_requires_expected_tool() -> None:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="wrong_tool", input=decision_payload())]
    )

    with pytest.raises(ValueError, match=DECISION_TOOL_NAME):
        parse_decision_response(response)


def test_decision_request_falls_back_when_strict_tools_are_unsupported() -> None:
    class FakeMessages:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("model does not support strict tools")
            text = json.dumps(decision_payload())
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])

    messages = FakeMessages()
    response = _request_decision_response(
        SimpleNamespace(messages=messages),
        {"model": "old-model"},
    )

    assert "tools" in messages.calls[0]
    assert "output_config" in messages.calls[1]
    assert parse_decision_response(response).hypothesis == decision_payload()["hypothesis"]


def test_decision_request_falls_back_to_plain_json_when_structured_outputs_fail() -> None:
    class FakeMessages:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("model does not support strict tools")
            if len(self.calls) == 2:
                raise RuntimeError("model does not support output_config")
            text = json.dumps(decision_payload())
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])

    messages = FakeMessages()
    response = _request_decision_response(
        SimpleNamespace(messages=messages),
        {"model": "old-model"},
    )

    assert "tools" in messages.calls[0]
    assert "output_config" in messages.calls[1]
    assert "tools" not in messages.calls[2]
    assert "output_config" not in messages.calls[2]
    assert parse_decision_response(response).candidate_edit == decision_payload()["candidate_edit"]


def test_decision_request_retries_transient_api_error() -> None:
    class FakeTransientError(RuntimeError):
        status_code = 500

    class FakeMessages:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise FakeTransientError("internal server error")
            text = json.dumps(decision_payload())
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])

    messages = FakeMessages()
    response = _request_decision_response(
        SimpleNamespace(messages=messages),
        {"model": "claude"},
        retry_delay_s=0.0,
    )

    assert len(messages.calls) == 2
    assert "tools" in messages.calls[0]
    assert "tools" in messages.calls[1]
    assert parse_decision_response(response).next_command == decision_payload()["next_command"]
