import json
from types import SimpleNamespace

import pytest

from avo.agent import (
    DECISION_TOOL_NAME,
    DEFAULT_AGENT_MODEL,
    VariationDecision,
    _request_decision_response,
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
