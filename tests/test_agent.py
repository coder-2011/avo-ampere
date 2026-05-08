import json
from types import SimpleNamespace

import pytest

from avo.agent import (
    DECISION_TOOL_NAME,
    VariationDecision,
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


def test_decision_tool_uses_strict_schema() -> None:
    tool = decision_tool()
    assert tool["name"] == DECISION_TOOL_NAME
    assert tool["strict"] is True
    assert tool["input_schema"]["additionalProperties"] is False


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
