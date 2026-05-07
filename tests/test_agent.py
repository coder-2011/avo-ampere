import json

import pytest

from avo.agent import VariationDecision, parse_decision_text


def test_parse_variation_decision() -> None:
    payload = {
        "hypothesis": "cp.async staging may hide K/V latency",
        "files_to_inspect": ["kernel.cu"],
        "candidate_edit": "inspect pipeline depth",
        "expected_effect": "better long-sequence throughput",
        "risk": "higher shared memory pressure",
        "next_command": "avo score --backend flash-attn",
    }
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
