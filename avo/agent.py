from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DECISION_TOOL_NAME = "record_variation_decision"

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "files_to_inspect": {"type": "array", "items": {"type": "string"}},
        "candidate_edit": {"type": "string"},
        "expected_effect": {"type": "string"},
        "risk": {"type": "string"},
        "next_command": {"type": "string"},
    },
    "required": [
        "hypothesis",
        "files_to_inspect",
        "candidate_edit",
        "expected_effect",
        "risk",
        "next_command",
    ],
    "additionalProperties": False,
}


def decision_tool() -> dict[str, Any]:
    return {
        "name": DECISION_TOOL_NAME,
        "description": (
            "Record exactly one proposed Ampere AVO variation step. "
            "This is a planning tool only; the orchestrator validates and executes commands."
        ),
        "strict": True,
        "input_schema": DECISION_SCHEMA,
    }


@dataclass(frozen=True)
class VariationDecision:
    hypothesis: str
    files_to_inspect: list[str]
    candidate_edit: str
    expected_effect: str
    risk: str
    next_command: str

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> VariationDecision:
        missing = [key for key in DECISION_SCHEMA["required"] if key not in payload]
        if missing:
            raise ValueError(f"variation decision missing required keys: {', '.join(missing)}")
        files = payload["files_to_inspect"]
        if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
            raise ValueError("files_to_inspect must be a list of strings")
        return cls(
            hypothesis=_require_string(payload, "hypothesis"),
            files_to_inspect=files,
            candidate_edit=_require_string(payload, "candidate_edit"),
            expected_effect=_require_string(payload, "expected_effect"),
            risk=_require_string(payload, "risk"),
            next_command=_require_string(payload, "next_command"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "files_to_inspect": self.files_to_inspect,
            "candidate_edit": self.candidate_edit,
            "expected_effect": self.expected_effect,
            "risk": self.risk,
            "next_command": self.next_command,
        }


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key, value)


def parse_decision_text(text: str) -> VariationDecision:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _recover_json_object(text)
        if payload is None:
            raise ValueError(f"agent returned malformed JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("agent JSON response must be an object")
    return VariationDecision.from_mapping(payload)


def parse_decision_response(response: Any) -> VariationDecision:
    for block in getattr(response, "content", []):
        if _block_type(block) != "tool_use":
            continue
        if _block_value(block, "name") != DECISION_TOOL_NAME:
            continue
        payload = _block_value(block, "input")
        if not isinstance(payload, dict):
            raise ValueError("variation decision tool input must be an object")
        return VariationDecision.from_mapping(payload)

    text = _response_text(response)
    if text.strip():
        return parse_decision_text(text)
    raise ValueError(f"agent did not call {DECISION_TOOL_NAME}")


def request_variation_decision(
    *,
    lineage_summary: str,
    knowledge: str,
    model: str = "claude-sonnet-4-20250514",
) -> VariationDecision:
    try:
        import anthropic
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "install the agent extra to use Anthropic: pip install '.[agent]'"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for agent planning")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "You are the AVO variation operator for an Ampere sm_86 attention kernel.\n"
        "Use FlashAttention-2/Ampere assumptions only. FA4/Blackwell strategies are invalid.\n"
        "Return exactly one JSON object matching the requested schema.\n\n"
        f"Knowledge:\n{knowledge}\n\nLineage:\n{lineage_summary}\n"
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 1200,
        "messages": [{"role": "user", "content": prompt}],
    }
    # Prefer strict tool use because it schema-constrains the decision boundary.
    # Keep JSON output fallback for SDK/model combinations that do not support it.
    try:
        response = client.messages.create(
            **kwargs,
            tools=[decision_tool()],
            tool_choice={
                "type": "tool",
                "name": DECISION_TOOL_NAME,
                "disable_parallel_tool_use": True,
            },
        )
    except TypeError:
        try:
            response = client.messages.create(
                **kwargs,
                output_config={"format": {"type": "json_schema", "schema": DECISION_SCHEMA}},
            )
        except TypeError:
            response = client.messages.create(**kwargs)

    return parse_decision_response(response)


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _recover_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _response_text(response: Any) -> str:
    return "\n".join(
        str(_block_value(block, "text"))
        for block in getattr(response, "content", [])
        if _block_type(block) == "text" and _block_value(block, "text") is not None
    )


def _block_type(block: Any) -> str | None:
    return _block_value(block, "type")


def _block_value(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)
