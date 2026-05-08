from __future__ import annotations

import json
import os
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

DECISION_TOOL_NAME = "record_variation_decision"
DEFAULT_AGENT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_AGENT_REQUEST_ATTEMPTS = 3
DEFAULT_AGENT_RETRY_DELAY_S = 1.0
ALLOWED_NEXT_COMMANDS = frozenset({"env", "compile", "score"})
SHELL_CONTROL_TOKENS = frozenset({"&&", "||", ";", "|", ">", ">>", "<", "`"})
PATCH_REQUIRED_EDIT_VERBS = frozenset(
    {
        "add",
        "adjust",
        "alter",
        "change",
        "edit",
        "extend",
        "fix",
        "implement",
        "modify",
        "patch",
        "refactor",
        "remove",
        "replace",
        "rewrite",
        "update",
    }
)
NO_EDIT_PHRASES = (
    "no edit",
    "no code edit",
    "no patch",
    "without a patch",
    "without editing",
    "without any edit",
)

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis": {
            "type": "string",
            "description": "Ampere-specific performance hypothesis to investigate next.",
        },
        "files_to_inspect": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Repo-local files relevant to the hypothesis.",
        },
        "candidate_edit": {
            "type": "string",
            "description": "Small proposed change or inspection step before editing.",
        },
        "candidate_patch": {
            "type": "string",
            "description": (
                "Raw unified diff for one small candidate edit under candidates/, or empty string "
                "when the next step is inspection/scoring only. Do not use markdown fences."
            ),
        },
        "expected_effect": {
            "type": "string",
            "description": "Expected correctness or throughput effect if the hypothesis is right.",
        },
        "risk": {
            "type": "string",
            "description": "Main correctness, benchmark, or hardware risk.",
        },
        "next_command": {
            "type": "string",
            "description": (
                "One bounded command for the orchestrator. Must start with 'avo' and use only "
                "one of these subcommands: env, compile, score. Do not include shell pipes, "
                "redirection, command chaining, git, rm, cat, head, or arbitrary shell."
            ),
        },
    },
    "required": [
        "hypothesis",
        "files_to_inspect",
        "candidate_edit",
        "candidate_patch",
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
    candidate_patch: str = ""

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> VariationDecision:
        normalized_payload = {"candidate_patch": "", **payload}
        missing = [key for key in DECISION_SCHEMA["required"] if key not in normalized_payload]
        if missing:
            raise ValueError(f"variation decision missing required keys: {', '.join(missing)}")
        files = normalized_payload["files_to_inspect"]
        if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
            raise ValueError("files_to_inspect must be a list of strings")
        candidate_edit = _require_string(normalized_payload, "candidate_edit")
        candidate_patch = _validate_candidate_patch(normalized_payload, "candidate_patch")
        _validate_candidate_edit_matches_patch(candidate_edit, candidate_patch)
        return cls(
            hypothesis=_require_string(normalized_payload, "hypothesis"),
            files_to_inspect=files,
            candidate_edit=candidate_edit,
            candidate_patch=candidate_patch,
            expected_effect=_require_string(normalized_payload, "expected_effect"),
            risk=_require_string(normalized_payload, "risk"),
            next_command=_validate_next_command(
                _require_string(normalized_payload, "next_command")
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "files_to_inspect": self.files_to_inspect,
            "candidate_edit": self.candidate_edit,
            "candidate_patch": self.candidate_patch,
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
    attempt_history: str = "",
    repo_context: str = "",
    model: str = DEFAULT_AGENT_MODEL,
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
    prompt = build_variation_prompt(
        knowledge=knowledge,
        lineage_summary=lineage_summary,
        attempt_history=attempt_history,
        repo_context=repo_context,
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}],
    }
    return _request_valid_decision(client, kwargs)


def build_variation_prompt(
    *,
    knowledge: str,
    lineage_summary: str,
    attempt_history: str = "",
    repo_context: str = "",
) -> str:
    context_section = f"\n\nLocal repo context:\n{repo_context}" if repo_context.strip() else ""
    attempt_section = (
        "\n\nRecent attempt history:\n"
        f"{attempt_history}\n"
        "Use this to avoid repeating failed or regressed directions."
        if attempt_history.strip()
        else ""
    )
    return (
        "You are the AVO variation operator for an Ampere sm_86 attention kernel.\n"
        "Use FlashAttention-2/Ampere assumptions only. FA4/Blackwell strategies are invalid.\n"
        "Use candidate_patch for exactly one small repo-local unified diff under candidates/. "
        "If no edit is needed, candidate_patch must be exactly the empty string \"\". Do not "
        "include markdown fences or commentary in candidate_patch. If candidate_edit describes "
        "a code change such as extending, updating, modifying, or fixing a candidate, "
        "candidate_patch must contain the raw diff for that change.\n"
        "Return exactly one decision. The next_command must be a single bounded command that "
        "starts with 'avo' and uses only one of: env, compile, score. Use valid CLI flags: "
        "compile requires --source SOURCE.cu and --out-dir DIR; candidate score requires "
        "--backend candidate and --candidate. Do not use shell pipes, redirection, command "
        "chaining, cat, head, git, rm, or arbitrary shell commands.\n\n"
        f"Knowledge:\n{knowledge}\n\nLineage:\n{lineage_summary}"
        f"{attempt_section}{context_section}\n"
    )


def build_repo_context(root: Path) -> str:
    candidates = _relative_files(root, "candidates", suffix=".py")
    cuda_sources = sorted(
        [
            *(_relative_files(root, "candidates", suffix=".cu")),
            *(_relative_files(root, "candidates", suffix=".cpp")),
        ]
    )
    lines = [
        "Use only files that exist in this repository.",
        "Do not propose upstream FlashAttention csrc paths unless they are present locally.",
        "Available bounded commands: avo env; avo compile --source SOURCE.cu --out-dir DIR; "
        "avo score --backend BACKEND ...",
        "Available edit channel: candidate_patch as a raw unified diff under candidates/, "
        "or empty.",
        "Candidate interface: module defines attention(q, k, v, causal: bool).",
    ]
    if candidates:
        lines.append("Candidate modules:")
        lines.extend(f"- {candidate}" for candidate in candidates)
    if cuda_sources:
        lines.append("CUDA candidate sources:")
        lines.extend(f"- {source}" for source in cuda_sources)
    preferred_command = _preferred_candidate_score_command(candidates)
    if preferred_command:
        lines.append(f"Preferred first local candidate score command: {preferred_command}")
    return "\n".join(lines)


def _request_decision_response(
    client: Any,
    kwargs: dict[str, Any],
    *,
    attempts: int = DEFAULT_AGENT_REQUEST_ATTEMPTS,
    retry_delay_s: float = DEFAULT_AGENT_RETRY_DELAY_S,
) -> Any:
    for attempt in range(attempts):
        try:
            return _request_decision_response_once(client, kwargs)
        except Exception as exc:
            is_last_attempt = attempt + 1 >= attempts
            if is_last_attempt or not _transient_api_error(exc):
                raise
            time.sleep(retry_delay_s * (2**attempt))
    raise AssertionError("unreachable")


def _request_valid_decision(
    client: Any,
    kwargs: dict[str, Any],
    *,
    attempts: int = DEFAULT_AGENT_REQUEST_ATTEMPTS,
    retry_delay_s: float = DEFAULT_AGENT_RETRY_DELAY_S,
) -> VariationDecision:
    last_error: ValueError | None = None
    for attempt in range(attempts):
        request_kwargs = _decision_kwargs_with_feedback(kwargs, last_error)
        response = _request_decision_response(client, request_kwargs)
        try:
            return parse_decision_response(response)
        except ValueError as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                raise ValueError(
                    f"agent returned invalid variation decision after {attempts} attempts: {exc}"
                ) from exc
            time.sleep(retry_delay_s * (2**attempt))
    raise AssertionError("unreachable")


def _decision_kwargs_with_feedback(
    kwargs: dict[str, Any],
    last_error: ValueError | None,
) -> dict[str, Any]:
    if last_error is None:
        return kwargs
    updated = dict(kwargs)
    messages = [dict(message) for message in kwargs["messages"]]
    messages[-1]["content"] = (
        f"{messages[-1]['content']}\n\n"
        "The previous decision was invalid and was not executed. "
        f"Validation error: {last_error}. Return one corrected decision."
    )
    updated["messages"] = messages
    return updated


def _request_decision_response_once(client: Any, kwargs: dict[str, Any]) -> Any:
    # Prefer strict tool use because it schema-constrains the decision boundary.
    # Keep JSON output fallback for SDK/model combinations that do not support it.
    try:
        return client.messages.create(
            **kwargs,
            tools=[decision_tool()],
            tool_choice={
                "type": "tool",
                "name": DECISION_TOOL_NAME,
                "disable_parallel_tool_use": True,
            },
        )
    except TypeError:
        return _request_decision_json_response(client, kwargs)
    except Exception as exc:
        if not _strict_tools_unsupported(exc):
            raise
        return _request_decision_json_response(client, kwargs)


def _request_decision_json_response(client: Any, kwargs: dict[str, Any]) -> Any:
    try:
        return client.messages.create(
            **kwargs,
            output_config={"format": {"type": "json_schema", "schema": DECISION_SCHEMA}},
        )
    except TypeError:
        return client.messages.create(**kwargs)
    except Exception as exc:
        if not _structured_outputs_unsupported(exc):
            raise
        return client.messages.create(**kwargs)


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _validate_candidate_patch(payload: dict[str, Any], key: str) -> str:
    value = _optional_string(payload, key)
    if not value.strip():
        return ""
    has_diff = any(line.startswith("diff --git ") for line in value.splitlines())
    if not has_diff:
        return ""
    if "```" in value:
        raise ValueError("candidate_patch must be raw diff text, not markdown")
    return value


def _validate_candidate_edit_matches_patch(candidate_edit: str, candidate_patch: str) -> None:
    if candidate_patch.strip() or not _candidate_edit_requires_patch(candidate_edit):
        return
    raise ValueError(
        "candidate_patch must be non-empty when candidate_edit describes a code change"
    )


def _candidate_edit_requires_patch(candidate_edit: str) -> bool:
    normalized = " ".join(candidate_edit.lower().replace("-", " ").split())
    if any(phrase in normalized for phrase in NO_EDIT_PHRASES):
        return False
    words = re.findall(r"[a-z]+", normalized)
    return any(word in PATCH_REQUIRED_EDIT_VERBS for word in words)


def _validate_next_command(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"next_command is not parseable: {exc}") from exc
    if len(parts) < 2 or parts[0] != "avo":
        raise ValueError("next_command must start with 'avo'")
    if any(part in SHELL_CONTROL_TOKENS for part in parts):
        raise ValueError("next_command must not contain shell control tokens")
    subcommand = parts[1]
    if subcommand not in ALLOWED_NEXT_COMMANDS:
        allowed = ", ".join(sorted(ALLOWED_NEXT_COMMANDS))
        raise ValueError(
            f"next_command uses unsupported avo subcommand '{subcommand}'; allowed: {allowed}"
        )
    _validate_subcommand_arguments(parts)
    return command


def _validate_subcommand_arguments(parts: list[str]) -> None:
    subcommand = parts[1]
    if subcommand == "compile":
        if _has_option(parts, "--candidate"):
            raise ValueError(
                "next_command compile does not support --candidate; use --source and --out-dir"
            )
        missing = [
            option
            for option in ("--source", "--out-dir")
            if _single_option_value(parts, option) is None
        ]
        if missing:
            raise ValueError(f"next_command compile requires {', '.join(missing)}")
        source = _single_option_value(parts, "--source")
        out_dir = _single_option_value(parts, "--out-dir")
        if source is not None:
            _validate_command_path(
                source,
                "--source",
                allowed_roots=("candidates/",),
                suffixes=(".cu",),
            )
        if out_dir is not None:
            _validate_command_path(out_dir, "--out-dir")
    elif subcommand == "score":
        backend = _single_option_value(parts, "--backend")
        if backend is None:
            raise ValueError("next_command score requires --backend")
        if backend not in {"torch-sdpa", "flash-attn", "candidate"}:
            raise ValueError(f"next_command score uses unsupported backend '{backend}'")
        candidate = _single_option_value(parts, "--candidate")
        if backend == "candidate":
            if candidate is None:
                raise ValueError("next_command candidate score requires --candidate")
            _validate_command_path(
                candidate,
                "--candidate",
                allowed_roots=("candidates/",),
                suffixes=(".py",),
            )


def _has_option(parts: list[str], option: str) -> bool:
    return _single_option_value(parts, option, allow_missing_value=True) is not None


def _single_option_value(
    parts: list[str],
    option: str,
    *,
    allow_missing_value: bool = False,
) -> str | None:
    values = _option_values(parts, option, allow_missing_value=allow_missing_value)
    if len(values) > 1:
        raise ValueError(f"next_command repeats {option}")
    return values[0] if values else None


def _option_values(
    parts: list[str],
    option: str,
    *,
    allow_missing_value: bool = False,
) -> list[str]:
    values: list[str] = []
    prefix = f"{option}="
    for index, part in enumerate(parts):
        if part == option:
            if index + 1 >= len(parts) or parts[index + 1].startswith("--"):
                if allow_missing_value:
                    values.append("")
                    continue
                raise ValueError(f"next_command {option} requires a value")
            values.append(parts[index + 1])
        elif part.startswith(prefix):
            value = part[len(prefix) :]
            if not value and not allow_missing_value:
                raise ValueError(f"next_command {option} requires a value")
            values.append(value)
    return values


def _validate_command_path(
    raw_path: str,
    option: str,
    *,
    allowed_roots: tuple[str, ...] | None = None,
    suffixes: tuple[str, ...] | None = None,
) -> None:
    if "\x00" in raw_path or "\\" in raw_path:
        raise ValueError(f"next_command {option} path contains unsupported characters")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"next_command {option} must be a repo-relative path")
    normalized = path.as_posix()
    if allowed_roots is not None and not any(normalized.startswith(root) for root in allowed_roots):
        roots = ", ".join(root.rstrip("/") for root in allowed_roots)
        raise ValueError(f"next_command {option} must be under: {roots}")
    if suffixes is not None and path.suffix not in suffixes:
        allowed = ", ".join(suffixes)
        raise ValueError(f"next_command {option} must reference a {allowed} file")


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


def _relative_files(root: Path, dirname: str, *, suffix: str) -> list[str]:
    base = root / dirname
    if not base.exists():
        return []
    return sorted(
        str(path.relative_to(root))
        for path in base.rglob(f"*{suffix}")
        if "__pycache__" not in path.parts
    )


def _preferred_candidate_score_command(candidates: list[str]) -> str:
    if "candidates/cuda_mma_attention_seed.py" in candidates:
        return (
            "avo score --backend candidate "
            "--candidate candidates/cuda_mma_attention_seed.py "
            "--seq-lens 32 --total-tokens 32 --num-heads 1 --head-dim 16 "
            "--dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300"
        )
    if "candidates/cuda_warp_rows_attention_seed.py" in candidates:
        return (
            "avo score --backend candidate "
            "--candidate candidates/cuda_warp_rows_attention_seed.py "
            "--seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 "
            "--dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300"
        )
    if "candidates/cuda_tiled_attention_seed.py" in candidates:
        return (
            "avo score --backend candidate "
            "--candidate candidates/cuda_tiled_attention_seed.py "
            "--seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 "
            "--dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300"
        )
    if "candidates/cuda_naive_attention_seed.py" in candidates:
        return (
            "avo score --backend candidate "
            "--candidate candidates/cuda_naive_attention_seed.py "
            "--seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 "
            "--dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300"
        )
    if "candidates/cuda_identity_seed.py" in candidates:
        return (
            "avo score --backend candidate --candidate candidates/cuda_identity_seed.py "
            "--seq-lens 4096 --causal false --repeats 1 --warmup 1 --timeout-s 300"
        )
    if "candidates/torch_sdpa_seed.py" in candidates:
        return (
            "avo score --backend candidate --candidate candidates/torch_sdpa_seed.py "
            "--seq-lens 4096 --causal both --repeats 1 --warmup 1 --timeout-s 300"
        )
    return ""


def _strict_tools_unsupported(exc: Exception) -> bool:
    text = str(exc).lower()
    return "strict" in text and "tool" in text and _unsupported(text)


def _structured_outputs_unsupported(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("output_config" in text or "structured output" in text) and _unsupported(text)


def _transient_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504, 529}:
        return True
    class_name = type(exc).__name__
    return class_name in {"APIConnectionError", "APITimeoutError"}


def _unsupported(text: str) -> bool:
    return "does not support" in text or "unsupported" in text or "not supported" in text


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
