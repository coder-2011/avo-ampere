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
TOOL_PARAMETER_MARKERS = ("<parameter ", "</parameter>")
DEFAULT_SCORE_HEAD_DIM = 128
DEFAULT_SCORE_NUM_HEADS = 16
DEFAULT_SCORE_SEQ_LENS = (4096, 8192, 16384, 32768)
DEFAULT_SCORE_TOTAL_TOKENS = 32768
MAX_REPO_CONTEXT_FILE_CHARS = 12_000
MAX_REPO_CONTEXT_SOURCE_CHARS = 45_000
WARP_ROWS_SEED = "candidates/cuda_warp_rows_attention_seed.py"
MMA_SEED = "candidates/cuda_mma_attention_seed.py"
TILED_SEED = "candidates/cuda_tiled_attention_seed.py"
RECORDED_NO_PATCH_COMPILE_SOURCES = frozenset(
    {
        "candidates/cuda_mma_attention/attention_kernel.cu",
        "candidates/cuda_tiled_attention/attention_kernel.cu",
        "candidates/cuda_warp_rows_attention/attention_kernel.cu",
    }
)
ENV_COMMAND_KEYWORDS = (
    "baseline",
    "build",
    "compiler",
    "cuda",
    "cuda home",
    "cuda path",
    "env",
    "environment",
    "flash attn",
    "flash-attn",
    "flash attention",
    "install",
    "nvcc",
    "torch",
)
COMPILE_COMMAND_KEYWORDS = (
    "build",
    "compile",
    "compilation",
    "compiler",
    "nvcc",
    "object",
    "ptxas",
    "syntax",
    "translation unit",
)
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
SELF_REJECTING_PATCH_PHRASES = (
    "cannot affect correctness or throughput",
    "cannot improve throughput",
    "not ready to apply",
    "not yet called",
    "stub is empty",
    "unused in this patch",
    "must be updated before scoring",
    "unused doubled buffers",
    "would break correctness",
    "will cause a compile error",
    "will break correctness",
    "will fail correctness",
    "will compute the wrong shared memory address",
    "likely segfault or produce incorrect results",
    "must define koutputelements before using it",
    "reject this direction",
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
            "description": (
                "Small proposed change or inspection step. If candidate_patch is empty, "
                "this must start with 'No edit;' and describe only a bounded diagnostic."
            ),
        },
        "candidate_patch": {
            "type": "string",
            "description": (
                "Raw git-style unified diff for one small candidate edit under candidates/, "
                "starting with 'diff --git', or empty string when the next step is inspection/"
                "scoring only. Use exact file context and whitespace-clean hunks. Do not use "
                "markdown fences."
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
        hypothesis = _require_string(normalized_payload, "hypothesis")
        candidate_edit = _require_string(normalized_payload, "candidate_edit")
        candidate_patch = _validate_candidate_patch(normalized_payload, "candidate_patch")
        expected_effect = _require_string(normalized_payload, "expected_effect")
        risk = _require_string(normalized_payload, "risk")
        _validate_candidate_edit_matches_patch(candidate_edit, candidate_patch)
        planning_text = "\n".join((hypothesis, candidate_edit, expected_effect, risk))
        _validate_candidate_patch_not_self_rejected(candidate_patch, planning_text)
        _validate_candidate_patch_domain_sanity(candidate_patch)
        next_command = _validate_next_command(
            _require_string(normalized_payload, "next_command"),
            candidate_patch=candidate_patch,
            planning_text=planning_text,
        )
        return cls(
            hypothesis=hypothesis,
            files_to_inspect=files,
            candidate_edit=candidate_edit,
            candidate_patch=candidate_patch,
            expected_effect=expected_effect,
            risk=risk,
            next_command=next_command,
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
        "Use one of exactly two edit modes. Edit mode: candidate_patch is one small raw "
        "git-style unified diff under candidates/ starting with 'diff --git', and "
        "candidate_edit summarizes that diff. The diff must apply cleanly with git apply: "
        "use exact current file context, keep hunk structure valid, and avoid trailing "
        "whitespace-only added lines. Prefer a smaller compile-checkable patch when the "
        "full change is uncertain. No-edit mode: candidate_patch is exactly the "
        "empty string \"\", candidate_edit starts with \"No edit; \", and next_command is only "
        "a bounded score, compile, or environment diagnostic for existing files. Do not include "
        "markdown fences or commentary in candidate_patch. Do not describe extending, updating, "
        "modifying, fixing, or implementing code unless candidate_patch contains the raw diff "
        "for that change.\n"
        "Return exactly one decision. The next_command must be a single bounded command that "
        "starts with 'avo' and uses only one of: env, compile, score. Use valid CLI flags: "
        "compile requires --source SOURCE.cu and --out-dir DIR; candidate score requires "
        "--backend candidate and --candidate. Use env only for CUDA/build environment "
        "diagnostics, not for source-file inspection. Use compile only for CUDA build/"
        "compilation diagnostics or to build-check a candidate_patch, not for source-file "
        "inspection. Do not use shell pipes, redirection, command chaining, cat, head, "
        "git, rm, or arbitrary shell commands.\n\n"
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
        "Use avo env only for CUDA/build environment diagnostics, not source-file inspection.",
        "Use avo compile only for CUDA build/compilation diagnostics or to build-check a "
        "candidate_patch, not source-file inspection.",
        "Standalone pragma-only performance patches are already recorded as regressed noise; "
        "if using unroll, pair it with a substantive code change and run a bounded score.",
        "WMMA matrix fragments in generic PyTorch kernels must use explicit CUDA element "
        "types, not scalar_t, because dispatch instantiates unsupported float/c10 types.",
        "Available edit channel: candidate_patch as a raw unified diff under candidates/, "
        "or empty. Patch hunks must use exact current file context, apply cleanly, and avoid "
        "trailing whitespace.",
        "Candidate interface: module defines attention(q, k, v, causal: bool).",
        "No-patch compile diagnostics are already recorded for "
        "candidates/cuda_mma_attention/attention_kernel.cu, "
        "candidates/cuda_warp_rows_attention/attention_kernel.cu, and "
        "candidates/cuda_tiled_attention/attention_kernel.cu; compile those sources only "
        "when build-checking a candidate_patch.",
        "Unpatched seed score caps: candidates/cuda_mma_attention_seed.py supports "
        "seq_lens 16, 32, 64, 128, or 256 with head_dim 128, total_tokens <= 1024, "
        "and num_heads <= 4; "
        "candidates/cuda_warp_rows_attention_seed.py supports seq_lens <= 256 and "
        "head_dim <= 128 with total_tokens <= 1024 and num_heads <= 4; "
        "candidates/cuda_tiled_attention_seed.py is only validated at seq_lens 16 with "
        "head_dim 16, total_tokens <= 16, and num_heads 1, but that no-patch smoke is "
        "already recorded and should not be repeated. Larger seed scores need "
        "candidate_patch to update the wrapper/kernel.",
        "For tiled scores outside the tiny validated smoke shape, changing only "
        "candidates/cuda_tiled_attention_seed.py wrapper caps is not a fix; include a "
        "kernel change for the known larger-shape correctness failure.",
        "The current tiled kernel already uses the online-softmax output recurrence "
        "output_acc * old_scale + tile_acc * tile_scale; do not repeat that stale fix.",
        "The tiled reduction-bound guard patch still failed seq128/head_dim128 "
        "correctness; do not repeat that reduce[tid] score/shifted guard change.",
        "The warp-row BF16 score_tiles shared-memory conversion preserved correctness "
        "but regressed throughput; do not repeat that buffer-precision change.",
        "The unpatched MMA seq64/head_dim128 score passed correctness but is already "
        "recorded as structural progress; do not repeat it without a new candidate_patch.",
        "The unpatched MMA seq128/head_dim128 score passed correctness but is already "
        "recorded as structural progress; do not repeat it without a new candidate_patch.",
        "The unpatched MMA seq256/head_dim128 score passed correctness and was accepted "
        "into lineage; do not repeat it without a structural candidate_patch.",
        "For MMA shared K staging, k_shared is tile-local: do not load from "
        "k_shared + key_start * kHeadDim + chunk_offset. Use k_shared + chunk_offset "
        "with leading dimension kHeadDim after staging the 16x128 tile.",
        "The corrected synchronous full-K MMA staging path preserved correctness but "
        "regressed throughput; do not repeat static k_shared staging unless adding "
        "real async-copy or double-buffered overlap.",
        "The synchronous double-buffered MMA V staging path preserved correctness but "
        "regressed throughput; do not repeat static v_shared[2] staging unless adding "
        "real async-copy overlap.",
        "For MMA probability-buffer skew, WMMA load_matrix_sync leading dimensions must "
        "stay 16-byte aligned; kTile + 1 is invalid, and the corrected kTile + 8 "
        "stride preserved correctness but regressed throughput.",
        "Scalar BF16 async-copy patches are invalid: do not use __pipeline_memcpy_async "
        "with sizeof(__nv_bfloat16) or per-element BF16 loops. Use async copy only for "
        "real aligned 16-byte groups in dataflow; otherwise choose a non-async patch.",
        "Patched MMA shape extensions beyond the current seq256/head_dim128 smoke must run "
        "an avo compile build-check first; do not jump straight to score.",
        "A partial MMA head_dim128 extension that changes only kHeadDim/SMOKE_HEAD_DIM "
        "and leaves four 16-wide chunks covers only 64 dimensions; do not repeat it.",
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
    excerpts = _candidate_source_excerpts(root, [*candidates, *cuda_sources])
    if excerpts:
        lines.append("Candidate source excerpts for exact patch context:")
        lines.extend(excerpts)
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
        f"Validation error: {last_error}. "
        f"{_validation_feedback_hint(last_error)}"
        "Return one corrected decision."
    )
    updated["messages"] = messages
    return updated


def _validation_feedback_hint(error: ValueError) -> str:
    message = str(error)
    if "candidate_patch must be non-empty" in message:
        return (
            "Choose exactly one valid mode. Edit mode: candidate_patch must be a raw "
            "git-style unified diff under candidates/ starting with 'diff --git'. No-edit "
            "mode: candidate_patch must be exactly '' and candidate_edit must start with "
            "'No edit;' followed only by the bounded score/compile/env diagnostic to run. "
            "Do not mention fixing, extending, updating, modifying, or implementing code in "
            "no-edit mode. "
        )
    if "scalar BF16 __pipeline_memcpy_async" in message:
        return (
            "Do not retry scalar sizeof(__nv_bfloat16) async copies. A valid Ampere "
            "async-copy patch must copy aligned 16-byte groups, which is 8 BF16 elements "
            "per copy, and keep any scalar tail path separate after the pipeline wait. "
            "Your corrected decision should avoid __pipeline_memcpy_async entirely unless "
            "the diff contains real 16-byte-group dataflow, not wrapper/API proof code. "
            "If you cannot express that cleanly as a small diff, choose a materially "
            "different non-async candidate patch in this retry. "
        )
    if "repeats synchronous MMA K shared-memory staging" in message:
        return (
            "Do not retry static k_shared MMA K staging. It already passed correctness "
            "and regressed throughput. A corrected K-staging decision must add real "
            "async-copy or double-buffered overlap; otherwise choose a different "
            "non-K-staging candidate patch. "
        )
    if "synchronous double-buffered MMA V shared-memory staging" in message:
        return (
            "Do not retry static v_shared[2] MMA V staging. It already passed correctness "
            "and regressed throughput. A corrected V-staging decision must add real "
            "async-copy overlap; otherwise choose a different non-V-staging candidate patch. "
        )
    return ""


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
    if any(marker in value for marker in TOOL_PARAMETER_MARKERS):
        raise ValueError(f"{key} must not contain tool parameter markup")
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
        "candidate_patch must be non-empty when candidate_edit describes a code change; "
        f"candidate_edit was {_validation_excerpt(candidate_edit)!r}"
    )


def _validate_candidate_patch_not_self_rejected(
    candidate_patch: str,
    planning_text: str,
) -> None:
    if not candidate_patch.strip():
        return
    normalized = " ".join(planning_text.lower().replace("-", " ").split())
    for phrase in SELF_REJECTING_PATCH_PHRASES:
        if phrase in normalized:
            raise ValueError(
                "candidate_patch is described as known invalid by the decision itself; "
                f"found phrase {phrase!r}. Return a corrected patch or choose no-edit mode."
            )
    if "stale" in normalized and ("must remove" in normalized or "undeclared" in normalized):
        raise ValueError(
            "candidate_patch is described as known invalid by the decision itself; "
            "stale code is called out as requiring removal. Return a corrected patch "
            "or choose no-edit mode."
        )
    if "incomplete removal" in normalized and "should be completely removed" in normalized:
        raise ValueError(
            "candidate_patch is described as known invalid by the decision itself; "
            "the risk calls out incomplete removal of old code. Return a corrected patch "
            "or choose no-edit mode."
        )


def _validate_candidate_patch_domain_sanity(candidate_patch: str) -> None:
    if not candidate_patch.strip():
        return
    for added_line in _candidate_patch_added_lines(candidate_patch):
        if added_line.rstrip(" \t") != added_line:
            raise ValueError("candidate_patch added lines must not contain trailing whitespace")
    if _candidate_patch_adds_only_unroll_pragmas(candidate_patch):
        raise ValueError(
            "candidate_patch is a pragma-only performance patch; recorded warp-row "
            "unroll scoring regressed throughput, so include a substantive code change"
        )
    meaningful_added_lines = [
        line.strip() for line in _candidate_patch_added_lines(candidate_patch) if line.strip()
    ]
    if (
        len(meaningful_added_lines) == 1
        and "can_stage_shared" in meaningful_added_lines[0]
        and "head_dim <= 128" in meaningful_added_lines[0]
    ):
        raise ValueError(
            "candidate_patch directly enables the warp-row shared path for head_dim 128; "
            "the recorded threshold-only change triggered CUDA misaligned-address failures"
        )
    if _candidate_patch_repeats_stale_tiled_rescale_fix(candidate_patch):
        raise ValueError(
            "candidate_patch repeats a stale tiled output-rescale fix; the current "
            "tiled kernel already uses output_acc * old_scale + tile_acc * tile_scale"
        )
    if _candidate_patch_repeats_tiled_reduction_guard_fix(candidate_patch):
        raise ValueError(
            "candidate_patch repeats the tiled reduction-bound guard fix; the recorded "
            "seq128/head_dim128 score still failed correctness"
        )
    if _candidate_patch_repeats_partial_mma_head_dim128_extension(candidate_patch):
        raise ValueError(
            "candidate_patch repeats the partial MMA head_dim128 extension; changing "
            "kHeadDim/SMOKE_HEAD_DIM to 128 while leaving four 16-wide chunks covers "
            "only 64 dimensions"
        )
    added_text = "\n".join(_candidate_patch_added_lines(candidate_patch))
    if "__pipeline_wait_prior<" in added_text:
        raise ValueError(
            "candidate_patch uses templated __pipeline_wait_prior; the public CUDA "
            "pipeline primitive is __pipeline_wait_prior(prior)"
        )
    if "__pipeline_memcpy_async" in added_text and "sizeof(__nv_bfloat16)" in added_text:
        raise ValueError(
            "candidate_patch uses scalar BF16 __pipeline_memcpy_async copies; use "
            "16-byte aligned groups for Ampere async copy patches"
        )
    if (
        "extern __shared__" in added_text
        and "k_tiles" in added_text
        and "v_tiles" in added_text
        and "memcpy_async" not in added_text
        and "cp.async" not in added_text
    ):
        raise ValueError(
            "candidate_patch is a standalone dynamic shared-memory K/V migration; "
            "the recorded version preserved correctness but regressed throughput, "
            "so include real async-copy or double-buffering logic"
        )
    if _candidate_patch_converts_warp_score_tiles_to_bf16(added_text):
        raise ValueError(
            "candidate_patch repeats the regressed warp-row BF16 score_tiles conversion; "
            "the recorded score preserved correctness but reduced geomean throughput"
        )
    if _candidate_patch_uses_unsupported_mma_score_k32(added_text):
        raise ValueError(
            "candidate_patch uses unsupported WMMA accumulator shape 16x16x32; "
            "keep the score fragment K at 16 and accumulate two 16-wide chunks "
            "into a 16x16 accumulator"
        )
    if _candidate_patch_uses_generic_scalar_wmma_fragments(added_text):
        raise ValueError(
            "candidate_patch uses scalar_t as a WMMA matrix fragment element; "
            "generic PyTorch kernels instantiate float/c10 scalar types that WMMA "
            "does not support, so use explicit CUDA WMMA element types in "
            "dtype-specific code"
        )
    if _candidate_patch_leaves_orphan_mma_k_fragment(added_text):
        raise ValueError(
            "candidate_patch leaves an orphan post-QK WMMA k_frag block after "
            "storing scores; remove old single-chunk QK fragment declarations "
            "completely"
        )
    if _candidate_patch_uses_global_offset_for_shared_k_tile(added_text):
        raise ValueError(
            "candidate_patch stages an MMA K tile in shared memory but loads it "
            "with the global key_start offset; use tile-local k_shared + "
            "chunk_offset addressing with kHeadDim as the leading dimension"
        )
    if _candidate_patch_repeats_sync_mma_q_staging(added_text):
        raise ValueError(
            "candidate_patch repeats synchronous MMA Q shared-memory staging; "
            "the recorded score preserved correctness but regressed throughput, "
            "so add real overlap or a materially different dataflow before revisiting"
        )
    if _candidate_patch_repeats_sync_mma_k_staging(added_text):
        raise ValueError(
            "candidate_patch repeats synchronous MMA K shared-memory staging; "
            "the recorded score preserved correctness but regressed throughput, "
            "so add real async-copy or double-buffered overlap before revisiting"
        )
    if _candidate_patch_repeats_sync_mma_v_staging(added_text):
        raise ValueError(
            "candidate_patch repeats synchronous double-buffered MMA V shared-memory staging; "
            "the recorded score preserved correctness but regressed throughput, "
            "so add real async-copy overlap before revisiting"
        )
    if _candidate_patch_uses_invalid_mma_probability_ldm(added_text):
        raise ValueError(
            "candidate_patch uses kTile + 1 as a WMMA probability leading dimension; "
            "load_matrix_sync requires a 16-byte-aligned stride for half-type multiplicands"
        )
    if _candidate_patch_repeats_mma_probability_stride_skew(added_text):
        raise ValueError(
            "candidate_patch repeats the stride-24 MMA probability-buffer skew; "
            "the recorded score preserved correctness but regressed geomean throughput"
        )
    if _candidate_patch_adds_unused_async_copy_helpers(added_text):
        raise ValueError(
            "candidate_patch adds async-copy helper wrappers without using them; "
            "the API availability smoke is already recorded, so include real dataflow"
        )
    if _candidate_patch_inserts_async_helper_inside_mma_signature(added_text):
        raise ValueError(
            "candidate_patch inserts async helper definitions inside the MMA kernel "
            "signature and duplicates the kernel declaration"
        )


def _candidate_patch_added_lines(candidate_patch: str) -> list[str]:
    return [
        line[1:]
        for line in candidate_patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def _candidate_patch_changed_paths(candidate_patch: str) -> set[str]:
    paths: set[str] = set()
    for line in candidate_patch.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[:2] != ["diff", "--git"]:
            continue
        for raw_path in parts[2:4]:
            if raw_path.startswith(("a/", "b/")):
                paths.add(raw_path[2:])
    return paths


def _candidate_patch_adds_only_unroll_pragmas(candidate_patch: str) -> bool:
    added_lines = [line.strip() for line in _candidate_patch_added_lines(candidate_patch)]
    meaningful_added_lines = [line for line in added_lines if line]
    return bool(meaningful_added_lines) and all(
        line == "#pragma unroll" for line in meaningful_added_lines
    )


def _candidate_patch_uses_unsupported_mma_score_k32(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    sets_khead32 = "kHeadDim=32" in compact
    symbolic_score = "fragment<wmma::accumulator,kTile,kTile,kHeadDim,float>" in compact
    literal_score = bool(
        re.search(
            r"fragment<(?:nvcuda::)?wmma::accumulator,16,16,32,float(?:,void)?>",
            compact,
        )
    )
    return (sets_khead32 and symbolic_score) or literal_score


def _candidate_patch_converts_warp_score_tiles_to_bf16(added_text: str) -> bool:
    return (
        "__shared__ __nv_bfloat16 score_tiles" in added_text
        and "__bfloat162float(scores" in added_text
    )


def _candidate_patch_uses_generic_scalar_wmma_fragments(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    return bool(
        re.search(
            r"fragment<(?:nvcuda::)?wmma::matrix_[ab],[^>]*,scalar_t,",
            compact,
        )
    )


def _candidate_patch_repeats_stale_tiled_rescale_fix(candidate_patch: str) -> bool:
    compact = re.sub(r"\s+", "", candidate_patch)
    return (
        "-output_acc=tile_acc*tile_scale;" in compact
        and "+output_acc=output_acc*old_scale+tile_acc*tile_scale;" in compact
    )


def _candidate_patch_repeats_tiled_reduction_guard_fix(candidate_patch: str) -> bool:
    compact = re.sub(r"\s+", "", candidate_patch)
    return (
        "+reduce[tid]=score;" in compact
        and "+reduce[tid]=-std::numeric_limits<float>::infinity();" in compact
        and "+reduce[tid]=shifted;" in compact
        and "+reduce[tid]=0.0f;" in compact
    )


def _candidate_patch_repeats_partial_mma_head_dim128_extension(candidate_patch: str) -> bool:
    compact = re.sub(r"\s+", "", candidate_patch)
    extends_constant = (
        "-constexprintkHeadDim=64;" in compact
        and "+constexprintkHeadDim=128;" in compact
        and "-SMOKE_HEAD_DIM=64" in compact
        and "+SMOKE_HEAD_DIM=128" in compact
    )
    if not extends_constant:
        return False
    adds_wider_loop = (
        "+for(intchunk=0;chunk<8;++chunk)" in compact
        or "+for(intchunk=0;chunk<kHeadDim/16;++chunk)" in compact
        or "+constexprintkHeadChunks=kHeadDim/16;" in compact
    )
    return not adds_wider_loop


def _candidate_patch_leaves_orphan_mma_k_fragment(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    return bool(
        re.search(
            r"store_matrix_sync\(scores,score_frag,[^)]*\);"
            r"\}__syncthreads\(\);if\(threadIdx\.x<warpSize\)\{"
            r"wmma::fragment<wmma::matrix_b,[^>]*>k_frag;",
            compact,
        )
    )


def _candidate_patch_uses_global_offset_for_shared_k_tile(added_text: str) -> bool:
    return bool(re.search(r"\bk_shared\s*\+\s*key_start\s*\*\s*kHeadDim\b", added_text))


def _candidate_patch_repeats_sync_mma_q_staging(added_text: str) -> bool:
    return (
        "__shared__ __nv_bfloat16 q_shared[kTile * kHeadDim]" in added_text
        and bool(re.search(r"\bq_shared\s*\+\s*chunk_offset\b", added_text))
        and "cp.async" not in added_text
        and "memcpy_async" not in added_text
    )


def _candidate_patch_repeats_sync_mma_k_staging(added_text: str) -> bool:
    return (
        "__shared__ __nv_bfloat16 k_shared[kTile * kHeadDim]" in added_text
        and bool(re.search(r"\bk_shared\s*\+\s*chunk_offset\b", added_text))
        and "cp.async" not in added_text
        and "memcpy_async" not in added_text
    )


def _candidate_patch_repeats_sync_mma_v_staging(added_text: str) -> bool:
    return (
        "__shared__ __nv_bfloat16 v_shared[2][kTile * kHeadDim]" in added_text
        and "v_shared[current_buffer][chunk_offset]" in added_text
        and "cp.async" not in added_text
        and "memcpy_async" not in added_text
    )


def _candidate_patch_uses_invalid_mma_probability_ldm(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    return "load_matrix_sync(probability_frag,probabilities,kTile+1)" in compact


def _candidate_patch_repeats_mma_probability_stride_skew(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    return (
        "kProbabilityStride=kTile+8" in compact
        and "load_matrix_sync(probability_frag,probabilities,kProbabilityStride)" in compact
    )


def _candidate_patch_adds_unused_async_copy_helpers(added_text: str) -> bool:
    return (
        "__pipeline_memcpy_async" in added_text
        and "async_copy_16" in added_text
        and added_text.count("async_copy_16") == 1
        and "async_commit" in added_text
        and added_text.count("async_commit") == 1
        and "async_wait" in added_text
        and added_text.count("async_wait") == 1
    )


def _candidate_patch_inserts_async_helper_inside_mma_signature(added_text: str) -> bool:
    return (
        "__pipeline_memcpy_async" in added_text
        and "__device__ __forceinline__" in added_text
        and "__global__ void mma_attention_kernel" in added_text
    )


def _validation_excerpt(value: str, *, max_length: int = 160) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3]}..."


def _candidate_edit_requires_patch(candidate_edit: str) -> bool:
    normalized = " ".join(candidate_edit.lower().replace("-", " ").split())
    if any(phrase in normalized for phrase in NO_EDIT_PHRASES):
        return False
    words = re.findall(r"[a-z]+", normalized)
    return any(word in PATCH_REQUIRED_EDIT_VERBS for word in words)


def _validate_next_command(
    command: str,
    *,
    candidate_patch: str = "",
    planning_text: str = "",
) -> str:
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
    _validate_subcommand_arguments(
        parts,
        candidate_patch=candidate_patch,
        planning_text=planning_text,
    )
    return command


def _validate_subcommand_arguments(
    parts: list[str],
    *,
    candidate_patch: str = "",
    planning_text: str = "",
) -> None:
    subcommand = parts[1]
    if subcommand == "env":
        _validate_env_command_context(planning_text)
    elif subcommand == "compile":
        _validate_compile_command_context(
            planning_text,
            candidate_patch=candidate_patch,
        )
        if _candidate_patch_adds_only_unroll_pragmas(candidate_patch):
            raise ValueError(
                "next_command compile is not useful for a pragma-only performance patch; "
                "run a bounded candidate score to measure correctness and TFLOPS"
            )
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
            _validate_compile_source_not_recorded_baseline(
                source,
                candidate_patch=candidate_patch,
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
            _validate_patched_mma_score_is_compile_checked_first(
                parts,
                candidate=candidate,
                candidate_patch=candidate_patch,
            )
            _validate_known_candidate_score_shape(
                parts,
                candidate=candidate,
                candidate_patch=candidate_patch,
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


def _validate_compile_source_not_recorded_baseline(
    source: str,
    *,
    candidate_patch: str,
) -> None:
    if candidate_patch.strip() or source not in RECORDED_NO_PATCH_COMPILE_SOURCES:
        return
    raise ValueError(
        "next_command repeats a recorded no-patch compile diagnostic; include "
        "candidate_patch to build-check a change or run a bounded score instead"
    )


def _validate_known_candidate_score_shape(
    parts: list[str],
    *,
    candidate: str,
    candidate_patch: str,
) -> None:
    seq_lens = _score_seq_lens(parts)
    head_dim = _score_head_dim(parts)
    total_tokens = _score_positive_int_option(
        parts,
        "--total-tokens",
        default=DEFAULT_SCORE_TOTAL_TOKENS,
    )
    num_heads = _score_positive_int_option(
        parts,
        "--num-heads",
        default=DEFAULT_SCORE_NUM_HEADS,
    )
    if candidate_patch.strip():
        if (
            candidate == TILED_SEED
            and _is_outside_tiled_validated_cap(
                seq_lens=seq_lens,
                head_dim=head_dim,
                total_tokens=total_tokens,
                num_heads=num_heads,
            )
            and _candidate_patch_changed_paths(candidate_patch) <= {TILED_SEED}
        ):
            raise ValueError(
                "next_command scores tiled outside the validated smoke shape after only "
                "changing wrapper caps; include a kernel fix for the larger-shape "
                "correctness failure"
            )
        return
    if candidate == WARP_ROWS_SEED:
        if (
            any(seq_len > 256 for seq_len in seq_lens)
            or head_dim > 128
            or total_tokens > 1024
            or num_heads > 4
        ):
            raise ValueError(
                "next_command scores cuda_warp_rows_attention_seed.py outside its "
                "unpatched seq_len<=256/head_dim<=128/total_tokens<=1024/num_heads<=4 "
                "cap; include candidate_patch to update the wrapper/kernel first"
            )
    elif candidate == MMA_SEED:
        if (
            any(seq_len not in {16, 32, 64, 128, 256} for seq_len in seq_lens)
            or head_dim != 128
            or total_tokens > 1024
            or num_heads > 4
        ):
            raise ValueError(
                "next_command scores cuda_mma_attention_seed.py outside its unpatched "
                "seq_len 16/32/64/128/256, head_dim 128, total_tokens<=1024, and "
                "num_heads<=4 cap; "
                "include candidate_patch to update the wrapper/kernel first"
            )
        raise ValueError(
            "next_command repeats a recorded unpatched MMA seed score; include "
            "candidate_patch to change kernel structure before scoring"
        )
    elif candidate == TILED_SEED:
        if _is_outside_tiled_validated_cap(
            seq_lens=seq_lens,
            head_dim=head_dim,
            total_tokens=total_tokens,
            num_heads=num_heads,
        ):
            raise ValueError(
                "next_command scores cuda_tiled_attention_seed.py outside its unpatched "
                "validated seq_len 16, head_dim 16, total_tokens<=16, and num_heads=1 "
                "cap; include candidate_patch to fix or extend the wrapper/kernel first"
            )
        raise ValueError(
            "next_command repeats the recorded no-patch tiled smoke score; include "
            "candidate_patch to fix or extend the tiled wrapper/kernel first"
        )


def _is_outside_tiled_validated_cap(
    *,
    seq_lens: tuple[int, ...],
    head_dim: int,
    total_tokens: int,
    num_heads: int,
) -> bool:
    return (
        any(seq_len != 16 for seq_len in seq_lens)
        or head_dim != 16
        or total_tokens > 16
        or num_heads != 1
    )


def _validate_patched_mma_score_is_compile_checked_first(
    parts: list[str],
    *,
    candidate: str,
    candidate_patch: str,
) -> None:
    if not candidate_patch.strip() or candidate != MMA_SEED:
        return
    seq_lens = _score_seq_lens(parts)
    head_dim = _score_head_dim(parts)
    total_tokens = _score_positive_int_option(
        parts,
        "--total-tokens",
        default=DEFAULT_SCORE_TOTAL_TOKENS,
    )
    num_heads = _score_positive_int_option(
        parts,
        "--num-heads",
        default=DEFAULT_SCORE_NUM_HEADS,
    )
    if (
        all(seq_len in {16, 32, 64, 128, 256} for seq_len in seq_lens)
        and head_dim == 128
        and total_tokens <= 1024
        and num_heads <= 4
    ):
        return
    raise ValueError(
        "next_command scores a patched MMA shape extension beyond the current "
        "seq256/head_dim128 smoke; "
        "first run avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/<name> to build-check the candidate_patch"
    )


def _validate_env_command_context(planning_text: str) -> None:
    normalized = " ".join(planning_text.lower().replace("_", " ").replace("-", " ").split())
    if any(keyword in normalized for keyword in ENV_COMMAND_KEYWORDS):
        return
    raise ValueError(
        "next_command avo env is only for CUDA/build environment diagnostics, "
        "not source-file inspection"
    )


def _validate_compile_command_context(
    planning_text: str,
    *,
    candidate_patch: str,
) -> None:
    if candidate_patch.strip():
        return
    normalized = " ".join(planning_text.lower().replace("_", " ").replace("-", " ").split())
    if any(keyword in normalized for keyword in COMPILE_COMMAND_KEYWORDS):
        return
    raise ValueError(
        "next_command avo compile is only for CUDA build/compilation diagnostics "
        "or checking a candidate_patch, not source-file inspection"
    )


def _score_seq_lens(parts: list[str]) -> tuple[int, ...]:
    raw = _single_option_value(parts, "--seq-lens")
    if raw is None:
        return DEFAULT_SCORE_SEQ_LENS
    values: list[int] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            raise ValueError("next_command --seq-lens must be a comma-separated list of integers")
        try:
            value = int(stripped)
        except ValueError as exc:
            raise ValueError("next_command --seq-lens must contain only integers") from exc
        if value <= 0:
            raise ValueError("next_command --seq-lens values must be positive")
        values.append(value)
    return tuple(values)


def _score_head_dim(parts: list[str]) -> int:
    return _score_positive_int_option(parts, "--head-dim", default=DEFAULT_SCORE_HEAD_DIM)


def _score_positive_int_option(parts: list[str], option: str, *, default: int) -> int:
    raw = _single_option_value(parts, option)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"next_command {option} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"next_command {option} must be positive")
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


def _relative_files(root: Path, dirname: str, *, suffix: str) -> list[str]:
    base = root / dirname
    if not base.exists():
        return []
    return sorted(
        str(path.relative_to(root))
        for path in base.rglob(f"*{suffix}")
        if "__pycache__" not in path.parts
    )


def _candidate_source_excerpts(
    root: Path,
    relative_paths: list[str],
    *,
    max_file_chars: int = MAX_REPO_CONTEXT_FILE_CHARS,
    max_total_chars: int = MAX_REPO_CONTEXT_SOURCE_CHARS,
) -> list[str]:
    excerpts: list[str] = []
    root = root.resolve()
    total_chars = 0
    for relative_path in relative_paths:
        if total_chars >= max_total_chars:
            break
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if not path.is_file():
            continue
        available_chars = min(max_file_chars, max_total_chars - total_chars)
        if available_chars <= 0:
            break
        content = path.read_text(encoding="utf-8", errors="replace")
        excerpt = content[:available_chars].rstrip()
        if len(content) > available_chars:
            excerpt = f"{excerpt}\n... truncated {len(content) - available_chars} chars ..."
        total_chars += len(excerpt)
        excerpts.extend(
            [
                f"-- {relative_path} --",
                excerpt,
                f"-- end {relative_path} --",
            ]
        )
    return excerpts


def _preferred_candidate_score_command(candidates: list[str]) -> str:
    if "candidates/cuda_mma_attention_seed.py" in candidates:
        return (
            "avo score --backend candidate "
            "--candidate candidates/cuda_mma_attention_seed.py "
            "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128 "
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
