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
STRUCTURED_TRANSFORM_OPS = frozenset(
    {
        "replace_once",
        "insert_before_once",
        "insert_after_once",
        "set_constexpr_int",
    }
)
PLANNING_RISK_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "no_effect_or_skeleton",
        (
            r"\b(?:cannot|does not|do not)\s+(?:affect|improve)\s+"
            r"(?:correctness(?:\s+or\s+throughput)?|throughput)",
            r"\bcompile\s+only\s+structural\s+probe\b",
            r"\b(?:stub is empty|not yet called|does not yet consume|unused in this patch)",
            r"\b(?:must not|do not)\s+be\s+scored\b",
            r"\bunused\s+(?:doubled buffers|preload|skeleton)",
        ),
    ),
    (
        "incomplete_or_malformed_edit",
        (
            r"\b(?:diff is incomplete|diff structure error|duplicate store line)\b",
            r"\b(?:not ready|must be updated before scoring)\b",
            r"\bdo not (?:apply this patch|use this diff)\b",
            r"\b(?:reject this direction|should be rejected)\b",
            r"\bstale\b.*\b(?:must remove|undeclared)\b",
            r"\bincomplete removal\b.*\bshould be completely removed\b",
            r"\bmust define\b.*\bbefore using it\b",
        ),
    ),
    (
        "predicted_compile_failure",
        (
            r"\b(?:will|would|likely)\s+(?:cause\s+)?(?:nvcc\s+)?(?:compile|compilation)\s+failure\b",
            r"\b(?:will|would)\s+(?:cause\s+)?(?:a\s+)?compile error\b",
            r"\b(?:will|would)\s+(?:fail|break)\s+(?:compile|compilation)\b",
            r"\bwill cause nvcc to fail\b",
        ),
    ),
    (
        "predicted_correctness_failure",
        (
            r"\b(?:will|would)\s+break correctness\b",
            r"\b(?:will|would)\s+fail correctness\b",
            r"\bwill compute the wrong\b",
            r"\blikely\s+(?:segfault|.*produce incorrect results)\b",
        ),
    ),
)
TRANSFORM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {
            "type": "string",
            "enum": sorted(STRUCTURED_TRANSFORM_OPS),
            "description": "Small structured edit operation materialized by the orchestrator.",
        },
        "path": {
            "type": "string",
            "description": "Repo-relative candidate source path under candidates/.",
        },
        "find": {"type": "string", "description": "Exact text to replace for replace_once."},
        "replace": {"type": "string", "description": "Replacement text for replace_once."},
        "anchor": {
            "type": "string",
            "description": "Exact text anchor for insert_before_once or insert_after_once.",
        },
        "text": {
            "type": "string",
            "description": "Inserted text for insert_before_once or insert_after_once.",
        },
        "name": {
            "type": "string",
            "description": "constexpr integer name for set_constexpr_int.",
        },
        "value": {
            "type": "integer",
            "description": "constexpr integer value for set_constexpr_int.",
        },
    },
    "required": ["op", "path"],
    "additionalProperties": False,
}

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
                "Legacy raw git-style unified diff for one small candidate edit under "
                "candidates/, starting with 'diff --git', or empty string. Prefer "
                "candidate_transform for CUDA edits so the orchestrator materializes and "
                "preflights the patch instead of trusting generated hunks. Do not use "
                "markdown fences."
            ),
        },
        "candidate_transform": {
            **TRANSFORM_SCHEMA,
            "description": (
                "Preferred edit channel for CUDA/kernel evolution. Use exactly one tiny "
                "structured transform, or omit this field for no-edit/legacy patch mode."
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
    candidate_transform: dict[str, Any] | None = None

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
        candidate_transform = _validate_candidate_transform(
            normalized_payload.get("candidate_transform")
        )
        if candidate_patch.strip() and candidate_transform is not None:
            raise ValueError(
                "candidate_patch and candidate_transform are mutually exclusive; "
                "use one edit channel"
            )
        expected_effect = _require_string(normalized_payload, "expected_effect")
        risk = _require_string(normalized_payload, "risk")
        edit_payload = candidate_patch or ("<structured-transform>" if candidate_transform else "")
        _validate_candidate_edit_channel_consistency(
            candidate_edit,
            candidate_patch=candidate_patch,
            candidate_transform=candidate_transform,
        )
        _validate_candidate_edit_matches_patch(candidate_edit, edit_payload)
        planning_text = "\n".join((hypothesis, candidate_edit, expected_effect, risk))
        _validate_candidate_patch_not_self_rejected(edit_payload, planning_text)
        _validate_candidate_patch_domain_sanity(candidate_patch)
        next_command = _validate_next_command(
            _require_string(normalized_payload, "next_command"),
            candidate_patch=edit_payload,
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
            candidate_transform=candidate_transform,
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
            "candidate_transform": self.candidate_transform,
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
        "Use one of exactly three modes. Preferred edit mode: candidate_transform is one "
        "tiny structured operation under candidates/ and candidate_patch is exactly the "
        "empty string \"\". "
        "Supported ops are replace_once, insert_before_once, insert_after_once, and "
        "set_constexpr_int; the orchestrator materializes and preflights the patch. "
        "Legacy edit mode: candidate_patch is one small raw git-style unified diff under "
        "candidates/ starting with 'diff --git'. Use this only when no structured "
        "transform can express the edit. No-edit mode: candidate_patch is exactly the "
        "empty string \"\", candidate_edit starts with \"No edit; \", and next_command is only "
        "a bounded score, compile, or environment diagnostic for existing files. Do not include "
        "markdown fences or commentary in candidate_patch. Do not describe extending, updating, "
        "modifying, fixing, or implementing code unless candidate_transform or candidate_patch "
        "contains that change.\n"
        "Return exactly one decision. The next_command must be a single bounded command that "
        "starts with 'avo' and uses only one of: env, compile, score. Use valid CLI flags: "
        "compile requires --source SOURCE.cu and --out-dir DIR; candidate score requires "
        "--backend candidate and --candidate. Use env only for CUDA/build environment "
        "diagnostics, not for source-file inspection. Use compile only for CUDA build/"
        "compilation diagnostics or to build-check a candidate_transform/candidate_patch, "
        "not for source-file inspection. Do not use shell pipes, redirection, command "
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
        "Use avo env only for CUDA/build environment diagnostics, not source-file inspection.",
        "The current CUDA/build environment is already recorded as stable "
        "(torch CUDA 13.0, nvcc CUDA 13.0, RTX A6000 sm_86, Anthropic key present); "
        "do not run no-edit avo env merely to confirm stability unless a recent "
        "build/environment failure gives a concrete reason.",
        "Use avo compile only for CUDA build/compilation diagnostics or to build-check a "
        "candidate_transform/candidate_patch, not source-file inspection.",
        "Standalone pragma-only performance patches are already recorded as regressed noise; "
        "if using unroll, pair it with a substantive code change and run a bounded score.",
        "WMMA matrix fragments in generic PyTorch kernels must use explicit CUDA element "
        "types, not scalar_t, because dispatch instantiates unsupported float/c10 types.",
        "Preferred edit channel: candidate_transform, a single tiny structured operation "
        "under candidates/. Supported ops: replace_once, insert_before_once, "
        "insert_after_once, set_constexpr_int. Legacy candidate_patch raw diffs are allowed "
        "only when a transform cannot express the change.",
        "Candidate interface: module defines attention(q, k, v, causal: bool).",
        "No-patch compile diagnostics are already recorded for "
        "candidates/cuda_mma_attention/attention_kernel.cu, "
        "candidates/cuda_warp_rows_attention/attention_kernel.cu, and "
        "candidates/cuda_tiled_attention/attention_kernel.cu; compile those sources only "
        "when build-checking a candidate_transform/candidate_patch.",
        "Unpatched seed score caps: candidates/cuda_mma_attention_seed.py supports "
        "seq_lens 16, 32, 64, 128, or 256 with head_dim 128, total_tokens <= 1024, "
        "and num_heads <= 4; "
        "candidates/cuda_warp_rows_attention_seed.py supports seq_lens <= 256 and "
        "head_dim <= 128 with total_tokens <= 1024 and num_heads <= 4; "
        "candidates/cuda_tiled_attention_seed.py is only validated at seq_lens 16 with "
        "head_dim 16, total_tokens <= 16, and num_heads 1, but that no-patch smoke is "
        "already recorded and should not be repeated. Larger seed scores need "
        "candidate_transform/candidate_patch to update the wrapper/kernel.",
        "For tiled scores outside the tiny validated smoke shape, changing only "
        "candidates/cuda_tiled_attention_seed.py wrapper caps is not a fix; include a "
        "kernel change for the known larger-shape correctness failure.",
        "The current tiled kernel already uses the online-softmax output recurrence "
        "output_acc * old_scale + tile_acc * tile_scale; do not repeat that stale fix.",
        "The tiled reduction-bound guard patch still failed seq128/head_dim128 "
        "correctness; do not repeat that reduce[tid] score/shifted guard change.",
        "The warp-row BF16 score_tiles shared-memory conversion preserved correctness "
        "but regressed throughput; do not repeat that buffer-precision change.",
        "The unpatched warp-row seq256/head_dim128 score passed correctness but "
        "regressed below the accepted MMA direct-accumulation kernel; do not repeat "
        "that no-patch diagnostic without a structural candidate_patch.",
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
        "Scalar async-copy validation has failed repeatedly in recent loops. Treat cp.async/"
        "__pipeline_memcpy_async as a cooled-down direction unless the diff is a complete "
        "16-byte-group dataflow change with exact current context and no scalar async calls.",
        "Compile-only WMMA skeletons that add fragments or shared buffers without wiring "
        "them into MMA and online-softmax dataflow are recorded no-ops; build-check only "
        "candidate patches that are intended to be scored after a successful compile.",
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
    if "variation decision missing required keys" in message:
        return (
            "Return one complete variation decision object with every required field: "
            "hypothesis, files_to_inspect, candidate_edit, candidate_patch, "
            "expected_effect, risk, and next_command. Do not omit expected_effect, "
            "risk, or next_command even in no-edit diagnostic mode. "
        )
    if "candidate_patch must be non-empty" in message:
        return (
            "Choose exactly one valid mode. Preferred edit mode: candidate_transform is "
            "one structured operation and candidate_patch is ''. Legacy edit mode: "
            "candidate_patch must be a raw git-style unified diff under candidates/ "
            "starting with 'diff --git'. No-edit mode: candidate_patch must be exactly "
            "'' and candidate_edit must start with 'No edit;' followed only by the "
            "bounded score/compile/env diagnostic to run. "
            "Do not mention fixing, extending, updating, modifying, or implementing code in "
            "no-edit mode. "
        )
    if "candidate_patch and candidate_transform are mutually exclusive" in message:
        return (
            "Choose one edit channel only. For preferred structured-transform mode, set "
            "candidate_transform to the operation object and set candidate_patch to exactly "
            "the empty string ''. For legacy raw-diff mode, set candidate_patch to the "
            "unified diff and omit candidate_transform. "
        )
    if "no-edit mode but includes an edit payload" in message:
        return (
            "No-edit mode cannot include candidate_transform or candidate_patch. Either "
            "remove the edit payload and run only the bounded diagnostic, or remove the "
            "'No edit;' prefix and describe the structured transform/raw diff as the edit. "
        )
    if "known invalid by the decision itself" in message:
        if "must not be scored" in message:
            return (
                "The previous patch described itself as 'must not be scored', so do not "
                "retry another compile-only skeleton. Choose exactly one valid mode: "
                "edit mode with a structured candidate_transform or raw diff that is "
                "complete enough to score after a successful compile and whose risk text "
                "does not say it must not be scored, or No edit; mode for a different "
                "already-bounded diagnostic. "
            )
        return (
            "Do not retry an edit whose own hypothesis, expected_effect, or risk puts it "
            "in a failed planning-risk class such as predicted compile failure, predicted "
            "correctness failure, no-effect skeleton, or incomplete edit. Prefer a "
            "structured candidate_transform for the corrected change; otherwise switch to "
            "No edit; mode for a bounded diagnostic. "
        )
    if "--out-dir must be under: build" in message:
        return (
            "For compile checks, set --out-dir to a repo-relative build subdirectory such "
            "as build/mma_probe. Do not write compiler outputs under candidates/ or beside "
            "source files. "
        )
    if "recorded unpatched MMA seed score" in message:
        return (
            "Do not retry a no-edit score of cuda_mma_attention_seed.py. That exact "
            "MMA seed score is already in lineage. To score this candidate again, use "
            "edit mode with candidate_transform or a legacy candidate_patch that "
            "structurally changes candidates/cuda_mma_attention/attention_kernel.cu or "
            "its wrapper; otherwise choose a different diagnostic such as a compile "
            "check for a new edit. "
        )
    if "recorded no-patch warp-row seed score" in message:
        return (
            "Do not retry a no-edit score of cuda_warp_rows_attention_seed.py on the "
            "recorded seq256/head_dim128 workload. That diagnostic already passed "
            "correctness and was gate-rejected for throughput. Use edit mode with "
            "candidate_transform or a legacy candidate_patch before scoring that "
            "workload again. "
        )
    if "recorded environment stability diagnostic" in message:
        return (
            "Do not spend a loop step on avo env just to reconfirm the already-recorded "
            "CUDA/build environment. Use avo env only when the decision cites a concrete "
            "recent build or environment failure such as a CUDA version mismatch, missing "
            "compiler, missing package, or extension-build error. "
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
    if (
        "synchronous MMA V shared-memory staging" in message
        or "synchronous double-buffered MMA V shared-memory staging" in message
    ):
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


def _validate_candidate_transform(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("candidate_transform must be an object when provided")
    op = value.get("op")
    path = value.get("path")
    if not isinstance(op, str) or op not in STRUCTURED_TRANSFORM_OPS:
        allowed = ", ".join(sorted(STRUCTURED_TRANSFORM_OPS))
        raise ValueError(f"candidate_transform op must be one of: {allowed}")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("candidate_transform path must be a non-empty string")
    _validate_candidate_transform_path(path)
    extra = set(value) - set(TRANSFORM_SCHEMA["properties"])
    if extra:
        raise ValueError(
            "candidate_transform contains unsupported keys: " + ", ".join(sorted(extra))
        )
    if op == "replace_once":
        _require_transform_string(value, "find")
        _require_transform_string(value, "replace")
    elif op in {"insert_before_once", "insert_after_once"}:
        _require_transform_string(value, "anchor")
        _require_transform_string(value, "text")
    elif op == "set_constexpr_int":
        _require_transform_string(value, "name")
        if not isinstance(value.get("value"), int):
            raise ValueError("candidate_transform value must be an integer")
    return dict(value)


def _validate_candidate_transform_path(raw_path: str) -> None:
    if "\x00" in raw_path or "\\" in raw_path:
        raise ValueError("candidate_transform path contains unsupported characters")
    if any(char.isspace() for char in raw_path):
        raise ValueError("candidate_transform path must not contain whitespace")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("candidate_transform path must be repo-relative")
    normalized = path.as_posix()
    if not normalized.startswith("candidates/"):
        raise ValueError("candidate_transform path must be under candidates/")
    if path.suffix not in (".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"):
        raise ValueError("candidate_transform path must reference a candidate source file")


def _require_transform_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate_transform {key} must be a non-empty string")
    return value


def _validate_candidate_edit_channel_consistency(
    candidate_edit: str,
    *,
    candidate_patch: str,
    candidate_transform: dict[str, Any] | None,
) -> None:
    normalized = " ".join(candidate_edit.lower().replace("-", " ").split())
    if not any(phrase in normalized for phrase in NO_EDIT_PHRASES):
        return
    if candidate_patch.strip() or candidate_transform is not None:
        raise ValueError(
            "candidate_edit starts in no-edit mode but includes an edit payload; "
            "remove candidate_transform/candidate_patch or describe the edit instead"
        )


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
    for risk_class, patterns in PLANNING_RISK_PATTERNS:
        for pattern in patterns:
            if not re.search(pattern, normalized):
                continue
            excerpt = re.search(pattern, normalized)
            found = excerpt.group(0) if excerpt is not None else pattern
            raise ValueError(
                "candidate_patch is described as known invalid by the decision itself; "
                f"planning risk class {risk_class!r} matched {found!r}. "
                "Return a corrected transform/patch or choose no-edit mode."
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
    if _candidate_patch_uses_unsupported_mma_m32_fragment(added_text):
        raise ValueError(
            "candidate_patch uses unsupported WMMA M=32 fragment shapes; "
            "Ampere WMMA BF16 paths in this kernel must stay on 16x16x16 fragments"
        )
    if _candidate_patch_uses_generic_scalar_wmma_fragments(added_text):
        raise ValueError(
            "candidate_patch uses scalar_t as a WMMA matrix fragment element; "
            "generic PyTorch kernels instantiate float/c10 scalar types that WMMA "
            "does not support, so use explicit CUDA WMMA element types in "
            "dtype-specific code"
        )
    if _candidate_patch_uses_missing_wmma_matrix_element_type(added_text):
        raise ValueError(
            "candidate_patch declares a WMMA matrix fragment without an element type; "
            "matrix_a/matrix_b fragments must include __nv_bfloat16 or another "
            "supported CUDA WMMA element type before the layout"
        )
    if _candidate_patch_leaves_orphan_mma_k_fragment(added_text):
        raise ValueError(
            "candidate_patch leaves an orphan post-QK WMMA k_frag block after "
            "storing scores; remove old single-chunk QK fragment declarations "
            "completely"
        )
    if _candidate_patch_uses_thread_local_mma_row_state_for_cross_thread_rows(added_text):
        raise ValueError(
            "candidate_patch moves MMA row softmax state into per-thread registers but "
            "later uses row / blockDim.x from other threads; the recorded score failed "
            "correctness with non-finite outputs"
        )
    if _candidate_patch_repeats_mma_qk_fragment_preload_chain(added_text):
        raise ValueError(
            "candidate_patch repeats the MMA QK k_frag_next preload chain; "
            "the recorded score preserved correctness but regressed geomean throughput"
        )
    if _candidate_patch_repeats_mma_q_fragment_preload_chain(added_text):
        raise ValueError(
            "candidate_patch repeats the MMA QK q_frag_next preload chain; "
            "the recorded score preserved correctness but regressed geomean throughput"
        )
    if _candidate_patch_adds_unused_mma_preload_fragment(added_text):
        raise ValueError(
            "candidate_patch adds an MMA preload fragment that is loaded but never consumed; "
            "compile-only unused preload skeletons do not affect correctness or throughput"
        )
    if _candidate_patch_adds_unused_wmma_compile_skeleton(added_text):
        raise ValueError(
            "candidate_patch adds a WMMA compile skeleton without any MMA or online-softmax "
            "dataflow; compile-only WMMA skeletons do not affect correctness or throughput"
        )
    if _candidate_patch_adds_stray_mma_probability_fragment_statement(added_text):
        raise ValueError(
            "candidate_patch adds a stray probability_frag statement in a PV preload patch; "
            "remove duplicate fragment declaration lines before compile-checking"
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
            "candidate_patch repeats synchronous MMA V shared-memory staging; "
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
    if _candidate_patch_repeats_mma_probability_stride20_skew(added_text):
        raise ValueError(
            "candidate_patch repeats the stride-20 MMA probability-buffer skew; "
            "the recorded score preserved correctness but regressed geomean throughput"
        )
    if _candidate_patch_uses_2d_probability_index_without_2d_declaration(added_text):
        raise ValueError(
            "candidate_patch uses probabilities[row][key] but does not declare "
            "probabilities as a 2D shared-memory tile"
        )
    if _candidate_patch_repeats_mma_score_stride_skew(added_text):
        raise ValueError(
            "candidate_patch repeats the MMA score-tile stride-24 skew; "
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


def _candidate_patch_uses_unsupported_mma_m32_fragment(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    explicit_m32 = bool(
        re.search(
            r"fragment<(?:nvcuda::)?wmma::(?:accumulator|matrix_[ab]),32,16,16,",
            compact,
        )
    )
    sets_ktile32 = "kTile=32" in compact or "kTile=32;" in compact
    symbolic_m32 = bool(
        re.search(
            r"fragment<(?:nvcuda::)?wmma::(?:accumulator|matrix_[ab]),kTile,16,16,",
            compact,
        )
    )
    return explicit_m32 or (sets_ktile32 and symbolic_m32)


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


def _candidate_patch_uses_missing_wmma_matrix_element_type(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    return bool(
        re.search(
            r"fragment<(?:nvcuda::)?wmma::matrix_[ab],[^,>]+,[^,>]+,[^,>]+,"
            r"(?:nvcuda::)?wmma::(?:row|col)_major>",
            compact,
        )
    )


def _candidate_patch_repeats_mma_qk_fragment_preload_chain(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    return (
        "k_frag_next" in added_text
        and "mma_sync(score_frag,q_frag,k_frag_next,score_frag)" in compact
        and "next_chunk=chunk+1" in compact
        and "next_chunk<8" in compact
        and "next_chunk*16" in compact
    )


def _candidate_patch_repeats_mma_q_fragment_preload_chain(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    return (
        "q_frag_next" in added_text
        and "q_frag=q_frag_next" in compact
        and "load_matrix_sync(q_frag_next" in compact
        and "next_chunk_offset=(chunk+1)*16" in compact
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


def _candidate_patch_uses_thread_local_mma_row_state_for_cross_thread_rows(
    added_text: str,
) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    return (
        (
            "row_max_reg" in compact
            and "row_sum_reg" in compact
            and "old_scale_reg" in compact
            and "row/blockDim.x" in compact
            and "output_acc[linear]*=old_scale_reg[reg_idx]" in compact
        )
        or (
            "floatrow_max[kTile];" in compact
            and "floatrow_sum[kTile];" in compact
            and "floatold_scale[kTile];" in compact
            and "threadIdx.x==0" in compact
        )
    )


def _candidate_patch_adds_unused_mma_preload_fragment(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    return (
        (
            "load_matrix_sync(k_frag_next," in compact
            or "load_matrix_sync(q_frag_next," in compact
            or "load_matrix_sync(probability_frag_next," in compact
        )
        and "mma_sync" not in compact
    )


def _candidate_patch_adds_unused_wmma_compile_skeleton(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    adds_wmma_fragment = (
        "wmma::fragment<wmma::accumulator" in compact
        or "wmma::fragment<wmma::matrix_a" in compact
        or "wmma::fragment<wmma::matrix_b" in compact
    )
    return adds_wmma_fragment and "wmma::fill_fragment" in compact and "mma_sync(" not in compact


def _candidate_patch_adds_stray_mma_probability_fragment_statement(added_text: str) -> bool:
    stripped_lines = {line.strip() for line in added_text.splitlines()}
    return (
        "probability_frag_next" in added_text
        and "probability_frag = probability_frag_next;" in added_text
        and "probability_frag;" in stripped_lines
    )


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
    compact = re.sub(r"\s+", "", added_text)
    return (
        (
            "__shared____nv_bfloat16v_shared[2][kTile*kHeadDim]" in compact
            or "__shared____nv_bfloat16v_shared[kTile*kHeadDim]" in compact
        )
        and (
            "v_shared[current_buffer][chunk_offset]" in compact
            or "v_shared+chunk_offset" in compact
        )
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
        and (
            "load_matrix_sync(probability_frag,probabilities,kProbabilityStride)" in compact
            or "load_matrix_sync(probability_frag,&probabilities[0][0],kProbabilityStride)"
            in compact
        )
    )


def _candidate_patch_repeats_mma_probability_stride20_skew(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    return (
        "kProbabilityStride=20" in compact
        and "__shared____nv_bfloat16probabilities[kTile][kProbabilityStride]" in compact
        and "load_matrix_sync(probability_frag,&probabilities[0][0],kProbabilityStride)"
        in compact
    )


def _candidate_patch_uses_2d_probability_index_without_2d_declaration(
    added_text: str,
) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    return (
        "probabilities[row][key]" in compact
        and "__shared____nv_bfloat16probabilities[kTile][" not in compact
    )


def _candidate_patch_repeats_mma_score_stride_skew(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    return (
        ("kScoreStride=24" in compact or "kScoreStride=kTile+8" in compact)
        and "__shared__floatscores[kTile][kScoreStride]" in compact
        and "store_matrix_sync(&scores[0][0],score_frag,kScoreStride" in compact
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
            _validate_command_path(out_dir, "--out-dir", allowed_roots=("build/",))
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
        "candidate_transform/candidate_patch to build-check a change or run a bounded "
        "score instead"
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
                "cap; include candidate_transform/candidate_patch to update the "
                "wrapper/kernel first"
            )
        if (
            seq_lens == (256,)
            and head_dim == 128
            and total_tokens == 1024
            and num_heads == 4
        ):
            raise ValueError(
                "next_command repeats a recorded no-patch warp-row seed score; include "
                "candidate_transform/candidate_patch to change kernel structure before scoring"
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
                "include candidate_transform/candidate_patch to update the wrapper/kernel first"
            )
        raise ValueError(
            "next_command repeats a recorded unpatched MMA seed score; include "
            "candidate_transform/candidate_patch to change kernel structure before scoring"
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
                "cap; include candidate_transform/candidate_patch to fix or extend the "
                "wrapper/kernel first"
            )
        raise ValueError(
            "next_command repeats the recorded no-patch tiled smoke score; include "
            "candidate_transform/candidate_patch to fix or extend the tiled "
            "wrapper/kernel first"
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
    if _env_command_repeats_recorded_stability_check(normalized):
        raise ValueError(
            "next_command repeats a recorded environment stability diagnostic; use avo env "
            "only after a concrete CUDA/build environment failure"
        )
    if any(keyword in normalized for keyword in ENV_COMMAND_KEYWORDS):
        return
    raise ValueError(
        "next_command avo env is only for CUDA/build environment diagnostics, "
        "not source-file inspection"
    )


def _env_command_repeats_recorded_stability_check(normalized_planning_text: str) -> bool:
    stability_claims = (
        "confirm cuda build setup",
        "confirm cuda build toolchain stability",
        "confirm toolchain stability",
        "environment is stable",
        "environment stability",
        "remains valid",
        "toolchain stability",
    )
    concrete_failure_terms = (
        "build failed",
        "build failure",
        "compile failed",
        "compile failure",
        "error",
        "failed",
        "failure",
        "mismatch",
        "misconfigured",
        "missing",
        "not found",
        "unavailable",
    )
    return any(claim in normalized_planning_text for claim in stability_claims) and not any(
        term in normalized_planning_text for term in concrete_failure_terms
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
