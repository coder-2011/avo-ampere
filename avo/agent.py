from __future__ import annotations

import json
import os
import re
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

DECISION_TOOL_NAME = "record_variation_decision"
DEFAULT_AGENT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_AGENT_REQUEST_ATTEMPTS = 3
DEFAULT_AGENT_RETRY_DELAY_S = 1.0
DEFAULT_AGENT_REQUEST_TIMEOUT_S = 180.0
AGENT_REQUEST_TIMEOUT_ENV = "AVO_AGENT_REQUEST_TIMEOUT_S"
ALLOWED_NEXT_COMMANDS = frozenset({"env", "compile", "profile", "score"})
EDIT_MODES = frozenset({"legacy_patch", "no_edit", "transform"})
SHELL_CONTROL_TOKENS = frozenset({"&&", "||", ";", "|", ">", ">>", "<", "`"})
TOOL_PARAMETER_MARKERS = ("<parameter ", "</parameter>")
DEFAULT_SCORE_HEAD_DIM = 128
DEFAULT_SCORE_NUM_HEADS = 16
DEFAULT_SCORE_SEQ_LENS = (4096, 8192, 16384, 32768)
DEFAULT_SCORE_TOTAL_TOKENS = 32768
MMA_ACCEPTED_VALIDATION_SEQ = 32768
MAX_REPO_CONTEXT_FILE_CHARS = 12_000
MAX_REPO_CONTEXT_SOURCE_CHARS = 45_000
PROFILER_UNSUPPORTED_RUNTIME_MARKER = Path("/etc/thunder/libthunder.so")
WARP_ROWS_SEED = "candidates/cuda_warp_rows_attention_seed.py"
MMA_SEED = "candidates/cuda_mma_attention_seed.py"
MMA_KERNEL_SOURCE = "candidates/cuda_mma_attention/attention_kernel.cu"
TILED_SEED = "candidates/cuda_tiled_attention_seed.py"
RECORDED_NO_PATCH_COMPILE_SOURCES = frozenset(
    {
        "candidates/cuda_mma_attention/attention_kernel.cu",
        "candidates/cuda_tiled_attention/attention_kernel.cu",
        "candidates/cuda_warp_rows_attention/attention_kernel.cu",
    }
)
MMA_BASE_SMOKE_SEQUENCES = frozenset(
    {16, 32, 64, 128, 256, 1024, 2048, 4096, 8192, 16384, 32768}
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
STRUCTURED_TRANSFORM_STEP_OPS = frozenset(
    {
        "add_include",
        "add_int_to_python_set",
        "replace_once",
        "insert_before_once",
        "insert_after_once",
        "set_constexpr_int",
    }
)
STRUCTURED_TRANSFORM_OPS = STRUCTURED_TRANSFORM_STEP_OPS | frozenset({"batch"})
MAX_TRANSFORM_BATCH_STEPS = 8
SUPPORT_ONLY_TRANSFORM_OPS = frozenset({"add_include"})
CONTRACT_ONLY_TRANSFORM_OPS = frozenset({"set_constexpr_int", "add_int_to_python_set"})
PLANNING_EDIT_TERMS = frozenset(
    {
        "change",
        "diff",
        "edit",
        "patch",
        "replace",
        "rewrite",
        "score",
        "scored",
        "scoring",
        "transform",
        "update",
    }
)
PLANNING_CODE_ARTIFACT_TERMS = frozenset(
    {
        "buffer",
        "buffers",
        "dataflow",
        "declaration",
        "declarations",
        "diff",
        "edit",
        "fragment",
        "fragments",
        "helper",
        "helpers",
        "include",
        "identifier",
        "identifiers",
        "patch",
        "preload",
        "stub",
        "stubs",
        "symbol",
        "symbols",
        "transform",
        "variable",
        "variables",
    }
)
PLANNING_DATAFLOW_ACTION_TERMS = frozenset(
    {
        "add",
        "coalesce",
        "consume",
        "double",
        "enable",
        "feed",
        "implement",
        "introduce",
        "load",
        "move",
        "pipeline",
        "prefetch",
        "process",
        "replace",
        "reuse",
        "route",
        "split",
        "stage",
        "store",
        "support",
    }
)
PLANNING_DATAFLOW_ARTIFACT_TERMS = frozenset(
    {
        "async",
        "buffer",
        "buffers",
        "concurrent",
        "cp",
        "dataflow",
        "load",
        "loads",
        "pipeline",
        "query",
        "reduction",
        "shared",
        "softmax",
        "stage",
        "staging",
        "store",
        "tile",
        "tiles",
        "tiling",
        "warp",
    }
)


@dataclass(frozen=True)
class CandidatePatchInspection:
    patch_text: str
    added_lines: tuple[str, ...]
    removed_lines: tuple[str, ...]
    added_text: str
    removed_text: str
    changed_paths: frozenset[str]

    @property
    def compact_patch(self) -> str:
        return re.sub(r"\s+", "", self.patch_text)

    @property
    def compact_added_text(self) -> str:
        return re.sub(r"\s+", "", self.added_text)

    @property
    def edits_cuda_source(self) -> bool:
        return any(path.endswith((".cu", ".cuh")) for path in self.changed_paths)


@dataclass(frozen=True)
class CandidatePatchPreflightTrack:
    name: str
    failure_class: str
    message: str
    detector: Callable[[CandidatePatchInspection], bool]


@dataclass(frozen=True)
class CandidatePatchAdvisoryTrack:
    name: str
    category: str
    message: str
    detector: Callable[[CandidatePatchInspection], bool]


TRANSFORM_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {
            "type": "string",
            "enum": sorted(STRUCTURED_TRANSFORM_STEP_OPS),
            "description": (
                "Primitive materialization step for a scoped semantic move; the step "
                "itself is not the optimization unit."
            ),
        },
        "path": {
            "type": "string",
            "description": (
                "Repo-relative candidate source path under candidates/. For op=batch this "
                "may be a default path for steps that omit path."
            ),
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
        "header": {
            "type": "string",
            "description": "Header for add_include, such as cuda_pipeline_primitives.h.",
        },
        "name": {
            "type": "string",
            "description": "Integer constant name for set_constexpr_int or add_int_to_python_set.",
        },
        "value": {
            "type": "integer",
            "description": "Integer value for set_constexpr_int or add_int_to_python_set.",
        },
    },
    "required": ["op", "path"],
    "additionalProperties": False,
}
TRANSFORM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **TRANSFORM_STEP_SCHEMA["properties"],
        "op": {
            "type": "string",
            "enum": sorted(STRUCTURED_TRANSFORM_OPS),
            "description": (
                "Scoped coherent semantic transform, or batch for an ordered bundle "
                "of primitive materialization steps."
            ),
        },
        "steps_json": {
            "type": "string",
            "description": (
                "For op=batch, a compact JSON array of primitive transform step objects. "
                "Each step has op, path, and the fields required by that op."
            ),
        },
    },
    "required": ["op"],
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
                "Scoped semantic move or inspection step tied to the hypothesis. If both "
                "edit channels are empty, this must start with 'No edit;' and describe "
                "only a bounded diagnostic."
            ),
        },
        "candidate_patch": {
            "type": "string",
            "description": (
                "Legacy raw git-style unified diff for one scoped candidate edit under "
                "candidates/, starting with 'diff --git', or empty string. Prefer "
                "candidate_transform for CUDA edits so the orchestrator materializes and "
                "preflights the patch instead of trusting generated hunks. Do not use "
                "markdown fences."
            ),
        },
        "edit_mode": {
            "type": "string",
            "enum": sorted(EDIT_MODES),
            "description": (
                "Explicit channel selection. Use transform when candidate_edit describes a code "
                "change represented by candidate_transform; legacy_patch only for non-CUDA raw "
                "diffs; no_edit only for bounded diagnostics with no source change."
            ),
        },
        "candidate_transform": {
            **TRANSFORM_SCHEMA,
            "type": ["object", "null"],
            "description": (
                "Required field. For edit_mode=transform, provide one coherent semantic "
                "transform or a scoped semantic batch of primitive materialization steps. "
                "For edit_mode=no_edit or edit_mode=legacy_patch, set this field to null "
                "instead of omitting it."
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
                "one of these subcommands: env, compile, profile, score. Do not include "
                "shell pipes, "
                "redirection, command chaining, git, rm, cat, head, or arbitrary shell."
            ),
        },
    },
    "required": [
        "hypothesis",
        "files_to_inspect",
        "candidate_edit",
        "candidate_patch",
        "edit_mode",
        "candidate_transform",
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
    edit_mode: str = "no_edit"
    candidate_patch: str = ""
    candidate_transform: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> VariationDecision:
        normalized_payload = {"candidate_patch": "", "candidate_transform": None, **payload}
        edit_mode_explicit = "edit_mode" in normalized_payload
        missing = [key for key in DECISION_SCHEMA["required"] if key not in normalized_payload]
        missing_without_edit_mode = [key for key in missing if key != "edit_mode"]
        if missing_without_edit_mode:
            raise ValueError(
                f"variation decision missing required keys: {', '.join(missing_without_edit_mode)}"
            )
        files = normalized_payload["files_to_inspect"]
        if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
            raise ValueError("files_to_inspect must be a list of strings")
        hypothesis = _require_string(normalized_payload, "hypothesis")
        candidate_edit = _require_string(normalized_payload, "candidate_edit")
        candidate_patch = _validate_candidate_patch(normalized_payload, "candidate_patch")
        if not edit_mode_explicit:
            normalized_payload["edit_mode"] = _infer_edit_mode(
                normalized_payload,
                candidate_patch=candidate_patch,
            )
        edit_mode = _validate_edit_mode(normalized_payload)
        raw_candidate_transform = normalized_payload.get("candidate_transform")
        if raw_candidate_transform is None and not candidate_patch.strip():
            raw_candidate_transform = _infer_candidate_transform_from_edit(
                candidate_edit,
                files_to_inspect=files,
            )
        candidate_transform = _validate_candidate_transform(raw_candidate_transform)
        if candidate_patch.strip() and candidate_transform is not None:
            raise ValueError(
                "candidate_patch and candidate_transform are mutually exclusive; "
                "use one edit channel"
            )
        _validate_edit_mode_payload(
            edit_mode,
            candidate_edit=candidate_edit,
            candidate_patch=candidate_patch,
            candidate_transform=candidate_transform,
            explicit=edit_mode_explicit,
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
        _validate_candidate_transform_matches_semantic_claim(
            candidate_transform,
            planning_text,
        )
        _validate_candidate_patch_not_self_rejected(edit_payload, planning_text)
        _validate_candidate_patch_preflight(candidate_patch)
        next_command = _validate_next_command(
            _require_string(normalized_payload, "next_command"),
            candidate_patch=candidate_patch,
            candidate_transform=candidate_transform,
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
            edit_mode=edit_mode,
            candidate_transform=candidate_transform,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "files_to_inspect": self.files_to_inspect,
            "candidate_edit": self.candidate_edit,
            "candidate_patch": self.candidate_patch,
            "edit_mode": self.edit_mode,
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


PayloadNormalizer = Callable[[dict[str, Any]], dict[str, Any]]


def parse_decision_text(
    text: str,
    *,
    normalize_payload: PayloadNormalizer | None = None,
) -> VariationDecision:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _recover_json_object(text)
        if payload is None:
            raise ValueError(f"agent returned malformed JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("agent JSON response must be an object")
    if normalize_payload is not None:
        payload = normalize_payload(payload)
    return VariationDecision.from_mapping(payload)


def parse_decision_response(
    response: Any,
    *,
    normalize_payload: PayloadNormalizer | None = None,
) -> VariationDecision:
    for block in getattr(response, "content", []):
        if _block_type(block) != "tool_use":
            continue
        if _block_value(block, "name") != DECISION_TOOL_NAME:
            continue
        payload = _block_value(block, "input")
        if not isinstance(payload, dict):
            raise ValueError("variation decision tool input must be an object")
        if normalize_payload is not None:
            payload = normalize_payload(payload)
        return VariationDecision.from_mapping(payload)

    text = _response_text(response)
    if text.strip():
        return parse_decision_text(text, normalize_payload=normalize_payload)
    raise ValueError(f"agent did not call {DECISION_TOOL_NAME}")


def request_variation_decision(
    *,
    lineage_summary: str,
    knowledge: str,
    attempt_history: str = "",
    repo_context: str = "",
    model: str = DEFAULT_AGENT_MODEL,
    normalize_payload: PayloadNormalizer | None = None,
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
        "timeout": _agent_request_timeout_s(),
    }
    return _request_valid_decision(client, kwargs, normalize_payload=normalize_payload)


def _agent_request_timeout_s() -> float:
    raw = os.environ.get(AGENT_REQUEST_TIMEOUT_ENV)
    if not raw:
        return DEFAULT_AGENT_REQUEST_TIMEOUT_S
    try:
        timeout = float(raw)
    except ValueError:
        return DEFAULT_AGENT_REQUEST_TIMEOUT_S
    if timeout <= 0.0:
        return DEFAULT_AGENT_REQUEST_TIMEOUT_S
    return timeout


def build_variation_prompt(
    *,
    knowledge: str,
    lineage_summary: str,
    attempt_history: str = "",
    repo_context: str = "",
) -> str:
    profile_available = not PROFILER_UNSUPPORTED_RUNTIME_MARKER.exists()
    bounded_command_list = (
        "env, compile, profile, score" if profile_available else "env, compile, score"
    )
    no_edit_command_text = (
        "a bounded score, profile, compile, or environment diagnostic"
        if profile_available
        else "a bounded score, compile, or environment diagnostic"
    )
    profile_flag_text = (
        "candidate profile uses the same score-shape flags and wraps the worker in "
        "Nsight Compute. "
        if profile_available
        else "candidate profile is unavailable in this runtime; use score for "
        "correctness, timing, and TFLOPS. "
    )
    context_section = f"\n\nLocal repo context:\n{repo_context}" if repo_context.strip() else ""
    attempt_section = (
        "\n\nRecent attempt history:\n"
        f"{attempt_history}\n"
        "Use this to avoid repeating failed or regressed transform families."
        if attempt_history.strip()
        else ""
    )
    return (
        "You are the AVO variation operator for an Ampere sm_86 attention kernel.\n"
        "Use FlashAttention-2/Ampere assumptions only. FA4/Blackwell strategies are invalid.\n"
        "Optimize toward realistic long-sequence BF16 attention workloads; current small "
        "candidate smoke shapes are safety fences for unsupported seeds, not the end target.\n"
        "Use one of exactly three explicit edit_mode values. Preferred edit mode: set "
        "edit_mode=\"transform\", provide candidate_transform as one scoped coherent "
        "semantic transformation or an ordered semantic batch under candidates/, and set "
        "candidate_patch to exactly the empty string \"\". "
        "Supported ops are add_include, replace_once, insert_before_once, "
        "insert_after_once, set_constexpr_int, and add_int_to_python_set; use op=batch "
        "with steps_json containing the materialization steps needed when one coherent "
        "candidate needs coordinated wrapper/kernel edits. The orchestrator "
        "materializes and preflights the patch. Make the smallest coherent "
        "transformation that preserves kernel invariants and can be validated against "
        "the hypothesis. Scoped means reviewable and recoverable, not the smallest "
        "possible textual edit; do not use a one-line constant edit as a stand-in for "
        "a dataflow, tiling, or scheduling change that it does not actually implement. "
        "The claimed semantic delta must be source-verifiable from the transform: if "
        "you claim fewer/reused loads, moved staging, different work mapping, or "
        "pipeline overlap, the steps must remove, replace, or relocate the relevant "
        "load/store/loop sites in the current source excerpts. "
        "Primitive steps are only the representation; do not submit support-only edits "
        "such as adding a header, unused helper, or unused buffer as a standalone "
        "candidate. If a CUDA idea cannot be "
        "expressed as an exact coherent transform over the current source excerpts, choose a "
        "smaller coherent transform unit or a no-edit diagnostic; do not describe a broad "
        "CUDA rewrite in candidate_edit without that exact transform. "
        "If recent attempt history contains 'Exact pending candidate_transform JSON', "
        "copy that JSON object verbatim into candidate_transform when scoring it; keep "
        "edit_mode=\"transform\" and candidate_patch=\"\" because compile-only patches are "
        "cleaned up before follow-up scoring. A no_edit score would score the unmodified "
        "seed, not the compiled transform. "
        "Legacy edit mode: set edit_mode=\"legacy_patch\" and provide candidate_patch as "
        "one scoped raw git-style unified diff under candidates/ starting with 'diff --git'. "
        "It may edit wrappers or other non-CUDA candidate files, but it must not edit "
        ".cu/.cuh kernel sources directly. No-edit mode: set edit_mode=\"no_edit\", "
        "candidate_patch is exactly the empty string \"\", candidate_edit starts with "
        "\"No edit; \", and next_command is only "
        f"{no_edit_command_text} for existing files. "
        "Do not include markdown fences or commentary in candidate_patch. "
        "Do not describe extending, updating, "
        "modifying, fixing, or implementing code unless candidate_transform or candidate_patch "
        "contains that change.\n"
        "Return exactly one decision. The next_command must be a single bounded command that "
        f"starts with 'avo' and uses only one of: {bounded_command_list}. "
        "Always include the candidate_transform field; use null outside transform mode. "
        "Use valid CLI flags: "
        "compile requires --source SOURCE.cu and --out-dir DIR; candidate score requires "
        f"--backend candidate and --candidate; {profile_flag_text}Use env only for "
        "CUDA/build environment "
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
    profile_available = not PROFILER_UNSUPPORTED_RUNTIME_MARKER.exists()
    profile_command_context = (
        "avo profile --backend candidate --candidate CANDIDATE.py ..."
        if profile_available
        else "avo profile is unavailable in this runtime"
    )
    profile_usage_context = (
        "Use avo profile only for bounded Nsight Compute diagnostics on candidate kernels "
        "when profiler evidence such as occupancy, scheduler behavior, or memory workload "
        "would change the next transform choice. Profiling may report unavailable if the "
        "current driver/container blocks CUPTI or performance counters."
        if profile_available
        else "Do not choose avo profile in this runtime; Nsight/CUPTI profiling is "
        "unavailable under the current Thunder-backed execution environment. Use score "
        "for correctness, timing, and TFLOPS, or compile a candidate_transform."
    )
    lines = [
        "Use only files that exist in this repository.",
        "Do not propose upstream FlashAttention csrc paths unless they are present locally.",
        "Available bounded commands: avo env; avo compile --source SOURCE.cu --out-dir DIR; "
        f"avo score --backend BACKEND ...; {profile_command_context}",
        "Use avo env only for CUDA/build environment diagnostics, not source-file inspection.",
        "The current CUDA/build environment is already recorded as stable "
        "(torch CUDA 13.0, nvcc CUDA 13.0, RTX A6000 sm_86, Anthropic key present); "
        "do not run no-edit avo env merely to confirm stability unless a recent "
        "build/environment failure gives a concrete reason.",
        "Use avo compile only for CUDA build/compilation diagnostics or to build-check a "
        "candidate_transform/candidate_patch, not source-file inspection.",
        "Use avo score only for correctness validation, timing samples, and TFLOPS. "
        "It does not report profiler metrics such as bandwidth, occupancy, scheduler "
        "stalls, instruction mix, or tensor-core utilization.",
        profile_usage_context,
        "Target workload is realistic long-sequence BF16 attention on sm_86: seq_lens "
        "4096/8192/16384/32768, total_tokens around 32768, num_heads around 16, "
        "head_dim 128, and both causal modes. Small candidate shapes are smoke fences, "
        "not the optimization objective.",
        "Preferred edit channel: candidate_transform, one scoped coherent semantic "
        "transformation or a scoped semantic batch under candidates/. Supported step ops: "
        "add_include, replace_once, insert_before_once, insert_after_once, set_constexpr_int, "
        "and add_int_to_python_set. Use op=batch with steps_json when wrapper and kernel "
        "caps must change together. Legacy "
        "candidate_patch raw diffs are allowed only for non-CUDA candidate files; "
        ".cu/.cuh kernel edits must use candidate_transform.",
        "Make the smallest coherent transformation that preserves invariants and can be "
        "validated. Primitive steps are not the objective, and fewer text edits are not "
        "better when the semantic move requires coordinated contract/dataflow changes; "
        "support-only edits such as adding a header, unused helper, or unused buffer must "
        "be part of a semantic batch that changes executable dataflow or a validation "
        "contract.",
        "The semantic delta must be source-verifiable: claims about fewer or reused "
        "loads, staging, work mapping, or pipeline overlap must correspond to exact "
        "transform steps that remove, replace, or relocate the relevant load/store/loop "
        "sites in the current source excerpts.",
        "If a CUDA idea is not representable as an exact coherent transform, shrink it to "
        "the smallest coherent transform unit or choose a no-edit diagnostic; do not "
        "describe a broad kernel rewrite without candidate_transform.",
        "Candidate interface: module defines attention(q, k, v, causal: bool).",
        "No-patch compiles of existing CUDA candidates are baseline diagnostics, not "
        "optimization steps; compile only when build-checking a materialized edit.",
        "Unpatched seed caps are safety fences, not search targets. Use exact small-shape "
        "caps only to avoid invalid no-edit scores; real optimization steps should move "
        "toward the target workload or explain a targeted compile diagnostic.",
        "Structural CUDA preflight tracks are class-oriented hard checks: edit-channel "
        "integrity, transform path/materialization, wrapper/kernel shape-contract "
        "consistency, WMMA contract validity, async pipeline stage lifecycle, "
        "shared-tile scope, symbol lifecycle, and disconnected "
        "skeleton/dataflow. Failed attempt classes can be promoted by the evolve loop "
        "into active hard tracks.",
        "CUDA transforms should preserve executable contracts, not just compile text: "
        "tensor-core fragment declarations must match Ampere-supported contracts, async "
        "copy changes should prefer aligned vector groups and real consumed dataflow, "
        "but copy-width preference alone is not a hard rejection when the transform is "
        "otherwise coherent and repairable; "
        "pipeline waits must match the number of committed stages, double-buffer stage "
        "indices must advance after consumption, staged shared-memory addresses must stay "
        "tile-local, declarations must have clear lifetimes, and shape graduation must "
        "update wrapper and kernel contracts together.",
        "After a shape-support compile succeeds, score the realistic target lane instead "
        "of spending additional no-edit steps on smoke-only shapes.",
    ]
    if candidates:
        lines.append("Candidate modules:")
        lines.extend(f"- {candidate}" for candidate in candidates)
    if cuda_sources:
        lines.append("CUDA candidate sources:")
        lines.extend(f"- {source}" for source in cuda_sources)
    preferred_command = _preferred_candidate_score_command(candidates)
    if preferred_command:
        lines.append(
            "Preferred target candidate score command after a successful shape-support "
            f"compile: {preferred_command}"
        )
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
    normalize_payload: PayloadNormalizer | None = None,
) -> VariationDecision:
    last_error: ValueError | None = None
    for attempt in range(attempts):
        request_kwargs = _decision_kwargs_with_feedback(kwargs, last_error)
        response = _request_decision_response(client, request_kwargs)
        try:
            return parse_decision_response(response, normalize_payload=normalize_payload)
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
            "hypothesis, files_to_inspect, candidate_edit, candidate_patch, edit_mode, "
            "expected_effect, risk, and next_command. Do not omit expected_effect, "
            "risk, or next_command even in no-edit diagnostic mode. "
        )
    if (
        "candidate_patch must be non-empty" in message
        or "candidate_transform or candidate_patch must be provided" in message
    ):
        return (
            "Choose exactly one valid edit_mode. Preferred edit mode: set "
            "edit_mode='transform', provide candidate_transform as one scoped semantic "
            "transform or coherent batch, and set candidate_patch to ''. Legacy edit mode: set "
            "edit_mode='legacy_patch' and provide candidate_patch as a raw git-style "
            "unified diff under candidates/ starting with 'diff --git'. No-edit mode: set "
            "edit_mode='no_edit', candidate_patch must be exactly '' and candidate_edit "
            "must start with 'No edit;' followed only by the "
            "bounded score/compile/env diagnostic to run. "
            "Do not mention fixing, extending, updating, modifying, or implementing code in "
            "no-edit mode. If this is a follow-up score for a compiled transform, include "
            "the exact candidate_transform object from the follow-up signal. Do not "
            "restate a broad CUDA change in candidate_edit without an exact "
            "candidate_transform; reduce it to the smallest coherent semantic transform "
            "or a bounded coherent materialization batch over current source excerpts. "
            "Small means scoped, reviewable, recoverable, and tied to a clear hypothesis; "
            "it does not mean the smallest possible textual edit. Support-only "
            "edits such as adding a header or unused helper are not valid standalone "
            "optimization candidates. "
        )
    if "candidate_transform semantic mismatch" in message:
        return (
            "Make the transform match the semantic claim. A constant/set transform may "
            "retune an existing invariant or validation contract, but it must not be used "
            "as a proxy for a dataflow, tiling, staging, or scheduling change. Either "
            "narrow the hypothesis to the constant change being made, or provide one "
            "coherent candidate_transform batch that implements the claimed executable "
            "behavior and preserves its invariants. The semantic delta must be "
            "source-verifiable: if you claim fewer/reused loads, moved staging, "
            "different work mapping, or overlap, the transform must remove, replace, "
            "or relocate the relevant current load/store/loop sites. "
        )
    if "candidate_patch and candidate_transform are mutually exclusive" in message:
        return (
            "Choose one edit channel only. For preferred structured-transform mode, set "
            "candidate_transform to the operation object or batch object and set "
            "candidate_patch to exactly the empty string ''. For legacy raw-diff mode, "
            "set candidate_patch to the "
            "unified diff and omit candidate_transform. "
        )
    if "no-edit mode but includes an edit payload" in message:
        return (
            "No-edit mode cannot include candidate_transform or candidate_patch. Either "
            "remove the edit payload and run only the bounded diagnostic, or remove the "
            "'No edit;' prefix and describe the structured transform/raw diff as the edit. "
        )
    if "must not edit CUDA source files directly" in message:
        return (
            "Do not use raw candidate_patch for .cu or .cuh files. Express the CUDA edit "
            "as candidate_transform: one operation, or op=batch when coordinated "
            "wrapper/kernel changes are required. Each step must be representable by "
            "add_include, replace_once, insert_before_once, insert_after_once, "
            "set_constexpr_int, or add_int_to_python_set. Primitive steps must compose "
            "into a coherent semantic move; add_include is support-only and cannot stand "
            "alone. "
            "Raw candidate_patch is only for non-CUDA candidate files. "
        )
    if "structural preflight track" in message:
        return (
            "Treat the rejected edit as a failed transform family, not as a phrase to patch "
            "around. Choose a coherent candidate_transform operation or batch that changes "
            "the claimed executable behavior and avoids the same structural class, or "
            "switch to a bounded diagnostic that gives new information for a different "
            "transform family. "
        )
    if "support-only" in message:
        return (
            "Make the smallest coherent transformation that preserves invariants and can "
            "be validated. Do not return a header-only, helper-only, or unused-buffer-only "
            "candidate. If add_include is needed, put it in an op=batch with the dataflow "
            "or validation-contract change that actually uses it. "
        )
    if "known invalid by the decision itself" in message:
        return (
            "The decision classified its own edit as invalid. Pick a different transform "
            "family or return a complete structured candidate_transform whose hypothesis, "
            "effect, and risk describe an executable edit instead of a known compile, "
            "correctness, skeleton, or incomplete-edit failure. "
        )
    if "--out-dir must be under: build" in message:
        return (
            "For compile checks, set --out-dir to a repo-relative build subdirectory such "
            "as build/mma_probe. Do not write compiler outputs under candidates/ or beside "
            "source files. "
        )
    if "recorded unpatched MMA seed score" in message:
        return (
            "Do not retry a no-edit score that is already in lineage. Do not return "
            "candidate_edit starting with 'No edit;' for this correction. To score the "
            "MMA candidate again, set edit_mode='transform', set candidate_patch to '', "
            "and provide candidate_transform as one exact operation or a scoped "
            "wrapper/kernel batch that makes a real kernel-structure change. Raw "
            "candidate_patch cannot edit CUDA kernel sources. Otherwise choose a "
            "diagnostic that provides new information for a different transform family. "
        )
    if "candidate_transform batch steps must contain 1 to" in message:
        return (
            "Return a bounded candidate_transform batch with 1 to "
            f"{MAX_TRANSFORM_BATCH_STEPS} primitive operations, or use a single "
            "non-batch candidate_transform operation. Empty batches are not edits; "
            "oversized batches must be split into the smallest coherent semantic move "
            "that preserves invariants and can be compiled or scored. Keep "
            "candidate_patch exactly ''. "
        )
    if "below the current accepted seq" in message and "validation lane" in message:
        return (
            f"Do not spend no-edit score steps below the accepted seq{MMA_ACCEPTED_VALIDATION_SEQ} "
            "lane. Score the current MMA seed at the accepted lane or use a structured "
            "transform batch that moves the wrapper and kernel toward larger "
            "long-sequence workloads. "
        )
    if "patched MMA shape extension beyond the current smoke cap" in message:
        return (
            "For larger MMA scores, use a structured transform batch that changes the "
            "wrapper cap and the kernel cap/dataflow together. Wrapper-only cap edits "
            "are not enough to graduate out of the smoke shape. "
        )
    if "scores MMA seq_lens beyond the transformed cap" in message:
        return (
            "Keep the score workload within the cap expressed by the structured transform. "
            "If scoring larger sequence lengths, first extend both kMaxSeqLen and the "
            "wrapper sequence set to cover every requested seq_len. "
        )
    if "recorded no-patch warp-row seed score" in message:
        return (
            "Do not retry a no-edit score of a smoke-only seed workload that was already "
            "recorded. That diagnostic already produced its signal; use edit mode with "
            "candidate_transform or a legacy non-CUDA candidate_patch before scoring "
            "that seed family again. "
        )
    if "recorded environment stability diagnostic" in message:
        return (
            "Do not spend a loop step on avo env just to reconfirm the already-recorded "
            "CUDA/build environment. Use avo env only when the decision cites a concrete "
            "recent build or environment failure such as a CUDA version mismatch, missing "
            "compiler, missing package, or extension-build error. "
        )
    if "planner-interface failure" in message:
        return (
            "Do not spend a loop step on avo env for planner-interface or schema failures. "
            "Return a valid candidate_transform for a representable CUDA edit, or choose a "
            "no-edit compile/score diagnostic that directly informs the kernel search. "
        )
    if "avo env cannot inspect source files" in message:
        return (
            "Do not use avo env for source inspection. The planner already receives local "
            "candidate excerpts in repo context; choose a structured candidate_transform, "
            "a compile/score diagnostic tied to a concrete source change, or a supported "
            "CUDA/build environment diagnostic. "
        )
    if "recorded no-patch compile diagnostic" in message:
        return (
            "Do not retry a no-edit compile of an already-recorded candidate source. If "
            "you are compile-checking a code change, include candidate_transform with a "
            "single operation or scoped batch and keep candidate_patch as ''. For an "
            "integer constant change, use set_constexpr_int with path, name, and value. "
            "If you are following up a successful compile-only transform, score that exact "
            "candidate_transform; do not score or compile the unmodified seed again. "
        )
    if "score cannot collect profiler metrics" in message:
        return (
            "Do not claim that avo score can provide profiler metrics. It reports "
            "correctness, timing, and TFLOPS only. Either use it to validate a concrete "
            "candidate_transform, or use avo profile for a bounded Nsight Compute diagnostic "
            "on a candidate kernel. Do not ask score for bandwidth, occupancy, scheduler, "
            "instruction-mix, roofline, or tensor-core utilization evidence. "
        )
    if "profile is unavailable" in message:
        return (
            "Do not choose avo profile in this runtime. Use avo score for correctness, "
            "timing, and TFLOPS validation, or propose a candidate_transform with a "
            "compile/score validation path. "
        )
    if "scalar BF16 __pipeline_memcpy_async" in message:
        return (
            "Prefer vector-group async-copy dataflow over scalar BF16 async-copy calls, "
            "but keep broader async-copy hypotheses in scoped candidate_transform batches "
            "so compile/score and compile repair can validate concrete CUDA errors. "
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


def _infer_edit_mode(payload: dict[str, Any], *, candidate_patch: str) -> str:
    if candidate_patch.strip():
        return "legacy_patch"
    if isinstance(payload.get("candidate_transform"), dict):
        return "transform"
    candidate_edit = payload.get("candidate_edit")
    if isinstance(candidate_edit, str) and _candidate_edit_requires_patch(candidate_edit):
        return "transform"
    return "no_edit"


def _validate_edit_mode(payload: dict[str, Any]) -> str:
    value = payload.get("edit_mode")
    if not isinstance(value, str) or value not in EDIT_MODES:
        allowed = ", ".join(sorted(EDIT_MODES))
        raise ValueError(f"edit_mode must be one of: {allowed}")
    return value


def _validate_edit_mode_payload(
    edit_mode: str,
    *,
    candidate_edit: str,
    candidate_patch: str,
    candidate_transform: dict[str, Any] | None,
    explicit: bool,
) -> None:
    if edit_mode == "transform":
        if candidate_patch.strip():
            raise ValueError("edit_mode transform requires candidate_patch to be empty")
        if candidate_transform is None:
            raise ValueError(
                "edit_mode transform requires candidate_transform; "
                "candidate_transform or candidate_patch must be provided when candidate_edit "
                "describes a code change; serialize the coherent semantic move as a "
                "structured transform instead of prose; "
                f"candidate_edit was {_validation_excerpt(candidate_edit)!r}"
            )
        return
    if edit_mode == "legacy_patch":
        if candidate_transform is not None:
            raise ValueError("edit_mode legacy_patch must omit candidate_transform")
        if not candidate_patch.strip():
            raise ValueError("edit_mode legacy_patch requires non-empty candidate_patch")
        return
    if candidate_patch.strip() or candidate_transform is not None:
        raise ValueError("edit_mode no_edit cannot include candidate_transform or candidate_patch")
    if not explicit:
        return
    if not candidate_edit.lstrip().lower().startswith("no edit;"):
        raise ValueError("edit_mode no_edit requires candidate_edit to start with 'No edit;'")


def _validate_candidate_transform(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("candidate_transform must be an object when provided")
    op = value.get("op")
    if not isinstance(op, str) or op not in STRUCTURED_TRANSFORM_OPS:
        allowed = ", ".join(sorted(STRUCTURED_TRANSFORM_OPS))
        raise ValueError(f"candidate_transform op must be one of: {allowed}")
    if op == "batch":
        extra = set(value) - {"op", "path", "steps", "steps_json"}
        if extra:
            raise ValueError(
                "candidate_transform batch contains unsupported keys: "
                + ", ".join(sorted(extra))
            )
        batch_path = value.get("path")
        if batch_path is not None:
            if not isinstance(batch_path, str):
                raise ValueError("candidate_transform batch path must be a string")
            _validate_candidate_transform_path(batch_path)
        steps = _candidate_transform_batch_steps(value)
        if (
            not isinstance(steps, list)
            or not steps
            or len(steps) > MAX_TRANSFORM_BATCH_STEPS
        ):
            raise ValueError(
                "candidate_transform batch steps must contain 1 to "
                f"{MAX_TRANSFORM_BATCH_STEPS} operations"
            )
        transform = {
            "op": "batch",
            "steps": [
                _validate_candidate_transform_step(
                    _candidate_transform_batch_step_with_default_path(step, batch_path),
                    label=f"batch step {index}",
                )
                for index, step in enumerate(steps)
            ],
        }
        _validate_candidate_transform_semantic_coherence(transform)
        return transform
    transform = _validate_candidate_transform_step(value)
    _validate_candidate_transform_semantic_coherence(transform)
    return transform


def _candidate_transform_batch_steps(value: dict[str, Any]) -> list[Any]:
    steps = value.get("steps")
    if isinstance(steps, list):
        return steps
    steps_json = value.get("steps_json")
    if not isinstance(steps_json, str) or not steps_json.strip():
        raise ValueError("candidate_transform batch requires steps_json")
    try:
        parsed = json.loads(steps_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"candidate_transform batch steps_json is invalid JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError("candidate_transform batch steps_json must be a JSON array")
    return parsed


def _candidate_transform_batch_step_with_default_path(
    step: Any,
    batch_path: Any,
) -> Any:
    if not isinstance(step, dict) or "path" in step or batch_path is None:
        return step
    return {"path": batch_path, **step}


def _validate_candidate_transform_semantic_coherence(transform: dict[str, Any]) -> None:
    steps = transform.get("steps") if transform.get("op") == "batch" else [transform]
    if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
        return
    step_ops = {str(step.get("op") or "") for step in steps}
    if step_ops and step_ops <= SUPPORT_ONLY_TRANSFORM_OPS:
        raise ValueError(
            "candidate_transform is support-only; make the smallest coherent semantic "
            "transformation that preserves invariants and can be validated. Support-only "
            "steps such as add_include must be part of a batch with a real dataflow or "
            "validation-contract change."
        )


def _validate_candidate_transform_matches_semantic_claim(
    candidate_transform: dict[str, Any] | None,
    planning_text: str,
) -> None:
    if candidate_transform is None:
        return
    load_reduction_mismatch = _candidate_transform_load_reduction_mismatch(
        candidate_transform,
        planning_text,
    )
    if load_reduction_mismatch is not None:
        operand, claim = load_reduction_mismatch
        raise ValueError(
            "candidate_transform semantic mismatch: planning text claims reduced or "
            f"reused {operand} loads, but the structured transform does not reduce that "
            "operand's load sites or move them out of the repeated loop. "
            f"Matched claim: {claim!r}"
        )
    if not _candidate_transform_is_contract_only(candidate_transform):
        return
    claim = _planning_text_dataflow_claim(planning_text)
    if claim is None:
        return
    raise ValueError(
        "candidate_transform semantic mismatch: contract-only transforms such as "
        "set_constexpr_int/add_int_to_python_set may retune an existing invariant or "
        "shape contract, but they do not implement new dataflow, tiling, staging, or "
        f"scheduling behavior claimed by planning text: {claim!r}"
    )


LOAD_REDUCTION_OPERAND_ALIASES: dict[str, tuple[str, ...]] = {
    "Q": ("q", "q tile", "q fragment", "q fragments", "q_frags"),
    "K": ("k", "k tile", "k fragment"),
    "V": ("v", "v tile", "v fragment"),
    "probability": (
        "probability",
        "probabilities",
        "probability tile",
        "probability fragment",
        "probability_frag",
    ),
}


def _candidate_transform_load_reduction_mismatch(
    transform: dict[str, Any],
    planning_text: str,
) -> tuple[str, str] | None:
    for operand, claim in _planning_text_load_reduction_claims(planning_text).items():
        if _candidate_transform_supports_load_reduction_claim(transform, operand):
            continue
        return operand, claim
    return None


def _planning_text_load_reduction_claims(planning_text: str) -> dict[str, str]:
    normalized_text = planning_text.lower().replace("-", " ")
    windows = [
        " ".join(window.split())
        for window in re.split(r"(?<=[.;!?])\s+|\n+", normalized_text)
        if window.strip()
    ] or [" ".join(normalized_text.split())]
    claims: dict[str, str] = {}
    for window in windows:
        if _planning_window_is_historical_failure_context(
            window
        ) or _planning_window_describes_existing_state(window):
            continue
        if not _window_claims_load_reduction(window):
            continue
        for operand, aliases in LOAD_REDUCTION_OPERAND_ALIASES.items():
            if not any(_contains_operand_alias(window, alias) for alias in aliases):
                continue
            claims.setdefault(operand, _validation_excerpt(window))
    return claims


def _window_claims_load_reduction(window: str) -> bool:
    load_terms = r"(?:loads?|traffic|memory traffic|global memory|global loads?)"
    reduction_terms = (
        r"(?:reduce|reduces|reduced|reducing|eliminate|eliminates|avoid|avoids|"
        r"reuse|reuses|reused|hoist|hoists|hoisted)"
    )
    return bool(
        re.search(rf"\b{reduction_terms}\b.{0,80}\b{load_terms}\b", window)
        or re.search(rf"\b{load_terms}\b.{0,80}\b{reduction_terms}\b", window)
        or re.search(r"\bload(?:s|ing)?\b.{0,80}\bonce\b", window)
    )


def _contains_operand_alias(text: str, alias: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", text))


def _candidate_transform_supports_load_reduction_claim(
    transform: dict[str, Any],
    operand: str,
) -> bool:
    for step in _candidate_transform_steps(transform):
        if str(step.get("op") or "") != "replace_once":
            continue
        before = str(step.get("find") or "")
        after = str(step.get("replace") or "")
        before_count = _operand_load_site_count(before, operand)
        after_count = _operand_load_site_count(after, operand)
        if before_count > 0 and after_count < before_count:
            return True
        if before_count > 0 and after_count == before_count and _moves_operand_load_out_of_loop(
            before,
            after,
            operand,
        ):
            return True
    return False


def _candidate_transform_steps(transform: dict[str, Any]) -> list[dict[str, Any]]:
    steps = transform.get("steps") if transform.get("op") == "batch" else [transform]
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)]


def _operand_load_site_count(text: str, operand: str) -> int:
    compact = re.sub(r"\s+", "", text)
    if operand == "probability":
        return compact.count("load_matrix_sync(probability_frag,")
    lower_operand = operand.lower()
    return len(
        re.findall(
            rf"load_matrix_sync\({re.escape(lower_operand)}(?:_frag|_frags)?(?:\[[^\]]+\])?,",
            compact,
        )
    )


def _moves_operand_load_out_of_loop(before: str, after: str, operand: str) -> bool:
    before_load = _first_operand_load_index(before, operand)
    after_load = _first_operand_load_index(after, operand)
    if before_load is None or after_load is None:
        return False
    return _has_loop_before_index(before, before_load) and not _has_loop_before_index(
        after,
        after_load,
    )


def _first_operand_load_index(text: str, operand: str) -> int | None:
    compact = re.sub(r"\s+", "", text)
    if operand == "probability":
        needle = "load_matrix_sync(probability_frag,"
        index = compact.find(needle)
        return index if index >= 0 else None
    lower_operand = operand.lower()
    match = re.search(
        rf"load_matrix_sync\({re.escape(lower_operand)}(?:_frag|_frags)?(?:\[[^\]]+\])?,",
        compact,
    )
    return match.start() if match else None


def _has_loop_before_index(text: str, index: int) -> bool:
    compact = re.sub(r"\s+", "", text)
    return "for(" in compact[:index]


def _candidate_transform_is_contract_only(transform: dict[str, Any]) -> bool:
    non_support_ops = [
        str(step.get("op") or "")
        for step in _candidate_transform_steps(transform)
        if str(step.get("op") or "") not in SUPPORT_ONLY_TRANSFORM_OPS
    ]
    return bool(non_support_ops) and all(
        op in CONTRACT_ONLY_TRANSFORM_OPS for op in non_support_ops
    )


def _planning_text_dataflow_claim(planning_text: str) -> str | None:
    normalized_text = planning_text.lower().replace("-", " ")
    windows = [
        " ".join(window.split())
        for window in re.split(r"(?<=[.;!?])\s+|\n+", normalized_text)
        if window.strip()
    ]
    if not windows:
        windows = [" ".join(normalized_text.split())]
    for window in windows:
        if _planning_window_is_historical_failure_context(
            window
        ) or _planning_window_describes_existing_state(window):
            continue
        words = set(re.findall(r"[a-z_]+", window))
        if (
            words & PLANNING_DATAFLOW_ACTION_TERMS
            and words & PLANNING_DATAFLOW_ARTIFACT_TERMS
        ):
            return _validation_excerpt(window)
    return None


def _validate_candidate_transform_step(value: Any, *, label: str = "operation") -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"candidate_transform {label} must be an object")
    op = value.get("op")
    path = value.get("path")
    if not isinstance(op, str) or op not in STRUCTURED_TRANSFORM_STEP_OPS:
        allowed = ", ".join(sorted(STRUCTURED_TRANSFORM_STEP_OPS))
        raise ValueError(f"candidate_transform {label} op must be one of: {allowed}")
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"candidate_transform {label} path must be a non-empty string")
    _validate_candidate_transform_path(path)
    extra = set(value) - set(TRANSFORM_STEP_SCHEMA["properties"])
    if extra:
        raise ValueError(
            f"candidate_transform {label} contains unsupported keys: "
            + ", ".join(sorted(extra))
        )
    if op == "replace_once":
        _require_transform_string(value, "find")
        _require_transform_string(value, "replace")
    elif op in {"insert_before_once", "insert_after_once"}:
        _require_transform_string(value, "anchor")
        _require_transform_string(value, "text")
    elif op == "add_include":
        _require_transform_string(value, "header")
        if PurePosixPath(path).suffix not in (".cpp", ".cu", ".cuh", ".h", ".hpp"):
            raise ValueError("candidate_transform add_include path must be a C++/CUDA source file")
    elif op in {"set_constexpr_int", "add_int_to_python_set"}:
        _require_transform_string(value, "name")
        if not isinstance(value.get("value"), int):
            raise ValueError(f"candidate_transform {label} value must be an integer")
    return dict(value)


def _infer_candidate_transform_from_edit(
    candidate_edit: str,
    *,
    files_to_inspect: list[str] | None = None,
) -> dict[str, Any] | None:
    candidate_paths = re.findall(
        r"\b(candidates/[A-Za-z0-9_./-]+\.(?:cu|cuh|cpp|h|hpp|py))\b",
        candidate_edit,
    )
    candidate_paths.extend(files_to_inspect or [])
    candidate_paths.extend(_candidate_source_alias_paths_from_edit(candidate_edit))
    candidate_paths = [path for path in candidate_paths if path.startswith("candidates/")]
    candidate_paths = list(dict.fromkeys(candidate_paths))
    if not candidate_paths:
        return None
    source_suffixes = {".cu", ".cuh", ".cpp", ".h", ".hpp"}
    source_path = next(
        (path for path in candidate_paths if Path(path).suffix in source_suffixes),
        None,
    )
    if source_path is None:
        return None
    steps: list[dict[str, Any]] = []
    include_header = _infer_include_header_from_edit(candidate_edit)
    literal_replace = _infer_backtick_replace_from_edit(candidate_edit)
    if literal_replace is not None:
        if include_header is not None:
            steps.append({"op": "add_include", "path": source_path, "header": include_header})
        steps.append(
            {
                "op": "replace_once",
                "path": source_path,
                "find": literal_replace[0],
                "replace": literal_replace[1],
            }
        )
        return steps[0] if len(steps) == 1 else {"op": "batch", "steps": steps}
    const_match = _infer_integer_constant_change(candidate_edit)
    if const_match is None:
        const_alias = _infer_integer_constant_alias_change(candidate_edit)
    else:
        const_alias = (const_match.group("name"), int(const_match.group("value")))
    if const_alias is None:
        return None
    if include_header is not None:
        steps.append({"op": "add_include", "path": source_path, "header": include_header})
    steps.append(
        {
            "op": "set_constexpr_int",
            "path": source_path,
            "name": const_alias[0],
            "value": const_alias[1],
        }
    )
    replace_match = re.search(
        r"\breplace\s+(?P<find>(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{[^}]+\})"
        r"\s+with\s+(?P<replace>(?P=name)\s*=\s*\{[^}]+\})",
        candidate_edit,
    )
    wrapper_path = next((path for path in candidate_paths if path.endswith(".py")), None)
    if replace_match is not None and wrapper_path is not None:
        steps.append(
            {
                "op": "replace_once",
                "path": wrapper_path,
                "find": replace_match.group("find"),
                "replace": replace_match.group("replace"),
            }
        )
    elif wrapper_path is not None:
        set_name = _infer_python_set_name_from_edit(candidate_edit)
        if set_name is None:
            return steps[0]
        add_match = re.search(r"\badd(?:ing)?\s+(?P<value>[-+]?\d+)\b", candidate_edit)
        value = int(add_match.group("value")) if add_match is not None else steps[0]["value"]
        steps.append(
            {
                "op": "add_int_to_python_set",
                "path": wrapper_path,
                "name": set_name,
                "value": value,
            }
        )
    if len(steps) == 1:
        return steps[0]
    return {"op": "batch", "steps": steps}


def _candidate_source_alias_paths_from_edit(candidate_edit: str) -> list[str]:
    normalized = " ".join(candidate_edit.lower().replace("-", " ").split())
    paths: list[str] = []
    if "kernel" not in normalized:
        return paths
    if "mma" in normalized:
        paths.append(MMA_KERNEL_SOURCE)
    if "warp row" in normalized or "warp rows" in normalized:
        paths.append("candidates/cuda_warp_rows_attention/attention_kernel.cu")
    if "tiled" in normalized:
        paths.append("candidates/cuda_tiled_attention/attention_kernel.cu")
    if "naive" in normalized:
        paths.append("candidates/cuda_naive_attention/attention_kernel.cu")
    return paths


def _infer_include_header_from_edit(candidate_edit: str) -> str | None:
    header_pattern = r"(?P<header><[^>\n]+>|\"[^\"\n]+\"|[A-Za-z0-9_./+-]+\.(?:cuh|hpp|hh|h))"
    patterns = (
        rf"\b(?:add|insert)\s+(?:the\s+)?(?:#include\s+)?{header_pattern}\s+"
        r"(?:header|include)\b",
        rf"\b(?:add|insert)\s+(?:the\s+)?include\s+(?:for\s+)?{header_pattern}\b",
        rf"\b#include\s+{header_pattern}",
    )
    for pattern in patterns:
        match = re.search(pattern, candidate_edit, flags=re.IGNORECASE)
        if match is not None:
            return match.group("header")
    return None


def _infer_backtick_replace_from_edit(candidate_edit: str) -> tuple[str, str] | None:
    patterns = (
        r"\breplace\b.*?`(?P<find>[^`]+)`.*?\bwith\s+`(?P<replace>[^`]+)`",
        r"\bchange\b.*?`(?P<find>[^`]+)`.*?\bto\s+`(?P<replace>[^`]+)`",
    )
    for pattern in patterns:
        match = re.search(pattern, candidate_edit, flags=re.IGNORECASE | re.DOTALL)
        if match is None:
            continue
        find = match.group("find").strip()
        replacement = match.group("replace").strip()
        if _is_exact_transform_snippet(find) and _is_exact_transform_snippet(replacement):
            return find, replacement
    return None


def _is_exact_transform_snippet(value: str) -> bool:
    return bool(value and "..." not in value and "…" not in value)


def _infer_integer_constant_change(candidate_edit: str) -> re.Match[str] | None:
    patterns = (
        r"\b(?:change|changing|decrease|decreasing|extend|extending|increase|"
        r"increasing|narrow|narrowing|retune|retuning|set|setting|update|updating|"
        r"widen|widening)\s+"
        r"(?:the\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
        r"(?:\s+(?:constant|cap|value))?"
        r"(?:\s+from\s+[-+]?\d+)?\s*(?:to|=)\s*(?P<value>[-+]?\d+)\b",
        r"\b(?:change|changing|decrease|decreasing|extend|extending|increase|"
        r"increasing|narrow|narrowing|retune|retuning|set|setting|update|updating|"
        r"widen|widening)\b"
        r"(?:(?!\b(?:to|=)\b).){0,96}?"
        r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+(?:constant|cap|value)"
        r"(?:\s+from\s+[-+]?\d+)?\s*(?:to|=)\s*(?P<value>[-+]?\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, candidate_edit, flags=re.IGNORECASE)
        if match is not None:
            return match
    return None


def _infer_integer_constant_alias_change(candidate_edit: str) -> tuple[str, int] | None:
    patterns = (
        r"\b(?:change|changing|decrease|decreasing|increase|increasing|retune|"
        r"retuning|set|setting|update|updating)\s+(?:the\s+)?(?:block\s+)?thread\s+count"
        r"(?:\s+from\s+[-+]?\d+)?\s*(?:to|=)\s*(?P<value>[-+]?\d+)\b",
        r"\b(?:change|changing|decrease|decreasing|increase|increasing|retune|"
        r"retuning|set|setting|update|updating)\s+(?:threads\s+per\s+block|block\s+threads)"
        r"(?:\s+from\s+[-+]?\d+)?\s*(?:to|=)\s*(?P<value>[-+]?\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, candidate_edit, flags=re.IGNORECASE)
        if match is not None:
            return "kThreads", int(match.group("value"))
    return None


def _infer_python_set_name_from_edit(candidate_edit: str) -> str | None:
    direct_match = re.search(
        r"\b(?:to|into|in|update|updating)\s+(?P<name>[A-Z][A-Z0-9_]*_[A-Z0-9_]+)\b",
        candidate_edit,
    )
    if direct_match is not None:
        return direct_match.group("name")
    names = re.findall(r"\b[A-Z][A-Z0-9_]*_[A-Z0-9_]+\b", candidate_edit)
    return names[0] if len(set(names)) == 1 else None


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
        "candidate_transform or candidate_patch must be provided when candidate_edit "
        "describes a code change; "
        f"candidate_edit was {_validation_excerpt(candidate_edit)!r}"
    )


def _validate_candidate_patch_not_self_rejected(
    candidate_patch: str,
    planning_text: str,
) -> None:
    if not candidate_patch.strip():
        return
    risk = _planning_text_risk(planning_text)
    if risk is None:
        return
    risk_class, excerpt = risk
    raise ValueError(
        "candidate_patch is described as known invalid by the decision itself; "
        f"planning risk class {risk_class!r} matched {excerpt!r}. "
        "Return a corrected transform/patch or choose no-edit mode."
    )


def _planning_text_risk(planning_text: str) -> tuple[str, str] | None:
    normalized_text = planning_text.lower().replace("-", " ")
    if not normalized_text.strip():
        return None
    windows = [
        " ".join(window.split())
        for window in re.split(r"(?<=[.;!?])\s+|\n+", normalized_text)
        if window.strip()
    ]
    if not windows:
        windows = [" ".join(normalized_text.split())]
    for window in windows:
        if _planning_window_is_historical_failure_context(window):
            continue
        if _planning_window_predicts_preflight_rejection(window):
            return "predicted_structural_preflight", _validation_excerpt(window)
        if _planning_window_predicts_compile_failure(window):
            return "predicted_compile_failure", _validation_excerpt(window)
        if _planning_window_predicts_correctness_failure(window):
            return "predicted_correctness_failure", _validation_excerpt(window)
        if not _planning_window_mentions_edit_surface(window):
            continue
        if _planning_window_self_rejects_edit(window):
            return "incomplete_or_malformed_edit", _validation_excerpt(window)
        if _planning_window_describes_no_effect_skeleton(window):
            return "no_effect_or_skeleton", _validation_excerpt(window)
        if _planning_window_describes_incomplete_edit(window):
            return "incomplete_or_malformed_edit", _validation_excerpt(window)
    return None


def _planning_window_mentions_edit_surface(text: str) -> bool:
    words = set(re.findall(r"[a-z_]+", text))
    return bool(words & PLANNING_EDIT_TERMS) or bool(words & PLANNING_CODE_ARTIFACT_TERMS)


def _planning_window_describes_no_effect_skeleton(text: str) -> bool:
    words = set(re.findall(r"[a-z_]+", text))
    has_skeleton_artifact = bool(
        words
        & {
            "buffer",
            "buffers",
            "fragment",
            "fragments",
            "helper",
            "helpers",
            "preload",
            "skeleton",
            "stub",
            "stubs",
        }
    )
    has_disconnected_signal = bool(
        re.search(
            r"\b(?:unused|empty|not\s+(?:wired|called|consumed)|does\s+not\s+"
            r"(?:affect|consume|improve)|cannot\s+(?:affect|improve))\b",
            text,
        )
    )
    return (
        "no effect" in text
        or "no-effect" in text
        or ("compile only" in text and "does not affect" in text)
        or ("compile only" in text and has_skeleton_artifact)
        or (has_skeleton_artifact and has_disconnected_signal)
        or bool(re.search(r"\b(?:must not|do not)\s+be\s+scored\b", text))
    )


def _planning_window_self_rejects_edit(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:do\s+not|don't|must\s+not|should\s+not)\s+"
            r"(?:apply|compile|execute|score|use)\b.{0,80}\b"
            r"(?:diff|edit|patch|transform)\b",
            text,
        )
        or re.search(
            r"\b(?:diff|edit|patch|transform)\b.{0,80}\b"
            r"(?:do\s+not|don't|must\s+not|should\s+not)\s+"
            r"(?:apply|compile|execute|score|use)\b",
            text,
        )
    )


def _planning_window_is_historical_failure_context(text: str) -> bool:
    if _planning_window_describes_current_attempt_failure(text):
        return False
    has_history_marker = re.search(
        r"\b(?:previous|prior|earlier|last|historical)\b",
        text,
    )
    has_failure_signal = re.search(
        r"\b(?:failed|failure|invalid|malformed|undefined|undeclared|"
        r"ambiguous|rejected|regressed|regression)\b",
        text,
    )
    if not has_history_marker or not has_failure_signal:
        return False
    return bool(
        re.match(r"^(?:the\s+)?(?:previous|prior|earlier|last|historical)\b", text)
        or re.search(
            r"\b(?:request|history|attempt|summary|feedback|evidence|record|"
            r"recorded|reported|shows?|showed|observed)\b.{0,100}\b"
            r"(?:previous|prior|earlier|last|historical)\b",
            text,
        )
        or re.search(
            r"\b(?:previous|prior|earlier|last|historical)\b.{0,100}\b"
            r"(?:request|history|attempt|summary|feedback|evidence|record|"
            r"recorded|reported|shows?|showed|observed)\b",
            text,
        )
    )


def _planning_window_describes_existing_state(text: str) -> bool:
    existing_subject = (
        r"(?:(?:current|existing|baseline)\s+"
        r"(?:kernel|seed|implementation|state|dataflow)|"
        r"(?:accepted|previous|prior)\s+"
        r"(?:kernel|seed|candidate|implementation|state|dataflow))"
    )
    if re.search(rf"\b(?:the\s+)?{existing_subject}\b", text):
        return True
    proposal_terms = r"(?:candidate|change|edit|patch|proposal|proposed|transform)"
    if re.search(rf"\b(?:this|the)\s+{proposal_terms}\b", text):
        return False
    return "already" in text


def _planning_window_describes_incomplete_edit(text: str) -> bool:
    if _planning_window_is_conditional_execution_risk(text):
        return False
    words = set(re.findall(r"[a-z_]+", text))
    has_incomplete_signal = bool(
        words
        & {
            "duplicate",
            "incomplete",
            "invalid",
            "malformed",
            "missing",
            "partial",
            "stale",
            "undefined",
            "undeclared",
        }
    )
    return has_incomplete_signal and bool(words & PLANNING_CODE_ARTIFACT_TERMS)


def _planning_window_is_conditional_execution_risk(text: str) -> bool:
    if not re.match(r"^(?:if|when)\b", text):
        return False
    if re.match(r"^(?:if|when)\s+(?:the\s+)?(?:diff|edit|patch|transform)\b", text):
        return False
    execution_risk = (
        r"(?:compile|compilation|nvcc|score|scoring|benchmark|correctness)"
    )
    failure_signal = r"(?:error|fail|fails|failure|regress|regresses|regression)"
    return bool(
        re.search(rf"\b{execution_risk}\b.{{0,80}}\b{failure_signal}\b", text)
        or re.search(rf"\b{failure_signal}\b.{{0,80}}\b{execution_risk}\b", text)
    )


def _planning_window_predicts_compile_failure(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:will|would|likely)\b.{0,80}\b"
            r"(?:(?:compile|compilation|nvcc)\s+(?:fail|failure|error)|"
            r"(?:fail|break)\s+(?:compile|compilation))\b",
            text,
        )
    )


def _planning_window_predicts_preflight_rejection(text: str) -> bool:
    if not _planning_window_describes_current_attempt_failure(text):
        return False
    return bool(
        re.search(
            r"\b(?:will|would|likely)\b.{0,100}\b"
            r"(?:reject|rejected|fail|failed)\b.{0,100}\b"
            r"(?:preflight|structural|contract|validator|validation)\b",
            text,
        )
        or re.search(
            r"\b(?:preflight|structural|contract|validator|validation)\b"
            r".{0,100}\b(?:will|would|likely)\b.{0,100}\b"
            r"(?:reject|rejected|fail|failed)\b",
            text,
        )
    )


def _planning_window_describes_current_attempt_failure(text: str) -> bool:
    current_subject = (
        r"(?:this|current|proposed|candidate)\s+"
        r"(?:candidate|change|diff|edit|patch|transform)"
    )
    edit_as_written = (
        r"(?:candidate|change|diff|edit|patch|transform)\s+"
        r"(?:as\s+(?:written|is)|as-is|in\s+this\s+decision)"
    )
    failure_signal = (
        r"(?:will|would|likely|is|are|cannot|can't|can\s+not)\b.{0,100}\b"
        r"(?:fail|failure|invalid|malformed|undefined|undeclared|ambiguous|"
        r"rejected|reject|regressed|regression|wrong|incorrect|segfault|"
        r"break\s+correctness)"
    )
    return bool(
        re.search(rf"\b{current_subject}\b.{{0,120}}\b{failure_signal}\b", text)
        or re.search(rf"\b{edit_as_written}\b.{{0,120}}\b{failure_signal}\b", text)
    )


def _planning_window_predicts_correctness_failure(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:will|would|likely)\b.{0,80}\b"
            r"(?:break correctness|fail correctness|wrong|incorrect|segfault)\b",
            text,
        )
    )


def _validate_candidate_patch_preflight(candidate_patch: str) -> None:
    validate_candidate_patch_structural_preflight(
        candidate_patch,
        allow_cuda_source_edits=False,
    )


def validate_candidate_patch_structural_preflight(
    candidate_patch: str,
    *,
    allow_cuda_source_edits: bool,
    promoted_preflight_classes: frozenset[str] = frozenset(),
) -> None:
    if not candidate_patch.strip():
        return
    inspection = _inspect_candidate_patch(candidate_patch)
    for added_line in inspection.added_lines:
        if added_line.rstrip(" \t") != added_line:
            raise ValueError("candidate_patch added lines must not contain trailing whitespace")
    for track in CUDA_STRUCTURAL_PREFLIGHT_TRACKS:
        if track.detector(inspection):
            raise ValueError(
                f"structural preflight track {track.name} classified as "
                f"{track.failure_class}: {track.message}"
            )
    for track in CUDA_PROMOTED_STRUCTURAL_PREFLIGHT_TRACKS:
        if track.failure_class not in promoted_preflight_classes:
            continue
        if track.detector(inspection):
            raise ValueError(
                f"structural preflight track {track.name} classified as "
                f"{track.failure_class}: {track.message}"
            )
    if inspection.edits_cuda_source and not allow_cuda_source_edits:
        raise ValueError(
            "candidate_patch must not edit CUDA source files directly; use "
            "candidate_transform for .cu/.cuh kernel edits"
        )


def candidate_patch_structural_advisories(candidate_patch: str) -> tuple[str, ...]:
    if not candidate_patch.strip():
        return ()
    inspection = _inspect_candidate_patch(candidate_patch)
    return tuple(
        f"structural advisory track {track.name} categorized as "
        f"{track.category}: {track.message}"
        for track in CUDA_STRUCTURAL_ADVISORY_TRACKS
        if track.detector(inspection)
    )


def _inspect_candidate_patch(candidate_patch: str) -> CandidatePatchInspection:
    added_lines = tuple(_candidate_patch_added_lines(candidate_patch))
    removed_lines = tuple(_candidate_patch_removed_lines(candidate_patch))
    return CandidatePatchInspection(
        patch_text=candidate_patch,
        added_lines=added_lines,
        removed_lines=removed_lines,
        added_text="\n".join(added_lines),
        removed_text="\n".join(removed_lines),
        changed_paths=frozenset(_candidate_patch_changed_paths(candidate_patch)),
    )


def _candidate_patch_added_lines(candidate_patch: str) -> list[str]:
    return [
        line[1:]
        for line in candidate_patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def _candidate_patch_removed_lines(candidate_patch: str) -> list[str]:
    return [
        line[1:]
        for line in candidate_patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
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


def _candidate_edit_present(
    candidate_patch: str,
    candidate_transform: dict[str, Any] | None,
) -> bool:
    return bool(candidate_patch.strip() or candidate_transform is not None)


def _candidate_edit_changed_paths(
    candidate_patch: str,
    candidate_transform: dict[str, Any] | None,
) -> set[str]:
    if candidate_transform is not None and candidate_transform.get("op") == "batch":
        return {
            str(step["path"])
            for step in candidate_transform.get("steps", [])
            if isinstance(step, dict) and isinstance(step.get("path"), str)
        }
    if candidate_transform is not None and isinstance(candidate_transform.get("path"), str):
        return {candidate_transform["path"]}
    return _candidate_patch_changed_paths(candidate_patch)


def _candidate_patch_edits_cuda_source(candidate_patch: str) -> bool:
    return any(
        path.endswith((".cu", ".cuh"))
        for path in _candidate_patch_changed_paths(candidate_patch)
    )


def _candidate_patch_adds_only_unroll_pragmas(candidate_patch: str) -> bool:
    added_lines = [line.strip() for line in _candidate_patch_added_lines(candidate_patch)]
    meaningful_added_lines = [line for line in added_lines if line]
    return bool(meaningful_added_lines) and all(
        line == "#pragma unroll" for line in meaningful_added_lines
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


def _candidate_patch_uses_unsupported_wmma_fragment_shape(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    constants = _integer_constants_from_text(added_text)
    for args in _wmma_fragment_args(compact):
        if len(args) < 5:
            continue
        use = args[0].removeprefix("nvcuda::")
        if use not in {"wmma::matrix_a", "wmma::matrix_b", "wmma::accumulator"}:
            continue
        dims = tuple(_resolve_wmma_dim(arg, constants) for arg in args[1:4])
        if any(dim is not None and dim != 16 for dim in dims):
            return True
    return False


def _candidate_patch_uses_unresolved_wmma_fragment_shape(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    constants = _integer_constants_from_text(added_text)
    for args in _wmma_fragment_args(compact):
        if len(args) < 5:
            continue
        use = args[0].removeprefix("nvcuda::")
        if use not in {"wmma::matrix_a", "wmma::matrix_b", "wmma::accumulator"}:
            continue
        dims = tuple(_resolve_wmma_dim(arg, constants) for arg in args[1:4])
        if any(dim is None for dim in dims):
            return True
    return False


def _integer_constants_from_text(text: str) -> dict[str, int]:
    constants: dict[str, int] = {}
    for match in re.finditer(
        r"\bconstexpr\s+int\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        r"(?P<value>[-+]?\d+)\s*;",
        text,
    ):
        constants[match.group("name")] = int(match.group("value"))
    return constants


def _wmma_fragment_args(compact_text: str) -> list[list[str]]:
    fragments: list[list[str]] = []
    for match in re.finditer(r"(?:nvcuda::)?wmma::fragment<(?P<body>[^<>]+)>", compact_text):
        fragments.append([part for part in match.group("body").split(",") if part])
    return fragments


def _resolve_wmma_dim(value: str, constants: dict[str, int]) -> int | None:
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    return constants.get(value)


def _candidate_patch_has_incomplete_mma_head_dim_extension(candidate_patch: str) -> bool:
    kernel_before = re.search(
        r"(?m)^-\s*constexpr\s+int\s+kHeadDim\s*=\s*(?P<value>\d+)\s*;",
        candidate_patch,
    )
    kernel_after = re.search(
        r"(?m)^\+\s*constexpr\s+int\s+kHeadDim\s*=\s*(?P<value>\d+)\s*;",
        candidate_patch,
    )
    wrapper_before = re.search(
        r"(?m)^-\s*SMOKE_HEAD_DIM\s*=\s*(?P<value>\d+)\b",
        candidate_patch,
    )
    wrapper_after = re.search(
        r"(?m)^\+\s*SMOKE_HEAD_DIM\s*=\s*(?P<value>\d+)\b",
        candidate_patch,
    )
    if (
        kernel_before is None
        or kernel_after is None
        or wrapper_before is None
        or wrapper_after is None
    ):
        return False
    extends_constant = (
        int(kernel_after.group("value")) > int(kernel_before.group("value"))
        and int(wrapper_after.group("value")) > int(wrapper_before.group("value"))
    )
    if not extends_constant:
        return False
    compact_added = re.sub(r"\s+", "", "\n".join(_candidate_patch_added_lines(candidate_patch)))
    adds_wider_loop = (
        "for(intchunk=0;chunk<" in compact_added
        or "kHeadDim/16" in compact_added
        or "kHeadChunks" in compact_added
    )
    return not adds_wider_loop


def _candidate_patch_has_inconsistent_mma_sequence_cap(candidate_patch: str) -> bool:
    kernel_before = re.search(
        r"(?m)^-\s*constexpr\s+int\s+kMaxSeqLen\s*=\s*(?P<value>\d+)\s*;",
        candidate_patch,
    )
    kernel_after = re.search(
        r"(?m)^\+\s*constexpr\s+int\s+kMaxSeqLen\s*=\s*(?P<value>\d+)\s*;",
        candidate_patch,
    )
    wrapper_before = _python_int_set_from_patch_line(candidate_patch, "-", "SMOKE_SEQUENCES")
    wrapper_after = _python_int_set_from_patch_line(candidate_patch, "+", "SMOKE_SEQUENCES")
    if kernel_after is not None:
        new_cap = int(kernel_after.group("value"))
        old_cap = int(kernel_before.group("value")) if kernel_before is not None else 0
        if new_cap > old_cap and (wrapper_after is None or new_cap not in wrapper_after):
            return True
    if wrapper_before is not None and wrapper_after is not None:
        new_values = wrapper_after - wrapper_before
        if (
            any(value not in MMA_BASE_SMOKE_SEQUENCES for value in new_values)
            and kernel_after is None
        ):
            return True
    return False


def _candidate_patch_changes_mma_shape_contract_in_one_layer(
    inspection: CandidatePatchInspection,
) -> bool:
    if not ({MMA_KERNEL_SOURCE, MMA_SEED} & inspection.changed_paths):
        return False
    changes_kernel_contract = bool(
        inspection.changed_paths & {MMA_KERNEL_SOURCE}
    ) and bool(
        re.search(
            r"(?m)^[+-]\s*constexpr\s+int\s+(?:kMaxSeqLen|kHeadDim)\s*=",
            inspection.patch_text,
        )
    )
    changes_wrapper_contract = bool(inspection.changed_paths & {MMA_SEED}) and bool(
        re.search(
            r"(?m)^[+-]\s*(?:SMOKE_SEQUENCES|SMOKE_HEAD_DIM)\s*=",
            inspection.patch_text,
        )
    )
    return changes_kernel_contract != changes_wrapper_contract


def _python_int_set_from_patch_line(
    candidate_patch: str,
    prefix: str,
    name: str,
) -> frozenset[int] | None:
    escaped_prefix = re.escape(prefix)
    match = re.search(
        rf"(?m)^{escaped_prefix}\s*{re.escape(name)}\s*=\s*\{{(?P<body>[^}}]*)\}}",
        candidate_patch,
    )
    if match is None:
        return None
    values: set[int] = set()
    for item in match.group("body").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.add(int(item))
        except ValueError:
            return None
    return frozenset(values)


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


def _candidate_patch_uses_invalid_mma_probability_ldm(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    return "load_matrix_sync(probability_frag,probabilities,kTile+1)" in compact


def _candidate_patch_uses_2d_probability_index_without_2d_declaration(
    added_text: str,
) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    return (
        "probabilities[row][key]" in compact
        and "__shared____nv_bfloat16probabilities[kTile][" not in compact
    )


def _candidate_patch_removes_declaration_but_still_uses_identifier(
    inspection: CandidatePatchInspection,
) -> bool:
    removed_identifiers = set(_cuda_declared_identifiers(inspection.removed_text))
    if not removed_identifiers:
        return False
    added_identifiers = set(_cuda_declared_identifiers(inspection.added_text))
    for identifier in sorted(removed_identifiers - added_identifiers):
        if re.search(rf"\b{re.escape(identifier)}\b", inspection.added_text):
            return True
    return False


def _candidate_patch_adds_duplicate_cuda_declarations(added_text: str) -> bool:
    identifiers = _cuda_declared_identifiers(added_text)
    return len(identifiers) != len(set(identifiers))


def _cuda_declared_identifiers(text: str) -> list[str]:
    identifiers: list[str] = []
    declaration_patterns = (
        r"\b(?:nvcuda::)?wmma::fragment<[^;]+>\s+"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:[;=])",
        r"\b(?:__shared__\s+)?(?:const\s+)?(?:unsigned\s+)?"
        r"(?:float|double|int|bool|size_t|long|short|auto|half|__half|"
        r"__nv_bfloat16|torch::Tensor)\s+[*&\s]*"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:[=;\[,])",
    )
    for pattern in declaration_patterns:
        identifiers.extend(
            match.group("name")
            for match in re.finditer(pattern, text)
        )
    return identifiers


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


def _candidate_patch_adds_unused_shared_staging_buffer(
    inspection: CandidatePatchInspection,
) -> bool:
    if not inspection.edits_cuda_source:
        return False
    added_names = _cuda_shared_declared_identifiers(inspection.added_text)
    if not added_names:
        return False
    removed_names = set(_cuda_shared_declared_identifiers(inspection.removed_text))
    new_names = set(added_names) - removed_names
    for name in new_names:
        occurrences = len(re.findall(rf"\b{re.escape(name)}\b", inspection.added_text))
        declaration_occurrences = added_names.count(name)
        if occurrences <= declaration_occurrences:
            return True
    return False


def _cuda_shared_declared_identifiers(text: str) -> list[str]:
    pattern = (
        r"\b(?:extern\s+)?__shared__\s+"
        r"(?:[A-Za-z_][A-Za-z0-9_:<>]*\s+)+[*&\s]*"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\[|;|=)"
    )
    return [match.group("name") for match in re.finditer(pattern, text)]


def _candidate_patch_inserts_async_helper_inside_mma_signature(added_text: str) -> bool:
    return (
        "__pipeline_memcpy_async" in added_text
        and "__device__ __forceinline__" in added_text
        and "__global__ void mma_attention_kernel" in added_text
    )


def _candidate_patch_has_invalid_async_pipeline_stage_lifecycle(
    inspection: CandidatePatchInspection,
) -> bool:
    added_text = inspection.added_text
    if "__pipeline_memcpy_async" not in added_text or "__pipeline_wait_prior(1)" not in added_text:
        return False
    return (
        _candidate_patch_waits_for_two_stage_pipeline_before_two_commits(added_text)
        or _candidate_patch_double_buffers_without_stage_advance(added_text)
    )


def _candidate_patch_uses_narrow_async_copy_granularity(added_text: str) -> bool:
    if "__pipeline_memcpy_async" not in added_text and "cuda::memcpy_async" not in added_text:
        return False
    narrow_size_patterns = (
        r"sizeof\s*\(\s*__nv_bfloat16\s*\)",
        r"sizeof\s*\(\s*__half\s*\)",
        r"sizeof\s*\(\s*half\s*\)",
    )
    return any(re.search(pattern, added_text) for pattern in narrow_size_patterns) or bool(
        re.search(
            r"(?:__pipeline_memcpy_async|cuda::memcpy_async)\s*\([^;]*,\s*(?:2|4)\s*\)",
            added_text,
            flags=re.DOTALL,
        )
    )


def _candidate_patch_waits_for_two_stage_pipeline_before_two_commits(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    first_wait = compact.find("__pipeline_wait_prior(1);")
    if first_wait < 0:
        return False
    commits_before_wait = compact[:first_wait].count("__pipeline_commit();")
    return commits_before_wait < 2


def _candidate_patch_double_buffers_without_stage_advance(added_text: str) -> bool:
    compact = re.sub(r"\s+", "", added_text)
    for match in re.finditer(r"1-(?P<name>[A-Za-z_][A-Za-z0-9_]*)", compact):
        name = match.group("name")
        if not re.search(rf"\[[^\]]*{re.escape(name)}[^\]]*\]", compact):
            continue
        if _compact_cuda_text_advances_binary_stage(compact, name):
            continue
        return True
    return False


def _compact_cuda_text_advances_binary_stage(compact_text: str, name: str) -> bool:
    escaped = re.escape(name)
    return bool(
        re.search(rf"{escaped}=1-{escaped};", compact_text)
        or re.search(rf"{escaped}\^=1;", compact_text)
        or re.search(rf"{escaped}=\({escaped}\+1\)%2;", compact_text)
        or re.search(rf"{escaped}=\({escaped}\+1\)&1;", compact_text)
    )


def _candidate_patch_has_obvious_cuda_delimiter_mismatch(
    inspection: CandidatePatchInspection,
) -> bool:
    if not inspection.edits_cuda_source:
        return False
    text = _cuda_text_without_comments_or_strings(inspection.added_text)
    return any(
        text.count(opening) != text.count(closing)
        for opening, closing in (("(", ")"), ("[", "]"))
    )


def _cuda_text_without_comments_or_strings(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    return re.sub(r"'(?:\\.|[^'\\])*'", "''", text)


CUDA_STRUCTURAL_PREFLIGHT_TRACKS: tuple[CandidatePatchPreflightTrack, ...] = (
    CandidatePatchPreflightTrack(
        name="no_effect_pragma_only",
        failure_class="no_effect_or_skeleton",
        message=(
            "pragma-only edits do not change dataflow; pair unroll directives with a "
            "substantive transform before compiling or scoring"
        ),
        detector=lambda inspection: _candidate_patch_adds_only_unroll_pragmas(
            inspection.patch_text
        ),
    ),
    CandidatePatchPreflightTrack(
        name="incomplete_shape_graduation",
        failure_class="incomplete_or_malformed_edit",
        message=(
            "shape-cap changes must update all loop/chunk structure that covers the new "
            "head dimension"
        ),
        detector=lambda inspection: _candidate_patch_has_incomplete_mma_head_dim_extension(
            inspection.patch_text
        ),
    ),
    CandidatePatchPreflightTrack(
        name="shape_cap_consistency",
        failure_class="incomplete_or_malformed_edit",
        message=(
            "shape-cap graduation must keep wrapper and kernel sequence caps consistent "
            "inside the same structured edit"
        ),
        detector=lambda inspection: _candidate_patch_has_inconsistent_mma_sequence_cap(
            inspection.patch_text
        ),
    ),
    CandidatePatchPreflightTrack(
        name="cuda_pipeline_api_shape",
        failure_class="cuda_syntax_error",
        message=(
            "CUDA pipeline waits use the runtime argument form, not a templated "
            "__pipeline_wait_prior<N> call"
        ),
        detector=lambda inspection: "__pipeline_wait_prior<" in inspection.added_text,
    ),
    CandidatePatchPreflightTrack(
        name="async_pipeline_stage_lifecycle",
        failure_class="correctness_nonfinite_output",
        message=(
            "two-stage async-copy pipelines must prefill enough committed stages before "
            "wait_prior(1), and double-buffer stage indices must advance after consumption"
        ),
        detector=_candidate_patch_has_invalid_async_pipeline_stage_lifecycle,
    ),
    CandidatePatchPreflightTrack(
        name="wmma_fragment_shape",
        failure_class="unsupported_wmma_shape",
        message="Ampere BF16 WMMA fragments in this kernel must use supported 16x16x16 shapes",
        detector=lambda inspection: _candidate_patch_uses_unsupported_wmma_fragment_shape(
            inspection.added_text
        ),
    ),
    CandidatePatchPreflightTrack(
        name="wmma_fragment_element_type",
        failure_class="unsupported_wmma_shape",
        message=(
            "WMMA matrix fragments need explicit CUDA element types such as "
            "__nv_bfloat16, not generic scalar_t or missing element-type parameters"
        ),
        detector=lambda inspection: _candidate_patch_uses_generic_scalar_wmma_fragments(
            inspection.added_text
        )
        or _candidate_patch_uses_missing_wmma_matrix_element_type(inspection.added_text),
    ),
    CandidatePatchPreflightTrack(
        name="shape_contract_batch",
        failure_class="incomplete_or_malformed_edit",
        message=(
            "MMA shape-contract edits must update wrapper and kernel contract layers "
            "together before compile or score"
        ),
        detector=_candidate_patch_changes_mma_shape_contract_in_one_layer,
    ),
    CandidatePatchPreflightTrack(
        name="symbol_lifecycle",
        failure_class="stale_or_undefined_symbol",
        message=(
            "fragment/probability-tile edits must remove stale declarations and declare "
            "the data structure shape used by later indexes"
        ),
        detector=lambda inspection: _candidate_patch_leaves_orphan_mma_k_fragment(
            inspection.added_text
        )
        or _candidate_patch_adds_stray_mma_probability_fragment_statement(
            inspection.added_text
        )
        or _candidate_patch_uses_2d_probability_index_without_2d_declaration(
            inspection.added_text
        ),
    ),
    CandidatePatchPreflightTrack(
        name="shared_tile_scope",
        failure_class="correctness_failed",
        message=(
            "shared-memory tiles are tile-local after staging; loads must not reapply the "
            "global key_start offset"
        ),
        detector=lambda inspection: _candidate_patch_uses_global_offset_for_shared_k_tile(
            inspection.added_text
        ),
    ),
    CandidatePatchPreflightTrack(
        name="wmma_load_alignment",
        failure_class="unsupported_wmma_shape",
        message="WMMA load_matrix_sync leading dimensions for half-type inputs must be aligned",
        detector=lambda inspection: _candidate_patch_uses_invalid_mma_probability_ldm(
            inspection.added_text
        ),
    ),
    CandidatePatchPreflightTrack(
        name="cross_thread_row_state",
        failure_class="correctness_failed",
        message=(
            "row softmax state cannot move to per-thread registers when later rows are "
            "owned by other threads"
        ),
        detector=(
            lambda inspection: (
                _candidate_patch_uses_thread_local_mma_row_state_for_cross_thread_rows(
                    inspection.added_text
                )
            )
        ),
    ),
    CandidatePatchPreflightTrack(
        name="no_effect_wmma_skeleton",
        failure_class="no_effect_or_skeleton",
        message=(
            "WMMA/preload fragments and shared buffers must feed MMA and online-softmax "
            "dataflow before a compile or score step is useful"
        ),
        detector=lambda inspection: _candidate_patch_adds_unused_mma_preload_fragment(
            inspection.added_text
        )
        or _candidate_patch_adds_unused_wmma_compile_skeleton(inspection.added_text),
    ),
    CandidatePatchPreflightTrack(
        name="no_effect_async_helpers",
        failure_class="no_effect_or_skeleton",
        message="async-copy helper wrappers must be called by real dataflow in the same edit",
        detector=lambda inspection: _candidate_patch_adds_unused_async_copy_helpers(
            inspection.added_text
        ),
    ),
    CandidatePatchPreflightTrack(
        name="no_effect_shared_staging_buffer",
        failure_class="no_effect_or_skeleton",
        message=(
            "new shared-memory staging buffers must be loaded from, stored to, or consumed "
            "by executable dataflow in the same transform"
        ),
        detector=_candidate_patch_adds_unused_shared_staging_buffer,
    ),
    CandidatePatchPreflightTrack(
        name="cuda_helper_placement",
        failure_class="cuda_syntax_error",
        message="helper definitions must not be inserted inside the CUDA kernel signature",
        detector=lambda inspection: _candidate_patch_inserts_async_helper_inside_mma_signature(
            inspection.added_text
        ),
    ),
)


CUDA_STRUCTURAL_ADVISORY_TRACKS: tuple[CandidatePatchAdvisoryTrack, ...] = (
    CandidatePatchAdvisoryTrack(
        name="async_copy_granularity_preference",
        category="async_copy_pipeline",
        message=(
            "narrow async-copy sizes are allowed for compile repair, but Ampere throughput "
            "hypotheses should prefer aligned vector groups when the dataflow supports them"
        ),
        detector=lambda inspection: _candidate_patch_uses_narrow_async_copy_granularity(
            inspection.added_text
        ),
    ),
)


CUDA_PROMOTED_STRUCTURAL_PREFLIGHT_TRACKS: tuple[CandidatePatchPreflightTrack, ...] = (
    CandidatePatchPreflightTrack(
        name="promoted_symbol_lifecycle_removed_declaration",
        failure_class="stale_or_undefined_symbol",
        message=(
            "after repeated stale-symbol failures, edits that remove a declaration must "
            "replace every remaining use or introduce the replacement declaration"
        ),
        detector=_candidate_patch_removes_declaration_but_still_uses_identifier,
    ),
    CandidatePatchPreflightTrack(
        name="promoted_symbol_lifecycle_duplicate_declaration",
        failure_class="stale_or_undefined_symbol",
        message=(
            "after repeated symbol-lifecycle failures, duplicate local declarations in "
            "one CUDA edit are rejected before compile"
        ),
        detector=lambda inspection: _candidate_patch_adds_duplicate_cuda_declarations(
            inspection.added_text
        ),
    ),
    CandidatePatchPreflightTrack(
        name="promoted_cuda_delimiter_balance",
        failure_class="cuda_syntax_error",
        message=(
            "after repeated CUDA syntax failures, added CUDA snippets must be "
            "delimiter-complete before compile"
        ),
        detector=_candidate_patch_has_obvious_cuda_delimiter_mismatch,
    ),
    CandidatePatchPreflightTrack(
        name="promoted_wmma_explicit_shape_contract",
        failure_class="unsupported_wmma_shape",
        message=(
            "after repeated WMMA shape failures, newly added fragments must use literal "
            "or same-edit constexpr 16x16x16 contracts"
        ),
        detector=lambda inspection: _candidate_patch_uses_unresolved_wmma_fragment_shape(
            inspection.added_text
        ),
    ),
)


def promoted_preflight_track_names_for_classes(
    failure_classes: frozenset[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            track.name
            for track in CUDA_PROMOTED_STRUCTURAL_PREFLIGHT_TRACKS
            if track.failure_class in failure_classes
        )
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
    if (
        _infer_integer_constant_change(candidate_edit) is not None
        or _infer_integer_constant_alias_change(candidate_edit) is not None
    ):
        return True
    words = re.findall(r"[a-z]+", normalized)
    return any(word in PATCH_REQUIRED_EDIT_VERBS for word in words)


def _validate_next_command(
    command: str,
    *,
    candidate_patch: str = "",
    candidate_transform: dict[str, Any] | None = None,
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
        candidate_transform=candidate_transform,
        planning_text=planning_text,
    )
    return command


def _validate_subcommand_arguments(
    parts: list[str],
    *,
    candidate_patch: str = "",
    candidate_transform: dict[str, Any] | None = None,
    planning_text: str = "",
) -> None:
    subcommand = parts[1]
    if subcommand == "env":
        _validate_env_command_context(planning_text)
    elif subcommand == "compile":
        _validate_compile_command_context(
            planning_text,
            candidate_patch=candidate_patch
            or ("<structured-transform>" if candidate_transform is not None else ""),
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
                candidate_transform=candidate_transform,
            )
            _validate_mma_compile_transform_cap_consistency(
                source=source,
                candidate_transform=candidate_transform,
            )
        if out_dir is not None:
            _validate_command_path(out_dir, "--out-dir", allowed_roots=("build/",))
    elif subcommand == "score":
        _validate_score_command_context(planning_text)
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
            _validate_patched_mma_score_shape_extension(
                parts,
                candidate=candidate,
                candidate_patch=candidate_patch,
                candidate_transform=candidate_transform,
            )
            _validate_known_candidate_score_shape(
                parts,
                candidate=candidate,
                candidate_patch=candidate_patch,
                candidate_transform=candidate_transform,
            )
    elif subcommand == "profile":
        backend = _single_option_value(parts, "--backend")
        if backend is None:
            raise ValueError("next_command profile requires --backend")
        if backend != "candidate":
            raise ValueError("next_command profile currently supports --backend candidate only")
        candidate = _single_option_value(parts, "--candidate")
        if candidate is None:
            raise ValueError("next_command profile requires --candidate")
        _validate_command_path(
            candidate,
            "--candidate",
            allowed_roots=("candidates/",),
            suffixes=(".py",),
        )
        _validate_patched_mma_score_shape_extension(
            parts,
            candidate=candidate,
            candidate_patch=candidate_patch,
            candidate_transform=candidate_transform,
        )
        _validate_profile_runtime_available()
        _validate_profile_command_context(planning_text)
        _validate_profile_candidate_shape(
            parts,
            candidate=candidate,
            candidate_patch=candidate_patch,
            candidate_transform=candidate_transform,
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
    candidate_transform: dict[str, Any] | None = None,
) -> None:
    if (
        _candidate_edit_present(candidate_patch, candidate_transform)
        or source not in RECORDED_NO_PATCH_COMPILE_SOURCES
    ):
        return
    raise ValueError(
        "next_command repeats a recorded no-patch compile diagnostic; include "
        "candidate_transform/candidate_patch to build-check a source change or score a "
        "pending compiled transform instead"
    )


def _validate_score_command_context(planning_text: str) -> None:
    text = " ".join(planning_text.lower().replace("-", " ").split())
    if not text:
        return
    profiler_intent = (
        "profile" in text
        or "profiling" in text
        or "nsight" in text
        or "ncu" in text
        or "classify the bottleneck" in text
    )
    unavailable_metrics = (
        "achieved bandwidth",
        "bank conflict",
        "bottleneck",
        "compute bound",
        "instruction mix",
        "memory bandwidth",
        "memory bound",
        "occupancy",
        "roofline",
        "scheduler",
        "stall",
        "tensor core utilization",
        "tensor core under utilization",
    )
    if profiler_intent and any(metric in text for metric in unavailable_metrics):
        raise ValueError(
            "next_command score cannot collect profiler metrics; avo score measures "
            "correctness, timing, and TFLOPS only. Use score for candidate validation, "
            "or add an actual supported profiler diagnostic before requesting bandwidth, "
            "occupancy, scheduler, instruction-mix, or tensor-core-utilization evidence"
        )


def _validate_profile_command_context(planning_text: str) -> None:
    text = " ".join(planning_text.lower().replace("-", " ").split())
    if not text:
        return
    if any(
        cue in text
        for cue in (
            "profile",
            "profiling",
            "nsight",
            "ncu",
            "bottleneck",
            "occupancy",
            "scheduler",
            "stall",
            "memory workload",
            "bandwidth",
            "roofline",
            "tensor core utilization",
            "instruction mix",
        )
    ):
        return
    raise ValueError(
        "next_command profile is only for profiler diagnostics; use score for correctness, "
        "timing, and TFLOPS validation"
        )


def _validate_profile_runtime_available() -> None:
    if PROFILER_UNSUPPORTED_RUNTIME_MARKER.exists():
        raise ValueError(
            "next_command profile is unavailable in this runtime because Nsight/CUPTI "
            "profiling is not supported; use score for timing/correctness or propose a "
            "candidate_transform with a compile/score validation"
        )


def _validate_profile_candidate_shape(
    parts: list[str],
    *,
    candidate: str,
    candidate_patch: str,
    candidate_transform: dict[str, Any] | None = None,
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
    if _candidate_edit_present(candidate_patch, candidate_transform):
        return
    if candidate == WARP_ROWS_SEED and (
        any(seq_len > 256 for seq_len in seq_lens)
        or head_dim > 128
        or total_tokens > 1024
        or num_heads > 4
    ):
        raise ValueError(
            "next_command profiles cuda_warp_rows_attention_seed.py outside its "
            "unpatched seq_len<=256/head_dim<=128/total_tokens<=1024/num_heads<=4 "
            "cap; include candidate_transform/candidate_patch to update the "
            "wrapper/kernel first"
        )
    if candidate == MMA_SEED and _is_outside_mma_validated_cap(
        seq_lens=seq_lens,
        head_dim=head_dim,
        total_tokens=total_tokens,
        num_heads=num_heads,
    ):
        raise ValueError(
            "next_command profiles cuda_mma_attention_seed.py outside its unpatched "
            "seq_len 16/32/64/128/256/1024/2048/4096/8192/16384/32768, "
            "head_dim 128, total_tokens<=32768, and num_heads<=16 cap; include "
            "candidate_transform/candidate_patch to update the wrapper/kernel first"
        )
    if candidate == TILED_SEED and _is_outside_tiled_validated_cap(
        seq_lens=seq_lens,
        head_dim=head_dim,
        total_tokens=total_tokens,
        num_heads=num_heads,
    ):
        raise ValueError(
            "next_command profiles cuda_tiled_attention_seed.py outside its validated "
            "shape cap; include candidate_transform/candidate_patch to update the "
            "wrapper/kernel first"
        )


def _validate_known_candidate_score_shape(
    parts: list[str],
    *,
    candidate: str,
    candidate_patch: str,
    candidate_transform: dict[str, Any] | None = None,
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
    if _candidate_edit_present(candidate_patch, candidate_transform):
        changed_paths = _candidate_edit_changed_paths(candidate_patch, candidate_transform)
        if (
            candidate == TILED_SEED
            and _is_outside_tiled_validated_cap(
                seq_lens=seq_lens,
                head_dim=head_dim,
                total_tokens=total_tokens,
                num_heads=num_heads,
            )
            and changed_paths <= {TILED_SEED}
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
        if _is_outside_mma_validated_cap(
            seq_lens=seq_lens,
            head_dim=head_dim,
            total_tokens=total_tokens,
            num_heads=num_heads,
        ):
            raise ValueError(
                "next_command scores cuda_mma_attention_seed.py outside its unpatched "
                "seq_len 16/32/64/128/256/1024/2048/4096/8192/16384/32768, "
                "head_dim 128, total_tokens<=32768, "
                "and num_heads<=16 cap; "
                "include candidate_transform/candidate_patch to update the wrapper/kernel first"
            )
        if max(seq_lens) < MMA_ACCEPTED_VALIDATION_SEQ:
            raise ValueError(
                "next_command scores cuda_mma_attention_seed.py below the current accepted "
                f"seq{MMA_ACCEPTED_VALIDATION_SEQ} validation lane; use "
                f"seq{MMA_ACCEPTED_VALIDATION_SEQ} or a larger structured "
                "shape-graduation score instead of another small smoke score"
            )
        if _is_recorded_mma_seed_score(
            seq_lens=seq_lens,
            head_dim=head_dim,
            total_tokens=total_tokens,
            num_heads=num_heads,
        ):
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


def _is_outside_mma_validated_cap(
    *,
    seq_lens: tuple[int, ...],
    head_dim: int,
    total_tokens: int,
    num_heads: int,
) -> bool:
    return (
        any(seq_len not in MMA_BASE_SMOKE_SEQUENCES for seq_len in seq_lens)
        or head_dim != 128
        or total_tokens > 32768
        or num_heads > 16
    )


def _is_recorded_mma_seed_score(
    *,
    seq_lens: tuple[int, ...],
    head_dim: int,
    total_tokens: int,
    num_heads: int,
) -> bool:
    return (
        (seq_lens, head_dim, total_tokens, num_heads)
        in {
            ((1024,), 128, 8192, 8),
            ((2048,), 128, 16384, 16),
            ((4096,), 128, 32768, 16),
            ((8192,), 128, 32768, 16),
            ((16384,), 128, 32768, 16),
            ((32768,), 128, 32768, 16),
            ((4096, 8192, 16384, 32768), 128, 32768, 16),
        }
    )


def _validate_patched_mma_score_shape_extension(
    parts: list[str],
    *,
    candidate: str,
    candidate_patch: str,
    candidate_transform: dict[str, Any] | None = None,
) -> None:
    if not _candidate_edit_present(candidate_patch, candidate_transform) or candidate != MMA_SEED:
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
    changed_paths = _candidate_edit_changed_paths(candidate_patch, candidate_transform)
    if candidate_transform is not None and {
        MMA_SEED,
        "candidates/cuda_mma_attention/attention_kernel.cu",
    } <= changed_paths:
        _validate_mma_transform_covers_score_shape(
            candidate_transform,
            seq_lens=seq_lens,
        )
        return
    if _is_mma_seed_smoke_shape(
        seq_lens=seq_lens,
        head_dim=head_dim,
        total_tokens=total_tokens,
        num_heads=num_heads,
    ):
        return
    raise ValueError(
        "next_command scores a patched MMA shape extension beyond the current smoke cap; "
        "use a structured transform batch that updates both the wrapper cap and kernel "
        "cap/dataflow together before scoring larger shapes"
    )


def _is_mma_seed_smoke_shape(
    *,
    seq_lens: tuple[int, ...],
    head_dim: int,
    total_tokens: int,
    num_heads: int,
) -> bool:
    return (
        all(seq_len in MMA_BASE_SMOKE_SEQUENCES for seq_len in seq_lens)
        and head_dim == 128
        and total_tokens <= 32768
        and num_heads <= 16
    )


def _validate_mma_transform_covers_score_shape(
    candidate_transform: dict[str, Any],
    *,
    seq_lens: tuple[int, ...],
) -> None:
    max_seq_len = _candidate_transform_int_value(
        candidate_transform,
        op="set_constexpr_int",
        name="kMaxSeqLen",
    )
    missing_base_sequences = sorted(set(seq_lens) - MMA_BASE_SMOKE_SEQUENCES)
    if max_seq_len is None and missing_base_sequences:
        missing_text = ",".join(str(seq_len) for seq_len in missing_base_sequences)
        raise ValueError(
            "next_command scores MMA seq_lens beyond the transformed cap; "
            f"kernel cap transform is missing for: {missing_text}"
        )
    if max_seq_len is not None and any(seq_len > max_seq_len for seq_len in seq_lens):
        raise ValueError(
            "next_command scores MMA seq_lens beyond the transformed cap; "
            f"max requested seq_len={max(seq_lens)} but kMaxSeqLen={max_seq_len}"
        )
    wrapper_sequences = _candidate_transform_python_set_values(
        candidate_transform,
        name="SMOKE_SEQUENCES",
    )
    if wrapper_sequences is None:
        if missing_base_sequences:
            missing_text = ",".join(str(seq_len) for seq_len in missing_base_sequences)
            raise ValueError(
                "next_command scores MMA seq_lens beyond the transformed cap; "
                f"wrapper sequence set is missing for: {missing_text}"
            )
        return
    missing = sorted(set(seq_lens) - (MMA_BASE_SMOKE_SEQUENCES | wrapper_sequences))
    if missing:
        missing_text = ",".join(str(seq_len) for seq_len in missing)
        raise ValueError(
            "next_command scores MMA seq_lens beyond the transformed cap; "
            f"wrapper sequence set does not include: {missing_text}"
        )


def _validate_mma_compile_transform_cap_consistency(
    *,
    source: str,
    candidate_transform: dict[str, Any] | None,
) -> None:
    if source != "candidates/cuda_mma_attention/attention_kernel.cu":
        return
    if candidate_transform is None:
        return
    max_seq_len = _candidate_transform_int_value(
        candidate_transform,
        op="set_constexpr_int",
        name="kMaxSeqLen",
    )
    wrapper_sequences = _candidate_transform_python_set_values(
        candidate_transform,
        name="SMOKE_SEQUENCES",
    )
    if max_seq_len is None:
        return
    if wrapper_sequences is None and max_seq_len not in MMA_BASE_SMOKE_SEQUENCES:
        raise ValueError(
            "candidate_transform has inconsistent MMA shape caps; "
            f"kMaxSeqLen={max_seq_len} but wrapper sequence set is not updated"
        )
    if wrapper_sequences is None:
        return
    if max_seq_len not in MMA_BASE_SMOKE_SEQUENCES | wrapper_sequences:
        raise ValueError(
            "candidate_transform has inconsistent MMA shape caps; "
            f"kMaxSeqLen={max_seq_len} but wrapper sequence set does not include it"
        )


def _candidate_transform_int_value(
    candidate_transform: dict[str, Any],
    *,
    op: str,
    name: str,
) -> int | None:
    for step in _candidate_transform_steps_for_validation(candidate_transform):
        if step.get("op") == op and step.get("name") == name and isinstance(step.get("value"), int):
            return int(step["value"])
    return None


def _candidate_transform_python_set_values(
    candidate_transform: dict[str, Any],
    *,
    name: str,
) -> frozenset[int] | None:
    values: set[int] = set()
    found_set_transform = False
    for step in _candidate_transform_steps_for_validation(candidate_transform):
        if (
            step.get("op") == "add_int_to_python_set"
            and step.get("name") == name
            and isinstance(step.get("value"), int)
        ):
            found_set_transform = True
            values.add(int(step["value"]))
        elif step.get("op") == "replace_once":
            replacement = str(step.get("replace") or "")
            parsed = _python_int_set_values_from_assignment(replacement, name=name)
            if parsed is not None:
                found_set_transform = True
                values.update(parsed)
    return frozenset(values) if found_set_transform else None


def _candidate_transform_steps_for_validation(
    candidate_transform: dict[str, Any],
) -> list[dict[str, Any]]:
    if candidate_transform.get("op") == "batch":
        return [
            step
            for step in candidate_transform.get("steps", [])
            if isinstance(step, dict)
        ]
    return [candidate_transform]


def _python_int_set_values_from_assignment(text: str, *, name: str) -> frozenset[int] | None:
    match = re.fullmatch(
        rf"\s*{re.escape(name)}\s*=\s*\{{(?P<body>[^}}]*)\}}\s*",
        text,
    )
    if match is None:
        return None
    values: set[int] = set()
    for raw_item in match.group("body").split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            values.add(int(item))
        except ValueError:
            return None
    return frozenset(values)


def _validate_env_command_context(planning_text: str) -> None:
    normalized = " ".join(planning_text.lower().replace("_", " ").replace("-", " ").split())
    if _env_command_is_planner_interface_recovery(normalized):
        raise ValueError(
            "next_command avo env is not useful for a planner-interface failure; return a "
            "valid candidate_transform or a kernel-search diagnostic instead"
        )
    if _env_command_repeats_recorded_stability_check(normalized):
        raise ValueError(
            "next_command repeats a recorded environment stability diagnostic; use avo env "
            "only after a concrete CUDA/build environment failure"
        )
    if _env_command_is_source_inspection(normalized):
        raise ValueError(
            "next_command avo env cannot inspect source files; repo context already includes "
            "candidate excerpts, and env is only for CUDA/build environment diagnostics"
        )
    if any(keyword in normalized for keyword in ENV_COMMAND_KEYWORDS):
        return
    raise ValueError(
        "next_command avo env is only for CUDA/build environment diagnostics, "
        "not source-file inspection"
    )


def _env_command_is_planner_interface_recovery(normalized_planning_text: str) -> bool:
    planner_failure_terms = (
        "decision payload",
        "edit payload",
        "invalid decision",
        "planner interface",
        "planner returned invalid",
        "planning validation",
        "planning validation failure",
        "planning missing edit payload",
        "schema failure",
    )
    return any(term in normalized_planning_text for term in planner_failure_terms)


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


def _env_command_is_source_inspection(normalized_planning_text: str) -> bool:
    inspection_actions = (
        "examine",
        "file inspection",
        "inspect",
        "look at",
        "read",
        "review",
        "understand",
    )
    source_targets = (
        ".cu",
        ".py",
        "candidate",
        "file",
        "files",
        "kernel",
        "seed",
        "source",
        "staging",
        "vectorization",
        "wrapper",
    )
    return any(action in normalized_planning_text for action in inspection_actions) and any(
        target in normalized_planning_text for target in source_targets
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
        "or checking candidate_transform/candidate_patch, not source-file inspection"
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
            "--seq-lens 4096,8192,16384,32768 --total-tokens 32768 --num-heads 16 "
            "--head-dim 128 "
            "--dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300"
        )
    if "candidates/cuda_warp_rows_attention_seed.py" in candidates:
        return (
            "avo score --backend candidate "
            "--candidate candidates/cuda_warp_rows_attention_seed.py "
            "--seq-lens 4096,8192,16384,32768 --total-tokens 32768 --num-heads 16 "
            "--head-dim 128 "
            "--dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300"
        )
    if "candidates/cuda_tiled_attention_seed.py" in candidates:
        return (
            "avo score --backend candidate "
            "--candidate candidates/cuda_tiled_attention_seed.py "
            "--seq-lens 4096,8192,16384,32768 --total-tokens 32768 --num-heads 16 "
            "--head-dim 128 "
            "--dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300"
        )
    if "candidates/cuda_naive_attention_seed.py" in candidates:
        return (
            "avo score --backend candidate "
            "--candidate candidates/cuda_naive_attention_seed.py "
            "--seq-lens 4096,8192,16384,32768 --total-tokens 32768 --num-heads 16 "
            "--head-dim 128 "
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
