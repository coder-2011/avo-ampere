import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from avo.agent import (
    DECISION_TOOL_NAME,
    DEFAULT_AGENT_MODEL,
    VariationDecision,
    _decision_kwargs_with_feedback,
    _request_decision_response,
    _request_valid_decision,
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
        "candidate_patch": "",
        "expected_effect": "better long-sequence throughput",
        "risk": "higher shared memory pressure",
        "next_command": "avo score --backend flash-attn",
    }


def test_parse_variation_decision() -> None:
    payload = decision_payload()
    decision = parse_decision_text(json.dumps(payload))
    assert isinstance(decision, VariationDecision)
    assert decision.files_to_inspect == ["kernel.cu"]
    assert decision.candidate_patch == ""


def test_parse_variation_decision_defaults_missing_candidate_patch() -> None:
    payload = decision_payload()
    del payload["candidate_patch"]

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_patch == ""


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


def test_parse_variation_decision_rejects_env_for_source_inspection() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Inspect the warp-row wrapper cap before patching."
    payload["next_command"] = "avo env"

    with pytest.raises(ValueError, match="only for CUDA/build environment diagnostics"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_allows_env_for_cuda_environment_check() -> None:
    payload = decision_payload()
    payload["hypothesis"] = "The CUDA build environment may be misconfigured."
    payload["candidate_edit"] = "Check CUDA and nvcc environment."
    payload["expected_effect"] = "Confirm whether torch and nvcc CUDA versions match."
    payload["risk"] = "Build diagnostics may show flash-attn is still unavailable."
    payload["next_command"] = "avo env"

    decision = parse_decision_text(json.dumps(payload))

    assert decision.next_command == "avo env"


def test_parse_variation_decision_rejects_compile_candidate_option() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Compile the CUDA source to validate the build."
    payload["next_command"] = "avo compile --candidate candidates/cuda_mma_attention_seed.py"

    with pytest.raises(ValueError, match="compile does not support --candidate"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_compile_without_required_paths() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Compile the CUDA source to validate the build."
    payload["next_command"] = "avo compile --source candidates/kernel.cu"

    with pytest.raises(ValueError, match="compile requires --out-dir"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_compile_for_source_inspection() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Inspect boundary handling and indexing logic in the kernel."
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/inspect"
    )

    with pytest.raises(ValueError, match="only for CUDA build/compilation diagnostics"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_allows_compile_source_out_dir() -> None:
    payload = decision_payload()
    payload["hypothesis"] = "The CUDA source may have a build regression."
    payload["candidate_edit"] = "Compile the CUDA source to verify nvcc accepts it."
    payload["expected_effect"] = "Confirm the translation unit still builds."
    payload["risk"] = "Compilation may expose syntax or include-path issues."
    payload["next_command"] = (
        "avo compile --source candidates/new_attention/attention_kernel.cu "
        "--out-dir build/smoke"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.next_command == payload["next_command"]


def test_parse_variation_decision_rejects_recorded_no_patch_compile_baseline() -> None:
    payload = decision_payload()
    payload["hypothesis"] = "The warp-row ptxas baseline may show a new opportunity."
    payload["candidate_edit"] = "Compile the CUDA source to inspect ptxas diagnostics."
    payload["expected_effect"] = "Review already-recorded register and shared-memory usage."
    payload["risk"] = "This repeats a known baseline without changing code."
    payload["next_command"] = (
        "avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "--out-dir build/warp_diag"
    )

    with pytest.raises(ValueError, match="recorded no-patch compile diagnostic"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_recorded_mma_no_patch_compile_baseline() -> None:
    payload = decision_payload()
    payload["hypothesis"] = "The MMA seed build path may have changed."
    payload["candidate_edit"] = "Compile the CUDA source to verify nvcc accepts it."
    payload["expected_effect"] = "Confirm already-recorded WMMA seed build diagnostics."
    payload["risk"] = "This repeats a known baseline without changing code."
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_baseline_check"
    )

    with pytest.raises(ValueError, match="recorded no-patch compile diagnostic"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_compile_python_source() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Compile the CUDA source to validate the build."
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention_seed.py --out-dir build/smoke"
    )

    with pytest.raises(ValueError, match=r"--source must reference a \.cu file"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_allows_patched_compile_build_check() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Fix kernel include usage and compile the patched source."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-#include <ATen/cuda/CUDAContext.h>\n"
        "+#include <ATen/cuda/CUDAContext.h>\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "--out-dir build/smoke"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.next_command == payload["next_command"]


def test_parse_variation_decision_rejects_pragma_only_compile_check() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Add unroll pragmas to improve V accumulation throughput."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "@@ -1 +1,2 @@\n"
        " old\n"
        "+#pragma unroll\n"
        "+#pragma unroll\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "--out-dir build/warp_unroll"
    )

    with pytest.raises(ValueError, match="pragma-only performance patch"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_pragma_only_score() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Add unroll pragmas and score the warp-row kernel."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "@@ -1 +1,2 @@\n"
        " old\n"
        "+#pragma unroll\n"
        "+#pragma unroll\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_warp_rows_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="pragma-only performance patch"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_candidate_score_without_candidate() -> None:
    payload = decision_payload()
    payload["next_command"] = "avo score --backend candidate"

    with pytest.raises(ValueError, match="candidate score requires --candidate"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_unpatched_warp_rows_score_outside_cap() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Score the existing warp-row seed at seq 512."
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_warp_rows_attention_seed.py "
        "--seq-lens 512 --total-tokens 512 --num-heads 1 --head-dim 128"
    )

    with pytest.raises(ValueError, match="outside its unpatched seq_len<=256"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_unpatched_warp_rows_workload_scaling() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Score the existing warp-row seed at more heads."
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_warp_rows_attention_seed.py "
        "--seq-lens 256 --total-tokens 2048 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="total_tokens<=1024"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_allows_unpatched_warp_rows_smoke_cap() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "No edit needed; score the existing warp-row seed."
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_warp_rows_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.next_command == payload["next_command"]


def test_parse_variation_decision_allows_patched_warp_rows_score_outside_cap() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Extend the warp-row wrapper cap for seq 512."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention_seed.py "
        "b/candidates/cuda_warp_rows_attention_seed.py\n"
        "--- a/candidates/cuda_warp_rows_attention_seed.py\n"
        "+++ b/candidates/cuda_warp_rows_attention_seed.py\n"
        "@@ -1 +1 @@\n"
        "-MAX_SMOKE_SEQUENCE = 256\n"
        "+MAX_SMOKE_SEQUENCE = 512\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_warp_rows_attention_seed.py "
        "--seq-lens 512 --total-tokens 512 --num-heads 1 --head-dim 128"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.next_command == payload["next_command"]


def test_parse_variation_decision_rejects_unpatched_mma_score_outside_cap() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Score the existing MMA seed at head_dim 128."
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 32 --total-tokens 32 --num-heads 1 --head-dim 128"
    )

    with pytest.raises(ValueError, match="outside its unpatched seq_len 16/32"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_allows_unpatched_mma_smoke_cap() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "No edit; score the existing MMA seed at its validated cap."
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 32 --total-tokens 32 --num-heads 1 --head-dim 64"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.next_command == payload["next_command"]


def test_parse_variation_decision_rejects_patched_mma_shape_score_before_compile() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Extend the MMA wrapper cap for head_dim 128."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention_seed.py "
        "b/candidates/cuda_mma_attention_seed.py\n"
        "--- a/candidates/cuda_mma_attention_seed.py\n"
        "+++ b/candidates/cuda_mma_attention_seed.py\n"
        "@@ -1 +1 @@\n"
        "-SMOKE_HEAD_DIM = 64\n"
        "+SMOKE_HEAD_DIM = 128\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 32 --total-tokens 32 --num-heads 1 --head-dim 128"
    )

    with pytest.raises(ValueError, match="first run avo compile"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_unpatched_mma_workload_scaling() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Score the existing MMA seed at more heads."
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 32 --total-tokens 64 --num-heads 2 --head-dim 64"
    )

    with pytest.raises(ValueError, match="total_tokens<=32"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_unpatched_tiled_score_outside_validated_cap() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Score the existing tiled seed at the larger smoke shape."
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_tiled_attention_seed.py "
        "--seq-lens 128 --total-tokens 512 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="outside its unpatched validated seq_len 16"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_recorded_unpatched_tiled_smoke() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "No edit needed; score the validated tiny tiled smoke."
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_tiled_attention_seed.py "
        "--seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16"
    )

    with pytest.raises(ValueError, match="recorded no-patch tiled smoke"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_allows_patched_tiled_score_outside_cap() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Fix the tiled kernel online softmax and score seq 128."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_tiled_attention/attention_kernel.cu "
        "b/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-#include <ATen/cuda/CUDAContext.h>\n"
        "+#include <ATen/cuda/CUDAContext.h>\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_tiled_attention_seed.py "
        "--seq-lens 128 --total-tokens 512 --num-heads 4 --head-dim 128"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.next_command == payload["next_command"]


def test_parse_variation_decision_rejects_tiled_wrapper_cap_only_score() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Raise tiled wrapper caps and score a larger shape."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_tiled_attention_seed.py "
        "b/candidates/cuda_tiled_attention_seed.py\n"
        "--- a/candidates/cuda_tiled_attention_seed.py\n"
        "+++ b/candidates/cuda_tiled_attention_seed.py\n"
        "@@ -1 +1 @@\n"
        "-MAX_SMOKE_SEQUENCE = 128\n"
        "+MAX_SMOKE_SEQUENCE = 256\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_tiled_attention_seed.py "
        "--seq-lens 64 --total-tokens 256 --num-heads 4 --head-dim 64"
    )

    with pytest.raises(ValueError, match="only changing wrapper caps"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_non_string_patch() -> None:
    payload = decision_payload()
    payload["candidate_patch"] = {"diff": "not-a-string"}

    with pytest.raises(ValueError, match="candidate_patch must be a string"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_recovers_non_diff_patch_text_as_empty() -> None:
    payload = decision_payload()
    payload["candidate_patch"] = "No edit needed."

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_patch == ""


def test_parse_variation_decision_allows_explicit_no_edit_with_empty_patch() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "No edit needed; score the existing MMA seed."

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_patch == ""


def test_parse_variation_decision_rejects_empty_patch_for_code_edit() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Extend the CUDA kernel and update the wrapper for head_dim 32."
    payload["candidate_patch"] = ""

    with pytest.raises(ValueError, match="candidate_edit was"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_tool_parameter_markup_in_string() -> None:
    payload = decision_payload()
    payload["expected_effect"] = (
        "score smoke shape\n<parameter name=\"risk\">risk text leaked into wrong field"
    )

    with pytest.raises(ValueError, match="expected_effect must not contain tool parameter markup"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_markdown_fenced_patch() -> None:
    payload = decision_payload()
    payload["candidate_patch"] = (
        "```diff\n"
        "diff --git a/candidates/seed.py b/candidates/seed.py\n"
        "--- a/candidates/seed.py\n"
        "+++ b/candidates/seed.py\n"
        "```\n"
    )

    with pytest.raises(ValueError, match="raw diff"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_added_trailing_whitespace() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch warp-row WMMA skeleton and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+wmma::load_matrix_sync(q_frag, src, 128); \n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "--out-dir build/warp"
    )

    with pytest.raises(ValueError, match="trailing whitespace"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_self_rejected_patch() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch tiled online softmax and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_tiled_attention/attention_kernel.cu "
        "b/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    payload["risk"] = "This patch leaves a stale reference and will cause a compile error."
    payload["next_command"] = (
        "avo compile --source candidates/cuda_tiled_attention/attention_kernel.cu "
        "--out-dir build/tiled"
    )

    with pytest.raises(ValueError, match="known invalid"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_correctness_breaking_patch_warning() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch tiled online softmax rescaling and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_tiled_attention/attention_kernel.cu "
        "b/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    payload["risk"] = (
        "The original formula is correct. This patch will break correctness. "
        "Reject this direction and diagnose the tiled kernel more carefully."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_tiled_attention/attention_kernel.cu "
        "--out-dir build/tiled"
    )

    with pytest.raises(ValueError, match="known invalid"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_would_break_correctness_warning() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch warp-row WMMA skeleton and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    payload["risk"] = "The early return would break correctness if this patch were scored."
    payload["next_command"] = (
        "avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "--out-dir build/warp"
    )

    with pytest.raises(ValueError, match="known invalid"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_unused_non_improving_structural_patch() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch dynamic shared K/V buffers and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+extern __shared__ char dyn_shared[];\n"
    )
    payload["risk"] = (
        "The patch introduces unused doubled buffers, cannot improve throughput, "
        "and the existing single-buffer indexing must be updated before scoring."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "--out-dir build/warp"
    )

    with pytest.raises(ValueError, match="known invalid"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_standalone_dynamic_kv_migration() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Move K/V tiles to dynamic shared memory and score it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "@@ -1 +1,4 @@\n"
        "-old\n"
        "+extern __shared__ char shared_buffer[];\n"
        "+scalar_t* k_tiles = reinterpret_cast<scalar_t*>(shared_buffer);\n"
        "+scalar_t* v_tiles = k_tiles + kTileKeys * (kMaxHeadDim + 1);\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_warp_rows_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="standalone dynamic shared-memory"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_direct_head_dim128_shared_threshold() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Raise the warp-row shared path threshold to head_dim 128."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-const bool can_stage_shared = head_dim <= 64 && "
        "block_query + kRowsPerBlock <= seq_len;\n"
        "+const bool can_stage_shared = head_dim <= 128 && "
        "block_query + kRowsPerBlock <= seq_len;\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_warp_rows_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="head_dim 128"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_stale_tiled_rescale_fix() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Fix tiled online softmax output rescaling."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_tiled_attention/attention_kernel.cu "
        "b/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-      output_acc = tile_acc * tile_scale;\n"
        "+      output_acc = output_acc * old_scale + tile_acc * tile_scale;\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_tiled_attention_seed.py "
        "--seq-lens 128 --total-tokens 512 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="stale tiled output-rescale"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_tiled_reduction_guard_fix() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Guard tiled reductions by tile_keys and score larger shape."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_tiled_attention/attention_kernel.cu "
        "b/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_tiled_attention/attention_kernel.cu\n"
        "@@ -1 +1,6 @@\n"
        "-old\n"
        "+reduce[tid] = score;\n"
        "+reduce[tid] = -std::numeric_limits<float>::infinity();\n"
        "+reduce[tid] = shifted;\n"
        "+reduce[tid] = 0.0f;\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_tiled_attention_seed.py "
        "--seq-lens 128 --total-tokens 512 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="tiled reduction-bound guard"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_symbolic_mma_score_k32_fragment() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA head_dim32 two-chunk score path and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,3 @@\n"
        "-constexpr int kHeadDim = 16;\n"
        "+constexpr int kHeadDim = 32;\n"
        "+wmma::fragment<wmma::accumulator, kTile, kTile, kHeadDim, float> "
        "score_frag;\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="unsupported WMMA accumulator"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_literal_mma_score_k32_fragment() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA literal head_dim32 score fragment and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,2 @@\n"
        "-old\n"
        "+nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 32, float, "
        "void> score_frag;\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="unsupported WMMA accumulator"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_scalar_t_wmma_matrix_fragment() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch warp-row WMMA skeleton and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "@@ -1 +1,3 @@\n"
        "-old\n"
        "+wmma::fragment<wmma::matrix_a, 16, 16, 16, scalar_t, "
        "wmma::row_major> q_frag;\n"
        "+wmma::fragment<wmma::matrix_b, 16, 16, 16, scalar_t, "
        "wmma::col_major> k_frag;\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "--out-dir build/warp_wmma"
    )

    with pytest.raises(ValueError, match="scalar_t as a WMMA matrix fragment"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_orphan_mma_k_fragment_block() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA two-chunk QK and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,10 @@\n"
        "-old\n"
        "+wmma::store_matrix_sync(scores, score_frag, kTile, wmma::mem_row_major);\n"
        "+}\n"
        "+__syncthreads();\n"
        "+if (threadIdx.x < warpSize) {\n"
        "+  wmma::fragment<wmma::matrix_b, kTile, kTile, 16, __nv_bfloat16, "
        "wmma::col_major> k_frag;\n"
        "+}\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="orphan post-QK WMMA k_frag"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_templated_pipeline_wait_patch() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA async copy wait and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+__pipeline_wait_prior<1>();\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="__pipeline_wait_prior"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_scalar_bf16_async_copy_patch() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA async copy staging and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+__pipeline_memcpy_async(dst, src, sizeof(__nv_bfloat16));\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="16-byte aligned"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_noop_async_copy_stub_patch() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch warp-row async copy helper and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+__device__ void async_copy_tile_kv() {}\n"
    )
    payload["risk"] = (
        "Compile-only step; the async_copy_tile_kv stub is empty and not yet called, "
        "so this cannot affect correctness or throughput."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "--out-dir build/warp"
    )

    with pytest.raises(ValueError, match="known invalid"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_bf16_score_tiles_patch() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Convert warp-row score_tiles shared memory to BF16."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "@@ -1,3 +1,3 @@\n"
        "-__shared__ float score_tiles[kRowsPerBlock][kTileKeys];\n"
        "+__shared__ __nv_bfloat16 score_tiles[kRowsPerBlock][kTileKeys];\n"
        "-tile_acc += scores[key_inner] * static_cast<float>(v_tiles[key_inner][dim]);\n"
        "+tile_acc += __bfloat162float(scores[key_inner]) * "
        "static_cast<float>(v_tiles[key_inner][dim]);\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_warp_rows_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="BF16 score_tiles conversion"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_stale_code_patch_warning() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA two-chunk QK and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    payload["risk"] = (
        "Main risk: stale QK load lines after the two-chunk loop may reference undeclared "
        "fragments. The patch must remove those lines."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="stale code"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_incomplete_removal_warning() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA two-chunk PV and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    payload["risk"] = (
        "Main risk: the patch may have incomplete removal of old single-chunk PV "
        "lines. Those old lines should be completely removed before compile."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="incomplete removal"):
        parse_decision_text(json.dumps(payload))


def test_decision_tool_uses_strict_schema() -> None:
    tool = decision_tool()
    assert tool["name"] == DECISION_TOOL_NAME
    assert tool["strict"] is True
    assert tool["input_schema"]["additionalProperties"] is False
    assert "candidate_patch" in tool["input_schema"]["required"]
    assert "must start with 'No edit;'" in tool["input_schema"]["properties"]["candidate_edit"][
        "description"
    ]
    assert "starting with 'diff --git'" in tool["input_schema"]["properties"]["candidate_patch"][
        "description"
    ]


def test_default_agent_model_supports_structured_outputs_family() -> None:
    assert DEFAULT_AGENT_MODEL == "claude-sonnet-4-5-20250929"


def test_build_repo_context_lists_local_candidates() -> None:
    context = build_repo_context(Path.cwd())

    assert "candidates/cuda_identity_seed.py" in context
    assert "candidates/cuda_mma_attention_seed.py" in context
    assert "candidates/cuda_naive_attention_seed.py" in context
    assert "candidates/cuda_tiled_attention_seed.py" in context
    assert "candidates/cuda_warp_rows_attention_seed.py" in context
    assert "candidates/torch_sdpa_seed.py" in context
    assert "candidates/cuda_mma_attention/attention_kernel.cu" in context
    assert "candidates/cuda_identity/identity_kernel.cu" in context
    assert "Unpatched seed score caps:" in context
    assert "total_tokens <= 1024" in context
    assert "cuda_tiled_attention_seed.py is only validated at seq_lens 16" in context
    assert (
        "changing only candidates/cuda_tiled_attention_seed.py wrapper caps is not a fix"
        in context
    )
    assert "current tiled kernel already uses the online-softmax output recurrence" in context
    assert "tiled reduction-bound guard patch" in context
    assert "BF16 score_tiles shared-memory conversion" in context
    assert "Use avo env only for CUDA/build environment diagnostics" in context
    assert "Use avo compile only for CUDA build/compilation diagnostics" in context
    assert "Standalone pragma-only performance patches" in context
    assert "WMMA matrix fragments in generic PyTorch kernels" in context
    assert "No-patch compile diagnostics are already recorded" in context
    assert "candidates/cuda_mma_attention/attention_kernel.cu, " in context
    assert "Patch hunks must use exact current file context" in context
    assert "Patched MMA shape extensions beyond the current head_dim64 smoke" in context
    assert "--candidate candidates/cuda_mma_attention_seed.py" in context
    assert "--seq-lens 32" in context
    assert "candidate_patch as a raw unified diff" in context
    assert "avo score --backend candidate" in context
    assert "Candidate source excerpts for exact patch context:" in context
    assert "-- candidates/cuda_mma_attention_seed.py --" in context
    assert "def attention(q, k, v, causal: bool):" in context
    assert "-- candidates/cuda_mma_attention/attention_kernel.cu --" in context
    assert "csrc/flash_attn" not in context


def test_build_repo_context_falls_back_to_tiled_candidate(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    cuda_source = candidates / "cuda_tiled_attention"
    cuda_source.mkdir(parents=True)
    (candidates / "cuda_tiled_attention_seed.py").write_text(
        "def attention(q, k, v, causal):\n    return q\n",
        encoding="utf-8",
    )
    (cuda_source / "attention_kernel.cu").write_text(
        "__global__ void tiled_attention_kernel() {}\n",
        encoding="utf-8",
    )

    context = build_repo_context(tmp_path)

    assert "--candidate candidates/cuda_tiled_attention_seed.py" in context
    assert "--candidate candidates/cuda_warp_rows_attention_seed.py" not in context
    assert "-- candidates/cuda_tiled_attention_seed.py --" in context
    assert "def attention(q, k, v, causal):" in context
    assert "__global__ void tiled_attention_kernel() {}" in context


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
    assert "candidate_patch" in prompt
    assert "No-edit mode" in prompt
    assert "The diff must apply cleanly with git apply" in prompt
    assert "avoid trailing whitespace-only added lines" in prompt
    assert 'candidate_edit starts with "No edit; "' in prompt
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


def test_decision_feedback_explains_empty_patch_validation_error() -> None:
    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": "Base prompt.",
            }
        ]
    }

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError("candidate_patch must be non-empty when candidate_edit describes a code change"),
    )

    content = updated["messages"][0]["content"]
    assert "candidate_patch must be a raw git-style unified diff" in content
    assert "Choose exactly one valid mode" in content
    assert "'No edit;'" in content
    assert "diff --git" in content
    assert "Do not mention fixing" in content


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


def test_valid_decision_request_retries_invalid_decision_with_feedback() -> None:
    class FakeMessages:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            payload = decision_payload()
            if len(self.calls) == 1:
                payload["candidate_patch"] = (
                    "```diff\n"
                    "diff --git a/candidates/seed.py b/candidates/seed.py\n"
                    "--- a/candidates/seed.py\n"
                    "+++ b/candidates/seed.py\n"
                    "```\n"
                )
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload))])

    messages = FakeMessages()
    decision = _request_valid_decision(
        SimpleNamespace(messages=messages),
        {"model": "claude", "messages": [{"role": "user", "content": "plan"}]},
        retry_delay_s=0.0,
    )

    assert decision.candidate_patch == ""
    assert len(messages.calls) == 2
    assert "Validation error:" in messages.calls[1]["messages"][0]["content"]
