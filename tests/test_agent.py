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
    validate_candidate_patch_structural_preflight,
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


def test_parse_variation_decision_accepts_structured_transform() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Set the MMA tile constant through a structured transform."
    payload["candidate_transform"] = {
        "op": "set_constexpr_int",
        "path": "candidates/cuda_mma_attention/attention_kernel.cu",
        "name": "kTile",
        "value": 16,
    }
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_transform"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_patch == ""
    assert decision.candidate_transform == payload["candidate_transform"]


def test_parse_variation_decision_accepts_add_include_transform() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Add the CUDA pipeline header through a structured transform."
    payload["candidate_transform"] = {
        "op": "add_include",
        "path": "candidates/cuda_mma_attention/attention_kernel.cu",
        "header": "cuda_pipeline_primitives.h",
    }
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_include"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_transform == payload["candidate_transform"]


def test_parse_variation_decision_infers_add_include_transform_from_edit() -> None:
    payload = decision_payload()
    payload["files_to_inspect"] = []
    payload["candidate_edit"] = (
        "Add cuda_pipeline_primitives.h include to MMA kernel to enable future cp.async work."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_include"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_transform == {
        "op": "add_include",
        "path": "candidates/cuda_mma_attention/attention_kernel.cu",
        "header": "cuda_pipeline_primitives.h",
    }


def test_parse_variation_decision_rejects_add_include_on_python_file() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Add a C++ include to the Python wrapper."
    payload["candidate_transform"] = {
        "op": "add_include",
        "path": "candidates/cuda_mma_attention_seed.py",
        "header": "cuda_pipeline_primitives.h",
    }

    with pytest.raises(ValueError, match="add_include path"):
        parse_decision_text(json.dumps(payload))


def test_structured_transform_allows_conditional_compile_risk_language() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Set the MMA tile constant through a structured transform."
    payload["candidate_transform"] = {
        "op": "set_constexpr_int",
        "path": "candidates/cuda_mma_attention/attention_kernel.cu",
        "name": "kTile",
        "value": 16,
    }
    payload["risk"] = (
        "If compile fails on shared memory limits or undefined async copy symbols, "
        "choose a different substrate."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_transform"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_transform == payload["candidate_transform"]


def test_structured_transform_rejects_declared_incomplete_edit() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Set the MMA tile constant through a structured transform."
    payload["candidate_transform"] = {
        "op": "set_constexpr_int",
        "path": "candidates/cuda_mma_attention/attention_kernel.cu",
        "name": "kTile",
        "value": 16,
    }
    payload["risk"] = "The patch is incomplete and leaves undefined symbols."
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_transform"
    )

    with pytest.raises(ValueError, match="known invalid"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_accepts_transform_batch() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Batch wrapper and kernel cap changes for seq_len 512."
    steps = [
        {
            "op": "set_constexpr_int",
            "path": "candidates/cuda_mma_attention/attention_kernel.cu",
            "name": "kMaxSeqLen",
            "value": 512,
        },
        {
            "op": "add_int_to_python_set",
            "path": "candidates/cuda_mma_attention_seed.py",
            "name": "SMOKE_SEQUENCES",
            "value": 512,
        },
    ]
    payload["candidate_transform"] = {
        "op": "batch",
        "steps_json": json.dumps(steps, separators=(",", ":")),
    }
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_batch"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_patch == ""
    assert decision.candidate_transform == {"op": "batch", "steps": steps}


def test_parse_variation_decision_infers_set_constexpr_transform_from_edit() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = (
        "Use candidate_transform set_constexpr_int to change kRowsPerBlock from 4 to 8 "
        "in candidates/cuda_warp_rows_attention/attention_kernel.cu, compile-check the change."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "--out-dir build/warp_rows"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_patch == ""
    assert decision.candidate_transform == {
        "op": "set_constexpr_int",
        "path": "candidates/cuda_warp_rows_attention/attention_kernel.cu",
        "name": "kRowsPerBlock",
        "value": 8,
    }


def test_parse_variation_decision_infers_constant_transform_without_channel_word() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = (
        "Change kRowsPerBlock from 4 to 8 in "
        "candidates/cuda_warp_rows_attention/attention_kernel.cu, then compile-check the source."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "--out-dir build/warp_rows"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_transform == {
        "op": "set_constexpr_int",
        "path": "candidates/cuda_warp_rows_attention/attention_kernel.cu",
        "name": "kRowsPerBlock",
        "value": 8,
    }


def test_parse_variation_decision_infers_batch_transform_from_edit_text() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = (
        "Batch transform: set kMaxSeqLen=512 in "
        "candidates/cuda_mma_attention/attention_kernel.cu and replace "
        "SMOKE_SEQUENCES = {16, 32, 64, 128, 256} with "
        "SMOKE_SEQUENCES = {16, 32, 64, 128, 256, 512} in "
        "candidates/cuda_mma_attention_seed.py."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_seq512"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_transform == {
        "op": "batch",
        "steps": [
            {
                "op": "set_constexpr_int",
                "path": "candidates/cuda_mma_attention/attention_kernel.cu",
                "name": "kMaxSeqLen",
                "value": 512,
            },
            {
                "op": "replace_once",
                "path": "candidates/cuda_mma_attention_seed.py",
                "find": "SMOKE_SEQUENCES = {16, 32, 64, 128, 256}",
                "replace": "SMOKE_SEQUENCES = {16, 32, 64, 128, 256, 512}",
            },
        ],
    }


def test_parse_variation_decision_infers_batch_from_files_to_inspect() -> None:
    payload = decision_payload()
    payload["files_to_inspect"] = [
        "candidates/cuda_mma_attention_seed.py",
        "candidates/cuda_mma_attention/attention_kernel.cu",
    ]
    payload["candidate_edit"] = (
        "Extend MMA seq support to 512 by updating SMOKE_SEQUENCES in the wrapper to "
        "add 512 and setting kMaxSeqLen to 512 in the kernel. Use op=batch with two "
        "tiny steps."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_seq512"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_transform == {
        "op": "batch",
        "steps": [
            {
                "op": "set_constexpr_int",
                "path": "candidates/cuda_mma_attention/attention_kernel.cu",
                "name": "kMaxSeqLen",
                "value": 512,
            },
            {
                "op": "add_int_to_python_set",
                "path": "candidates/cuda_mma_attention_seed.py",
                "name": "SMOKE_SEQUENCES",
                "value": 512,
            },
        ],
    }


def test_parse_variation_decision_infers_batch_from_extend_constant_text() -> None:
    payload = decision_payload()
    payload["files_to_inspect"] = [
        "candidates/cuda_mma_attention_seed.py",
        "candidates/cuda_mma_attention/attention_kernel.cu",
    ]
    payload["candidate_edit"] = (
        "Extend the MMA seed kMaxSeqLen constant from 256 to 1024 and add 1024 "
        "to the SMOKE_SEQUENCES wrapper set. This is a two-step batch: "
        "set_constexpr_int for kMaxSeqLen and add_int_to_python_set for "
        "SMOKE_SEQUENCES."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_seq1024"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_transform == {
        "op": "batch",
        "steps": [
            {
                "op": "set_constexpr_int",
                "path": "candidates/cuda_mma_attention/attention_kernel.cu",
                "name": "kMaxSeqLen",
                "value": 1024,
            },
            {
                "op": "add_int_to_python_set",
                "path": "candidates/cuda_mma_attention_seed.py",
                "name": "SMOKE_SEQUENCES",
                "value": 1024,
            },
        ],
    }


def test_parse_variation_decision_rejects_inconsistent_mma_compile_caps() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = (
        "Set kMaxSeqLen to 65536 in the kernel and add 32768 to SMOKE_SEQUENCES."
    )
    payload["candidate_transform"] = {
        "op": "batch",
        "steps": [
            {
                "op": "set_constexpr_int",
                "path": "candidates/cuda_mma_attention/attention_kernel.cu",
                "name": "kMaxSeqLen",
                "value": 65536,
            },
            {
                "op": "add_int_to_python_set",
                "path": "candidates/cuda_mma_attention_seed.py",
                "name": "SMOKE_SEQUENCES",
                "value": 32768,
            },
        ],
    }
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_65536"
    )

    with pytest.raises(ValueError, match="inconsistent MMA shape caps"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_kernel_only_mma_sequence_cap() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Set kMaxSeqLen to 65536 in the kernel."
    payload["candidate_transform"] = {
        "op": "set_constexpr_int",
        "path": "candidates/cuda_mma_attention/attention_kernel.cu",
        "name": "kMaxSeqLen",
        "value": 65536,
    }
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_65536"
    )

    with pytest.raises(ValueError, match="wrapper sequence set is not updated"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_infers_generic_python_set_add() -> None:
    payload = decision_payload()
    payload["files_to_inspect"] = [
        "candidates/example_seed.py",
        "candidates/example_kernel.cu",
    ]
    payload["candidate_edit"] = (
        "Extend the candidate by setting kMaxSeqLen to 1024 in the kernel and adding "
        "1024 to ALLOWED_SEQUENCES in the wrapper."
    )
    payload["next_command"] = (
        "avo compile --source candidates/example_kernel.cu --out-dir build/example"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_transform == {
        "op": "batch",
        "steps": [
            {
                "op": "set_constexpr_int",
                "path": "candidates/example_kernel.cu",
                "name": "kMaxSeqLen",
                "value": 1024,
            },
            {
                "op": "add_int_to_python_set",
                "path": "candidates/example_seed.py",
                "name": "ALLOWED_SEQUENCES",
                "value": 1024,
            },
        ],
    }


def test_parse_variation_decision_rejects_patch_and_transform_together() -> None:
    payload = decision_payload()
    payload["candidate_patch"] = (
        "diff --git a/candidates/seed.py b/candidates/seed.py\n"
        "--- a/candidates/seed.py\n"
        "+++ b/candidates/seed.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    payload["candidate_transform"] = {
        "op": "replace_once",
        "path": "candidates/seed.py",
        "find": "old",
        "replace": "new",
    }

    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_invalid_transform() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch the candidate through a structured transform."
    payload["candidate_transform"] = {
        "op": "replace_once",
        "path": "candidates/seed.py",
        "find": "old",
    }

    with pytest.raises(ValueError, match="candidate_transform replace"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_malformed_transform_path() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Set the MMA tile constant through a structured transform."
    payload["candidate_transform"] = {
        "op": "set_constexpr_int",
        "path": "candidates / cud a_w arp _ rows _ attention _kernel . cu",
        "name": "INVALID",
        "value": 0,
    }

    with pytest.raises(ValueError, match="candidate_transform path must not contain whitespace"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_no_edit_with_transform() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "No edit; score the existing candidate."
    payload["candidate_transform"] = {
        "op": "set_constexpr_int",
        "path": "candidates/cuda_mma_attention/attention_kernel.cu",
        "name": "kTile",
        "value": 16,
    }

    with pytest.raises(ValueError, match="no-edit mode but includes an edit payload"):
        parse_decision_text(json.dumps(payload))


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


def test_parse_variation_decision_rejects_recorded_env_stability_check() -> None:
    payload = decision_payload()
    payload["hypothesis"] = "Confirm CUDA build toolchain stability after recent attempts."
    payload["candidate_edit"] = "No edit; run environment diagnostic."
    payload["expected_effect"] = "Confirm CUDA build setup remains valid."
    payload["risk"] = "No source risk."
    payload["next_command"] = "avo env"

    with pytest.raises(ValueError, match="recorded environment stability diagnostic"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_env_for_planner_interface_recovery() -> None:
    payload = decision_payload()
    payload["hypothesis"] = "The recent attempts failed planning validation."
    payload["candidate_edit"] = "No edit; run environment diagnostic."
    payload["expected_effect"] = "Confirm the CUDA environment is still stable."
    payload["risk"] = "This only recovers from a missing edit payload."
    payload["next_command"] = "avo env"

    with pytest.raises(ValueError, match="planner-interface failure"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_allows_env_for_concrete_build_failure() -> None:
    payload = decision_payload()
    payload["hypothesis"] = "The last extension build failed with a missing compiler error."
    payload["candidate_edit"] = "No edit; run environment diagnostic."
    payload["expected_effect"] = "Confirm whether nvcc is missing or CUDA paths are misconfigured."
    payload["risk"] = "No source risk."
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


def test_parse_variation_decision_rejects_compile_out_dir_under_candidates() -> None:
    payload = decision_payload()
    payload["hypothesis"] = "Compile the patched CUDA source."
    payload["candidate_edit"] = "Compile the CUDA source to verify nvcc accepts it."
    payload["candidate_transform"] = {
        "op": "replace_once",
        "path": "candidates/cuda_mma_attention/attention_kernel.cu",
        "find": "#include <ATen/cuda/CUDAContext.h>",
        "replace": "#include <ATen/cuda/CUDAContext.h>",
    }
    payload["expected_effect"] = "Confirm the translation unit still builds."
    payload["risk"] = "Compilation may expose syntax or include-path issues."
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir candidates/cuda_mma_attention"
    )

    with pytest.raises(ValueError, match="--out-dir must be under: build"):
        parse_decision_text(json.dumps(payload))


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
    payload["candidate_transform"] = {
        "op": "replace_once",
        "path": "candidates/cuda_warp_rows_attention/attention_kernel.cu",
        "find": "#include <ATen/cuda/CUDAContext.h>",
        "replace": "#include <ATen/cuda/CUDAContext.h>",
    }
    payload["next_command"] = (
        "avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "--out-dir build/smoke"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.next_command == payload["next_command"]


def test_parse_variation_decision_rejects_raw_cuda_patch() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch CUDA source with a raw diff."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "--out-dir build/smoke"
    )

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
        parse_decision_text(json.dumps(payload))


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

    with pytest.raises(ValueError, match="no_effect_pragma_only"):
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

    with pytest.raises(ValueError, match="no_effect_pragma_only"):
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


def test_parse_variation_decision_rejects_unpatched_warp_rows_recorded_score() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "No edit needed; score the existing warp-row seed."
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_warp_rows_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="recorded no-patch warp-row seed score"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_allows_unpatched_warp_rows_different_smoke() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "No edit; score the existing warp-row seed at seq128."
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_warp_rows_attention_seed.py "
        "--seq-lens 128 --total-tokens 1024 --num-heads 4 --head-dim 128"
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
    payload["candidate_edit"] = "Score the existing MMA seed at head_dim 256."
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 32 --total-tokens 32 --num-heads 1 --head-dim 256"
    )

    with pytest.raises(ValueError, match="outside its unpatched seq_len 16/32/64/128/256"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_unpatched_mma_smoke_cap() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "No edit; score the existing MMA seed at its validated cap."
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 32768 --total-tokens 32768 --num-heads 16 --head-dim 128"
    )

    with pytest.raises(ValueError, match="recorded unpatched MMA seed score"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_wrapper_only_mma_shape_graduation() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Extend the MMA wrapper cap for head_dim 256."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention_seed.py "
        "b/candidates/cuda_mma_attention_seed.py\n"
        "--- a/candidates/cuda_mma_attention_seed.py\n"
        "+++ b/candidates/cuda_mma_attention_seed.py\n"
        "@@ -1 +1 @@\n"
        "-SMOKE_HEAD_DIM = 128\n"
        "+SMOKE_HEAD_DIM = 256\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 32 --total-tokens 32 --num-heads 1 --head-dim 256"
    )

    with pytest.raises(ValueError, match="shape_contract_batch"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_allows_batched_mma_shape_score() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = (
        "Change kMaxSeqLen from 256 to 512 in "
        "candidates/cuda_mma_attention/attention_kernel.cu and add 512 to "
        "SMOKE_SEQUENCES in candidates/cuda_mma_attention_seed.py."
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 512 --total-tokens 2048 --num-heads 4 --head-dim 128"
    )

    decision = parse_decision_text(json.dumps(payload))

    assert decision.candidate_transform == {
        "op": "batch",
        "steps": [
            {
                "op": "set_constexpr_int",
                "path": "candidates/cuda_mma_attention/attention_kernel.cu",
                "name": "kMaxSeqLen",
                "value": 512,
            },
            {
                "op": "add_int_to_python_set",
                "path": "candidates/cuda_mma_attention_seed.py",
                "name": "SMOKE_SEQUENCES",
                "value": 512,
            },
        ],
    }


def test_parse_variation_decision_rejects_mma_score_beyond_transform_cap() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = (
        "Change kMaxSeqLen from 256 to 1024 in "
        "candidates/cuda_mma_attention/attention_kernel.cu and add 1024 to "
        "SMOKE_SEQUENCES in candidates/cuda_mma_attention_seed.py."
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 1024,2048 --total-tokens 32768 --num-heads 16 --head-dim 128"
    )

    with pytest.raises(ValueError, match="beyond the transformed cap"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_unpatched_mma_below_accepted_lane() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Score the existing MMA seed at a small smoke workload."
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 256 --total-tokens 2048 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="below the current accepted seq32768"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_unpatched_mma_beyond_current_seq_cap() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Score the existing MMA seed at seq65536."
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 65536 --total-tokens 65536 --num-heads 16 --head-dim 128"
    )

    expected = "seq_len 16/32/64/128/256/1024/2048/4096/8192/16384/32768"
    with pytest.raises(ValueError, match=expected):
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
    payload["candidate_transform"] = {
        "op": "replace_once",
        "path": "candidates/cuda_tiled_attention/attention_kernel.cu",
        "find": "#include <ATen/cuda/CUDAContext.h>",
        "replace": "#include <ATen/cuda/CUDAContext.h>",
    }
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


def test_parse_variation_decision_rejects_do_not_use_this_diff_warning() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA Q preload and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    payload["risk"] = "Do not use this diff; a correct Q-preload patch must be simpler."
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_q_preload"
    )

    with pytest.raises(ValueError, match="known invalid"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_do_not_score_self_invalid_patch() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA PV preload and compile it."
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
        "The patch references chunk outside the loop and will fail compilation. "
        "Do not score this patch as-is."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_pv_preload"
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

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
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

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
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

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
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

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
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

    with pytest.raises(ValueError, match="wmma_fragment_shape"):
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

    with pytest.raises(ValueError, match="wmma_fragment_shape"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_literal_mma_m32_fragment() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA 32-row fragment shape and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,3 @@\n"
        "-old\n"
        "+wmma::fragment<wmma::accumulator, 32, 16, 16, float> score_frag;\n"
        "+wmma::fragment<wmma::matrix_a, 32, 16, 16, __nv_bfloat16, "
        "wmma::row_major> q_frag;\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="wmma_fragment_shape"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_symbolic_mma_m32_fragment() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA tile size to 32 rows and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,4 @@\n"
        "-constexpr int kTile = 16;\n"
        "+constexpr int kTile = 32;\n"
        "+wmma::fragment<wmma::matrix_a, kTile, 16, 16, __nv_bfloat16, "
        "wmma::row_major> q_frag;\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="wmma_fragment_shape"):
        parse_decision_text(json.dumps(payload))


def test_structural_preflight_rejects_inconsistent_sequence_cap_patch() -> None:
    patch = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-constexpr int kMaxSeqLen = 1024;\n"
        "+constexpr int kMaxSeqLen = 4096;\n"
        "diff --git a/candidates/cuda_mma_attention_seed.py "
        "b/candidates/cuda_mma_attention_seed.py\n"
        "--- a/candidates/cuda_mma_attention_seed.py\n"
        "+++ b/candidates/cuda_mma_attention_seed.py\n"
        "@@ -1 +1 @@\n"
        "-SMOKE_SEQUENCES = {16, 32, 64, 128, 256, 1024}\n"
        "+SMOKE_SEQUENCES = {16, 32, 64, 128, 256, 1024, 2048}\n"
    )

    with pytest.raises(ValueError, match="shape_cap_consistency"):
        validate_candidate_patch_structural_preflight(
            patch,
            allow_cuda_source_edits=True,
        )


def test_promoted_preflight_rejects_removed_declaration_still_used() -> None:
    patch = (
        "diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n"
        "--- a/candidates/kernel.cu\n"
        "+++ b/candidates/kernel.cu\n"
        "@@ -1,2 +1,2 @@\n"
        "-float old_scale = 1.0f;\n"
        "+float new_scale = 1.0f;\n"
        " acc *= old_scale;\n"
        "+acc += old_scale;\n"
    )

    validate_candidate_patch_structural_preflight(
        patch,
        allow_cuda_source_edits=True,
    )
    with pytest.raises(ValueError, match="promoted_symbol_lifecycle_removed_declaration"):
        validate_candidate_patch_structural_preflight(
            patch,
            allow_cuda_source_edits=True,
            promoted_preflight_classes=frozenset({"stale_or_undefined_symbol"}),
        )


def test_structural_preflight_rejects_single_layer_mma_shape_contract() -> None:
    patch = (
        "diff --git a/candidates/cuda_mma_attention_seed.py "
        "b/candidates/cuda_mma_attention_seed.py\n"
        "--- a/candidates/cuda_mma_attention_seed.py\n"
        "+++ b/candidates/cuda_mma_attention_seed.py\n"
        "@@ -1 +1 @@\n"
        "-SMOKE_HEAD_DIM = 128\n"
        "+SMOKE_HEAD_DIM = 256\n"
    )

    with pytest.raises(ValueError, match="shape_contract_batch"):
        validate_candidate_patch_structural_preflight(
            patch,
            allow_cuda_source_edits=False,
        )


def test_promoted_preflight_rejects_delimiter_incomplete_cuda_snippet() -> None:
    patch = (
        "diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n"
        "--- a/candidates/kernel.cu\n"
        "+++ b/candidates/kernel.cu\n"
        "@@ -1 +1,2 @@\n"
        "-old\n"
        "+float score = max(row_max, tile_max;\n"
    )

    validate_candidate_patch_structural_preflight(
        patch,
        allow_cuda_source_edits=True,
    )
    with pytest.raises(ValueError, match="promoted_cuda_delimiter_balance"):
        validate_candidate_patch_structural_preflight(
            patch,
            allow_cuda_source_edits=True,
            promoted_preflight_classes=frozenset({"cuda_syntax_error"}),
        )


def test_promoted_preflight_requires_explicit_wmma_shape_contract() -> None:
    patch = (
        "diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n"
        "--- a/candidates/kernel.cu\n"
        "+++ b/candidates/kernel.cu\n"
        "@@ -1 +1,2 @@\n"
        "-old\n"
        "+wmma::fragment<wmma::accumulator, kTile, kTile, 16, float> score_frag;\n"
    )

    validate_candidate_patch_structural_preflight(
        patch,
        allow_cuda_source_edits=True,
    )
    with pytest.raises(ValueError, match="promoted_wmma_explicit_shape_contract"):
        validate_candidate_patch_structural_preflight(
            patch,
            allow_cuda_source_edits=True,
            promoted_preflight_classes=frozenset({"unsupported_wmma_shape"}),
        )


def test_parse_variation_decision_rejects_partial_mma_head_dim128_patch() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Extend the MMA seed to head_dim128 and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,3 @@\n"
        "-constexpr int kHeadDim = 64;\n"
        "+constexpr int kHeadDim = 128;\n"
        "+#pragma unroll\n"
        "diff --git a/candidates/cuda_mma_attention_seed.py "
        "b/candidates/cuda_mma_attention_seed.py\n"
        "--- a/candidates/cuda_mma_attention_seed.py\n"
        "+++ b/candidates/cuda_mma_attention_seed.py\n"
        "@@ -1 +1 @@\n"
        "-SMOKE_HEAD_DIM = 64\n"
        "+SMOKE_HEAD_DIM = 128\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="incomplete_shape_graduation"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_self_reported_correctness_failure() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA shape and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    payload["risk"] = "This will fail correctness if scored before adding the second pass."
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="will fail correctness"):
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

    with pytest.raises(ValueError, match="wmma_fragment_element_type"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_missing_wmma_matrix_element_type() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA Q preload and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,3 @@\n"
        "-old\n"
        "+wmma::fragment<wmma::matrix_a, kTile, kTile, 16, "
        "wmma::row_major> q_frag_chunk;\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_q_preload"
    )

    with pytest.raises(ValueError, match="wmma_fragment_element_type"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_regressed_mma_qk_preload_chain() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA QK fragment preload chain and score it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,8 @@\n"
        "-old\n"
        "+wmma::fragment<wmma::matrix_b, kTile, kTile, 16, "
        "__nv_bfloat16, wmma::col_major> k_frag_next;\n"
        "+wmma::mma_sync(score_frag, q_frag, k_frag_next, score_frag);\n"
        "+const int next_chunk = chunk + 1;\n"
        "+if (next_chunk < 8) {\n"
        "+  wmma::load_matrix_sync(k_frag_next, k + next_chunk * 16, kHeadDim);\n"
        "+}\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_regressed_mma_q_preload_chain() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA Q fragment preload chain and score it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,9 @@\n"
        "-old\n"
        "+wmma::fragment<wmma::matrix_a, kTile, kTile, 16, "
        "__nv_bfloat16, wmma::row_major> q_frag_next;\n"
        "+q_frag = q_frag_next;\n"
        "+const int next_chunk_offset = (chunk + 1) * 16;\n"
        "+wmma::load_matrix_sync(q_frag_next, q + next_chunk_offset, kHeadDim);\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="no_effect_wmma_skeleton"):
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

    with pytest.raises(ValueError, match="symbol_lifecycle"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_global_offset_for_shared_mma_k_tile() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Stage the MMA K tile in shared memory and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,3 @@\n"
        "-old\n"
        "+__shared__ __nv_bfloat16 k_shared[kTile * kHeadDim];\n"
        "+wmma::load_matrix_sync(k_frag, k_shared + key_start * kHeadDim + chunk_offset, "
        "kHeadDim);\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="shared_tile_scope"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_thread_local_mma_row_state() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Move MMA row softmax state into registers and score it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,9 @@\n"
        "-old\n"
        "+float row_max_reg[1];\n"
        "+float row_sum_reg[1];\n"
        "+float old_scale_reg[1];\n"
        "+const int tid = threadIdx.x;\n"
        "+const int row = linear / kHeadDim;\n"
        "+const int reg_idx = row / blockDim.x;\n"
        "+output_acc[linear] *= old_scale_reg[reg_idx];\n"
        "+output[linear] = __float2bfloat16(output_acc[linear] / row_sum_reg[reg_idx]);\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="cross_thread_row_state"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_thread0_local_row_state_init() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Move MMA row state into local arrays and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,12 @@\n"
        "-old\n"
        "+float row_max[kTile];\n"
        "+float row_sum[kTile];\n"
        "+float old_scale[kTile];\n"
        "+if (key_start == 0 && threadIdx.x == 0) {\n"
        "+  for (int row = 0; row < kTile; ++row) {\n"
        "+    row_max[row] = -std::numeric_limits<float>::infinity();\n"
        "+    row_sum[row] = 0.0f;\n"
        "+  }\n"
        "+}\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="cross_thread_row_state"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_unused_mma_preload_fragment() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Preload the next MMA K fragment and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,6 @@\n"
        "-old\n"
        "+wmma::fragment<wmma::matrix_b, kTile, kTile, 16, __nv_bfloat16, "
        "wmma::col_major> k_frag_next;\n"
        "+if (key_start + kTile < seq_len) {\n"
        "+  wmma::load_matrix_sync(k_frag_next, k + base + next_key_start * kHeadDim, "
        "kHeadDim);\n"
        "+}\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_preload"
    )

    with pytest.raises(ValueError, match="no_effect_wmma_skeleton"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_sync_mma_k_staging_repeat() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Stage the full MMA K tile in shared memory and score it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,4 @@\n"
        "-old\n"
        "+__shared__ __nv_bfloat16 k_shared[kTile * kHeadDim];\n"
        "+wmma::load_matrix_sync(k_frag, k_shared + chunk_offset, kHeadDim);\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_sync_mma_q_staging_repeat() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Stage the full MMA Q tile in shared memory and score it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,4 @@\n"
        "-old\n"
        "+__shared__ __nv_bfloat16 q_shared[kTile * kHeadDim];\n"
        "+wmma::load_matrix_sync(q_frag, q_shared + chunk_offset, kHeadDim);\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_sync_mma_v_staging_repeat() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Stage the MMA V tile in double-buffered shared memory."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,4 @@\n"
        "-old\n"
        "+__shared__ __nv_bfloat16 v_shared[2][kTile * kHeadDim];\n"
        "+wmma::load_matrix_sync(v_frag, &v_shared[current_buffer][chunk_offset], kHeadDim);\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_single_buffer_sync_mma_v_staging_repeat() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Stage the MMA V tile in shared memory."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,4 @@\n"
        "-old\n"
        "+__shared__ __nv_bfloat16 v_shared[kTile * kHeadDim];\n"
        "+wmma::load_matrix_sync(v_frag, v_shared + chunk_offset, kHeadDim);\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_v_staging"
    )

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_invalid_probability_wmma_ldm() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Skew the MMA probability tile by one column."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,3 @@\n"
        "-old\n"
        "+wmma::load_matrix_sync(probability_frag, probabilities, kTile + 1);\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_prob_skew"
    )

    with pytest.raises(ValueError, match="wmma_load_alignment"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_probability_stride_skew_repeat() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Skew the MMA probability tile with an aligned stride."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,5 @@\n"
        "-old\n"
        "+constexpr int kProbabilityStride = kTile + 8;\n"
        "+wmma::load_matrix_sync(probability_frag, probabilities, kProbabilityStride);\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_probability_stride_skew_2d_repeat() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Skew the MMA probability tile with a 2D aligned stride."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,7 @@\n"
        "-old\n"
        "+constexpr int kProbabilityStride = kTile + 8;\n"
        "+__shared__ __nv_bfloat16 probabilities[kTile][kProbabilityStride];\n"
        "+wmma::load_matrix_sync(\n"
        "+    probability_frag, &probabilities[0][0], kProbabilityStride);\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_probability_stride20_skew_repeat() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Skew the MMA probability tile with a stride-20 layout."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,8 @@\n"
        "-old\n"
        "+constexpr int kProbabilityStride = 20;\n"
        "+__shared__ __nv_bfloat16 probabilities[kTile][kProbabilityStride];\n"
        "+wmma::load_matrix_sync(\n"
        "+    probability_frag, &probabilities[0][0], kProbabilityStride);\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_2d_probability_index_without_declaration() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Skew the MMA score tile and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,5 @@\n"
        "-old\n"
        "+constexpr int kScoreStride = kTile + 8;\n"
        "+__shared__ float scores[kTile][kScoreStride];\n"
        "+probabilities[row][key] = __float2bfloat16(weight);\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma_score_skew"
    )

    with pytest.raises(ValueError, match="symbol_lifecycle"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_regressed_mma_score_stride_skew() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Skew the MMA score tile and score it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,8 @@\n"
        "-old\n"
        "+constexpr int kScoreStride = 24;\n"
        "+__shared__ float scores[kTile][kScoreStride];\n"
        "+wmma::store_matrix_sync(\n"
        "+    &scores[0][0], score_frag, kScoreStride, wmma::mem_row_major);\n"
    )
    payload["next_command"] = (
        "avo score --backend candidate "
        "--candidate candidates/cuda_mma_attention_seed.py "
        "--seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128"
    )

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
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


def test_parse_variation_decision_rejects_patch_described_as_unused() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Add async copy wrappers and compile the source."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+#include <cuda_pipeline_primitives.h>\n"
    )
    payload["risk"] = "The helpers are unused in this patch, so there is no dataflow change."

    with pytest.raises(ValueError, match="known invalid"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_unused_async_copy_wrappers() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Add async copy wrappers and compile the source."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,11 @@\n"
        "-old\n"
        "+__device__ __forceinline__ void async_copy_16(void* dst, const void* src) {\n"
        "+  __pipeline_memcpy_async(dst, src, 16);\n"
        "+}\n"
        "+__device__ __forceinline__ void async_commit() {\n"
        "+  __pipeline_commit();\n"
        "+}\n"
        "+__device__ __forceinline__ void async_wait(int prior) {\n"
        "+  __pipeline_wait_prior(prior);\n"
        "+}\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="no_effect_async_helpers"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_async_helper_inside_mma_signature() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Add an async copy helper before the MMA kernel."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,8 @@\n"
        "-old\n"
        "+__device__ __forceinline__ void async_copy_16(void* dst, const void* src) {\n"
        "+  __pipeline_memcpy_async(dst, src, 16);\n"
        "+}\n"
        "+__global__ void mma_attention_kernel(const __nv_bfloat16* q) {\n"
        "+  async_copy_16(nullptr, nullptr);\n"
        "+}\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="cuda_helper_placement"):
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


def test_parse_variation_decision_rejects_patch_described_as_no_effect() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Add a warp-row WMMA build check and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+#include <mma.h>\n"
    )
    payload["expected_effect"] = (
        "This compile-only patch does not affect correctness or throughput."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "--out-dir build/warp"
    )

    with pytest.raises(ValueError, match="does not affect correctness or throughput"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_unused_wmma_compile_skeleton() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Add a warp-row WMMA skeleton and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_warp_rows_attention/attention_kernel.cu\n"
        "@@ -1 +1,8 @@\n"
        "-old\n"
        "+#include <mma.h>\n"
        "+__shared__ __nv_bfloat16 q_shared[kRowsPerBlock][kMaxHeadDim];\n"
        "+__shared__ __nv_bfloat16 k_shared[kTileKeys][kMaxHeadDim];\n"
        "+wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_frag;\n"
        "+wmma::fill_fragment(score_frag, 0.0f);\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu "
        "--out-dir build/warp"
    )

    with pytest.raises(ValueError, match="no_effect_wmma_skeleton"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_patch_described_as_nvcc_compile_failure() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA PV preload and compile it."
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
        "The duplicate store line is a diff structure error and will cause NVCC compile "
        "failure."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="known invalid"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_patch_marked_do_not_apply() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Add MMA warp reduction helpers and compile it."
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
        "The helper functions are inserted inside the kernel signature. This will cause "
        "NVCC to fail with syntax errors. Do not apply this patch; correct the helper "
        "placement first."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="do not apply this patch"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_stray_probability_frag_statement() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Patch MMA PV probability preload and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,8 @@\n"
        "-old\n"
        "+wmma::fragment<wmma::matrix_a, kTile, 16, kTile, __nv_bfloat16, "
        "wmma::row_major> probability_frag_next;\n"
        "+wmma::load_matrix_sync(probability_frag_next, probabilities, kTile);\n"
        "+probability_frag;\n"
        "+probability_frag = probability_frag_next;\n"
        "+wmma::mma_sync(output_frag, probability_frag, v_frag, output_frag);\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="symbol_lifecycle"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_unused_q_frag_preload_skeleton() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Add a Q-fragment double-buffer skeleton and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1,8 @@\n"
        "-old\n"
        "+wmma::fragment<wmma::matrix_a, kTile, kTile, 16, __nv_bfloat16, "
        "wmma::row_major> q_frag_next;\n"
        "+const bool has_next_tile = next_query_tile < query_tiles;\n"
        "+if (has_next_tile) {\n"
        "+  wmma::load_matrix_sync(q_frag_next, q + base, kHeadDim);\n"
        "+}\n"
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
    )

    with pytest.raises(ValueError, match="no_effect_wmma_skeleton"):
        parse_decision_text(json.dumps(payload))


def test_parse_variation_decision_rejects_compile_only_structural_probe_text() -> None:
    payload = decision_payload()
    payload["candidate_edit"] = "Add a Q-fragment double-buffer skeleton and compile it."
    payload["candidate_patch"] = (
        "diff --git a/candidates/cuda_mma_attention/attention_kernel.cu "
        "b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "--- a/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "+++ b/candidates/cuda_mma_attention/attention_kernel.cu\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    payload["expected_effect"] = (
        "This is a compile-only structural probe. It does not yet consume q_frag_next "
        "and must not be scored."
    )
    payload["next_command"] = (
        "avo compile --source candidates/cuda_mma_attention/attention_kernel.cu "
        "--out-dir build/mma"
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

    with pytest.raises(ValueError, match="must not edit CUDA source files directly"):
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

    with pytest.raises(ValueError, match="incomplete_or_malformed_edit"):
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

    with pytest.raises(ValueError, match="incomplete_or_malformed_edit"):
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
    assert "candidate_transform" in tool["input_schema"]["properties"]
    assert "add_include" in tool["input_schema"]["properties"]["candidate_transform"][
        "properties"
    ]["op"]["enum"]
    assert "replace_once" in tool["input_schema"]["properties"]["candidate_transform"][
        "properties"
    ]["op"]["enum"]


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
    assert "Target workload is realistic long-sequence BF16 attention" in context
    assert "Unpatched seed caps are safety fences, not search targets" in context
    assert "real optimization steps should move toward the target workload" in context
    assert "Use avo env only for CUDA/build environment diagnostics" in context
    assert "Use avo compile only for CUDA build/compilation diagnostics" in context
    assert "No-patch compiles of existing CUDA candidates" in context
    assert "candidates/cuda_mma_attention/attention_kernel.cu" in context
    assert "Preferred edit channel: candidate_transform" in context
    assert "Legacy candidate_patch raw diffs are allowed only for non-CUDA" in context
    assert ".cu/.cuh kernel edits must use candidate_transform" in context
    assert "at most four exact transform steps" in context
    assert "do not describe a broad kernel rewrite without candidate_transform" in context
    assert "Structural CUDA preflight tracks are class-oriented hard checks" in context
    assert "wrapper/kernel shape-contract consistency" in context
    assert "tensor-core fragment declarations must match" in context
    assert "staged shared-memory addresses must stay tile-local" in context
    assert "After a shape-support compile succeeds" in context
    assert "--candidate candidates/cuda_mma_attention_seed.py" in context
    assert "--seq-lens 4096,8192,16384,32768" in context
    assert "candidate_transform" in context
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
    assert "candidate_transform" in prompt
    assert "No-edit mode" in prompt
    assert "Supported ops are add_include" in prompt
    assert "at most four exact transform steps" in prompt
    assert "do not describe a broad CUDA rewrite in candidate_edit" in prompt
    assert "Legacy edit mode" in prompt
    assert "must not edit .cu/.cuh kernel sources directly" in prompt
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
    assert "avoid repeating failed or regressed transform families" in prompt


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
        ValueError(
            "candidate_transform or candidate_patch must be provided when candidate_edit "
            "describes a code change"
        ),
    )

    content = updated["messages"][0]["content"]
    assert "candidate_transform is one structured operation" in content
    assert "candidate_patch must be a raw git-style unified diff" in content
    assert "Choose exactly one valid mode" in content
    assert "'No edit;'" in content
    assert "diff --git" in content
    assert "Do not mention fixing" in content
    assert "exact candidate_transform object from the follow-up signal" in content
    assert "Do not restate a broad CUDA change" in content
    assert "at-most-four-step batch" in content


def test_decision_feedback_explains_mutually_exclusive_edit_channels() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError("candidate_patch and candidate_transform are mutually exclusive"),
    )

    content = updated["messages"][0]["content"]
    assert "Choose one edit channel only" in content
    assert "candidate_patch to exactly the empty string" in content
    assert "omit candidate_transform" in content


def test_decision_feedback_explains_no_edit_with_payload_error() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError("candidate_edit starts in no-edit mode but includes an edit payload"),
    )

    content = updated["messages"][0]["content"]
    assert "No-edit mode cannot include candidate_transform or candidate_patch" in content
    assert "remove the 'No edit;' prefix" in content


def test_decision_feedback_explains_raw_cuda_patch_error() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError("candidate_patch must not edit CUDA source files directly"),
    )

    content = updated["messages"][0]["content"]
    assert "Do not use raw candidate_patch for .cu or .cuh files" in content
    assert "Express the CUDA edit as candidate_transform" in content
    assert "add_include" in content
    assert "op=batch" in content
    assert "Raw candidate_patch is only for non-CUDA candidate files" in content


def test_decision_feedback_explains_missing_required_keys_error() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError(
            "variation decision missing required keys: expected_effect, risk, next_command"
        ),
    )

    content = updated["messages"][0]["content"]
    assert "complete variation decision object" in content
    assert "expected_effect" in content
    assert "risk" in content
    assert "next_command" in content
    assert "Do not omit" in content


def test_decision_feedback_explains_scalar_bf16_async_copy_error() -> None:
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
        ValueError(
            "candidate_patch uses scalar BF16 __pipeline_memcpy_async copies; use "
            "16-byte aligned groups for Ampere async copy patches"
        ),
    )

    content = updated["messages"][0]["content"]
    assert "valid Ampere async-copy transform" in content
    assert "async-copy structural contract" in content
    assert "small candidate_transform" in content
    assert "different transform family" in content


def test_decision_feedback_explains_self_invalid_patch_error() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError(
            "candidate_patch is described as known invalid by the decision itself; "
            "found phrase 'will cause a compile error'"
        ),
    )

    content = updated["messages"][0]["content"]
    assert "classified its own edit as invalid" in content
    assert "different transform family" in content
    assert "structured candidate_transform" in content
    assert "known compile, correctness, skeleton, or incomplete-edit failure" in content


def test_decision_feedback_explains_must_not_be_scored_error() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError(
            "candidate_patch is described as known invalid by the decision itself; "
            "found phrase 'must not be scored'"
        ),
    )

    content = updated["messages"][0]["content"]
    assert "must not be scored" in content
    assert "classified its own edit as invalid" in content
    assert "executable edit" in content
    assert "known compile, correctness, skeleton, or incomplete-edit failure" in content


def test_decision_feedback_explains_compile_out_dir_error() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError("next_command --out-dir must be under: build"),
    )

    content = updated["messages"][0]["content"]
    assert "--out-dir to a repo-relative build subdirectory" in content
    assert "Do not write compiler outputs under candidates/" in content


def test_decision_feedback_explains_recorded_env_stability_error() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError(
            "next_command repeats a recorded environment stability diagnostic; use avo env "
            "only after a concrete CUDA/build environment failure"
        ),
    )

    content = updated["messages"][0]["content"]
    assert "already-recorded CUDA/build environment" in content
    assert "concrete recent build or environment failure" in content
    assert "CUDA version mismatch" in content


def test_decision_feedback_explains_planner_recovery_env_error() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError(
            "next_command avo env is not useful for a planner-interface failure; return "
            "a valid candidate_transform or a kernel-search diagnostic instead"
        ),
    )

    content = updated["messages"][0]["content"]
    assert "Do not spend a loop step on avo env for planner-interface" in content
    assert "valid candidate_transform" in content


def test_decision_feedback_explains_recorded_no_patch_compile_error() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError(
            "next_command repeats a recorded no-patch compile diagnostic; include "
            "candidate_transform/candidate_patch to build-check a change or run a bounded "
            "score instead"
        ),
    )

    content = updated["messages"][0]["content"]
    assert "Do not retry a no-edit compile" in content
    assert "include candidate_transform" in content
    assert "set_constexpr_int" in content


def test_decision_feedback_explains_unpatched_mma_score_error() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError(
            "next_command repeats a recorded unpatched MMA seed score; include "
            "candidate_transform/candidate_patch to change kernel structure before scoring"
        ),
    )

    content = updated["messages"][0]["content"]
    assert "Do not retry a no-edit score that is already in lineage" in content
    assert "candidate_transform, preferably a small wrapper/kernel batch" in content
    assert "make a real kernel-structure change" in content
    assert "raw candidate_patch cannot edit CUDA kernel sources" in content


def test_decision_feedback_explains_wrapper_only_mma_shape_error() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError(
            "next_command scores a patched MMA shape extension beyond the current smoke cap; "
            "use a structured transform batch that updates both the wrapper cap and kernel "
            "cap/dataflow together before scoring larger shapes"
        ),
    )

    content = updated["messages"][0]["content"]
    assert "structured transform batch" in content
    assert "Wrapper-only cap edits are not enough" in content


def test_decision_feedback_explains_transform_cap_score_error() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError(
            "next_command scores MMA seq_lens beyond the transformed cap; "
            "max requested seq_len=2048 but kMaxSeqLen=1024"
        ),
    )

    content = updated["messages"][0]["content"]
    assert "within the cap expressed by the structured transform" in content
    assert "cover every requested seq_len" in content


def test_decision_feedback_explains_unpatched_warp_row_score_error() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError(
            "next_command repeats a recorded no-patch warp-row seed score; include "
            "candidate_transform/candidate_patch to change kernel structure before scoring"
        ),
    )

    content = updated["messages"][0]["content"]
    assert "Do not retry a no-edit score of a smoke-only seed workload" in content
    assert "candidate_transform or a legacy non-CUDA candidate_patch" in content


def test_decision_feedback_explains_structural_preflight_error() -> None:
    kwargs = {"messages": [{"role": "user", "content": "Base prompt."}]}

    updated = _decision_kwargs_with_feedback(
        kwargs,
        ValueError(
            "structural preflight track wmma_fragment_shape classified as "
            "unsupported_wmma_shape: unsupported fragment shape"
        ),
    )

    content = updated["messages"][0]["content"]
    assert "failed transform family" in content
    assert "smaller candidate_transform operation or batch" in content
    assert "avoids the same structural class" in content


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
