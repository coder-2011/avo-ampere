import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from avo.agent import VariationDecision
from avo.evolve import (
    CommandResult,
    EvolutionStep,
    PatchResult,
    VariationAttempt,
    _extract_score_payload,
    apply_candidate_patch,
    attempt_has_repairable_compile_failure,
    attempt_has_repairable_correctness_failure,
    cleanup_rejected_candidate_patch,
    command_from_decision,
    correctness_failure_class_for_attempt,
    finalize_attempt,
    load_promoted_preflight_classes,
    materialize_candidate_transform,
    paths_from_unified_diff,
    pending_compile_only_transform,
    planning_failure_step,
    revert_candidate_patch,
    run_decision_command,
    summarize_attempt_history,
    update_promoted_preflight_tracks,
    validate_decision_against_attempt_history,
    write_attempt,
    write_step,
    write_step_record,
)
from avo.lineage import GateDecision, best_geomean


def decision(
    next_command: str,
    *,
    candidate_patch: str = "",
    candidate_transform: dict[str, object] | None = None,
) -> VariationDecision:
    return VariationDecision(
        hypothesis="validate the execution substrate",
        files_to_inspect=["avo/evolve.py"],
        candidate_edit="run a bounded command",
        expected_effect="records an attempt without shell execution",
        risk="command may fail",
        next_command=next_command,
        candidate_patch=candidate_patch,
        candidate_transform=candidate_transform,
    )


def test_command_from_decision_rewrites_avo_to_module() -> None:
    command = command_from_decision(decision("avo score --backend torch-sdpa"))

    assert command[:3] == [sys.executable, "-m", "avo"]
    assert command[3:] == ["score", "--backend", "torch-sdpa"]


def test_command_from_decision_allows_profile() -> None:
    command = command_from_decision(
        decision(
            "avo profile --backend candidate --candidate candidates/cuda_mma_attention_seed.py "
            "--seq-lens 4096"
        )
    )

    assert command[:3] == [sys.executable, "-m", "avo"]
    assert command[3] == "profile"


def test_command_from_decision_rejects_shell() -> None:
    with pytest.raises(ValueError, match="must start with 'avo'"):
        command_from_decision(decision("rm -rf /"))

    with pytest.raises(ValueError, match="shell control"):
        command_from_decision(decision("avo env && rm -rf /"))


def test_command_from_decision_rejects_unsupported_subcommand() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        command_from_decision(decision("avo commit-score lineage score.json"))


def test_paths_from_unified_diff_extracts_candidate_paths() -> None:
    patch = candidate_value_patch()

    assert paths_from_unified_diff(patch) == ["candidates/seed.py"]


def test_apply_candidate_patch_dry_run_does_not_modify_file(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    patch = candidate_value_patch()

    result = apply_candidate_patch(patch, cwd=tmp_path, dry_run=True)

    assert result.ok
    assert result.patch_paths == ["candidates/seed.py"]
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_apply_candidate_patch_updates_candidate_file(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)

    result = apply_candidate_patch(candidate_value_patch(), cwd=tmp_path)

    assert result.ok
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_materialize_candidate_transform_generates_candidate_patch(tmp_path: Path) -> None:
    write_seed_candidate(tmp_path)

    patch = materialize_candidate_transform(
        {
            "op": "replace_once",
            "path": "candidates/seed.py",
            "find": "VALUE = 1",
            "replace": "VALUE = 2",
        },
        cwd=tmp_path,
    )

    assert patch.startswith("diff --git a/candidates/seed.py b/candidates/seed.py")
    assert "-VALUE = 1" in patch
    assert "+VALUE = 2" in patch


def test_materialize_candidate_transform_generates_batch_patch(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    seed.write_text("VALUES = {1}\n", encoding="utf-8")
    kernel = tmp_path / "candidates" / "kernel.cu"
    kernel.write_text("constexpr int kMaxSeqLen = 256;\n", encoding="utf-8")

    patch = materialize_candidate_transform(
        {
            "op": "batch",
            "steps": [
                {
                    "op": "set_constexpr_int",
                    "path": "candidates/kernel.cu",
                    "name": "kMaxSeqLen",
                    "value": 512,
                },
                {
                    "op": "add_int_to_python_set",
                    "path": "candidates/seed.py",
                    "name": "VALUES",
                    "value": 2,
                },
            ],
        },
        cwd=tmp_path,
    )

    assert "diff --git a/candidates/kernel.cu b/candidates/kernel.cu" in patch
    assert "+constexpr int kMaxSeqLen = 512;" in patch
    assert "diff --git a/candidates/seed.py b/candidates/seed.py" in patch
    assert "+VALUES = {1, 2}" in patch


def test_materialize_candidate_transform_skips_noop_batch_step(
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "candidates" / "kernel.cu"
    kernel.parent.mkdir(parents=True)
    kernel.write_text(
        "constexpr int kTile = 16;\n__shared__ float row_max[kTile];\n",
        encoding="utf-8",
    )

    patch = materialize_candidate_transform(
        {
            "op": "batch",
            "steps": [
                {
                    "op": "replace_once",
                    "path": "candidates/kernel.cu",
                    "find": "constexpr int kTile = 16;",
                    "replace": "constexpr int kTile = 32;",
                },
                {
                    "op": "replace_once",
                    "path": "candidates/kernel.cu",
                    "find": "__shared__ float row_max[kTile];",
                    "replace": "__shared__ float row_max[kTile];",
                },
            ],
        },
        cwd=tmp_path,
    )

    assert "+constexpr int kTile = 32;" in patch
    assert "-constexpr int kTile = 16;" in patch


def test_materialize_candidate_transform_rejects_all_noop_batch(
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "candidates" / "kernel.cu"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("__shared__ float row_max[kTile];\n", encoding="utf-8")

    with pytest.raises(ValueError, match="transform produced no source change"):
        materialize_candidate_transform(
            {
                "op": "batch",
                "steps": [
                    {
                        "op": "replace_once",
                        "path": "candidates/kernel.cu",
                        "find": "__shared__ float row_max[kTile];",
                        "replace": "__shared__ float row_max[kTile];",
                    },
                ],
            },
            cwd=tmp_path,
        )


def test_materialize_candidate_transform_inserts_after_anchor_on_new_line(
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "candidates" / "kernel.cu"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("  __shared__ float old_scale[kTile];\n  int next = 0;\n", encoding="utf-8")

    patch = materialize_candidate_transform(
        {
            "op": "insert_after_once",
            "path": "candidates/kernel.cu",
            "anchor": "  __shared__ float old_scale[kTile];",
            "text": "  __shared__ __nv_bfloat16 v_tile[kTile * kHeadDim];",
        },
        cwd=tmp_path,
    )

    assert "+  __shared__ __nv_bfloat16 v_tile[kTile * kHeadDim];" in patch
    assert "old_scale[kTile];  __shared__" not in patch


def test_materialize_candidate_transform_inserts_before_anchor_on_own_line(
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "candidates" / "kernel.cu"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("  int next = 0;\n", encoding="utf-8")

    patch = materialize_candidate_transform(
        {
            "op": "insert_before_once",
            "path": "candidates/kernel.cu",
            "anchor": "  int next = 0;",
            "text": "  int inserted = 1;",
        },
        cwd=tmp_path,
    )

    assert "+  int inserted = 1;" in patch
    assert "inserted = 1;  int next" not in patch


def test_materialize_candidate_transform_adds_include(tmp_path: Path) -> None:
    kernel = tmp_path / "candidates" / "kernel.cu"
    kernel.parent.mkdir()
    kernel.write_text(
        "#include <cuda_bf16.h>\n#include <mma.h>\n\nnamespace test {}\n",
        encoding="utf-8",
    )

    patch = materialize_candidate_transform(
        {
            "op": "add_include",
            "path": "candidates/kernel.cu",
            "header": "cuda_pipeline_primitives.h",
        },
        cwd=tmp_path,
    )

    assert "+#include <cuda_pipeline_primitives.h>" in patch
    assert " #include <mma.h>\n+#include <cuda_pipeline_primitives.h>" in patch


def test_materialize_candidate_transform_rejects_duplicate_include(tmp_path: Path) -> None:
    kernel = tmp_path / "candidates" / "kernel.cu"
    kernel.parent.mkdir()
    kernel.write_text("#include <cuda_pipeline_primitives.h>\n", encoding="utf-8")

    with pytest.raises(ValueError, match="produced no source change"):
        materialize_candidate_transform(
            {
                "op": "add_include",
                "path": "candidates/kernel.cu",
                "header": "<cuda_pipeline_primitives.h>",
            },
            cwd=tmp_path,
        )


def test_materialize_candidate_transform_rejects_non_object_batch_step(tmp_path: Path) -> None:
    write_seed_candidate(tmp_path).write_text("VALUES = {1}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="batch transform steps must be objects"):
        materialize_candidate_transform(
            {
                "op": "batch",
                "steps": ["not-a-step"],
            },
            cwd=tmp_path,
        )


def test_materialize_candidate_transform_rejects_ambiguous_anchor(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    seed.write_text("VALUE = 1\nVALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="matching start lines: 1, 2"):
        materialize_candidate_transform(
            {
                "op": "replace_once",
                "path": "candidates/seed.py",
                "find": "VALUE = 1",
                "replace": "VALUE = 2",
            },
            cwd=tmp_path,
        )


def test_materialize_candidate_transform_reports_ambiguous_insert_lines(
    tmp_path: Path,
) -> None:
    seed = write_seed_candidate(tmp_path)
    seed.write_text(
        "if (threadIdx.x < warpSize) {\nbody\nif (threadIdx.x < warpSize) {\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="matching start lines: 1, 3"):
        materialize_candidate_transform(
            {
                "op": "insert_before_once",
                "path": "candidates/seed.py",
                "anchor": "if (threadIdx.x < warpSize) {",
                "text": "// inserted\n",
            },
            cwd=tmp_path,
        )


def test_apply_candidate_patch_recounts_llm_hunk_lengths(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    patch = dedent(
        """\
        diff --git a/candidates/seed.py b/candidates/seed.py
        --- a/candidates/seed.py
        +++ b/candidates/seed.py
        @@ -1,3 +1,3 @@
        -VALUE = 1
        +VALUE = 2
        """
    )

    result = apply_candidate_patch(patch, cwd=tmp_path)

    assert result.ok
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_revert_candidate_patch_restores_candidate_file(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    apply_result = apply_candidate_patch(candidate_value_patch(), cwd=tmp_path)

    revert_result = revert_candidate_patch(candidate_value_patch(), cwd=tmp_path)

    assert apply_result.ok
    assert revert_result.ok
    assert revert_result.patch_paths == ["candidates/seed.py"]
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_apply_candidate_patch_rejects_non_candidate_path() -> None:
    result = apply_candidate_patch(
        dedent(
            """\
            diff --git a/README.md b/README.md
            --- a/README.md
            +++ b/README.md
            @@ -1 +1 @@
            -old
            +new
            """
        ),
        cwd=Path.cwd(),
    )

    assert_patch_rejected(result, "under: candidates")


def test_apply_candidate_patch_rejects_path_traversal() -> None:
    result = apply_candidate_patch(
        dedent(
            """\
            diff --git a/candidates/../evil.py b/candidates/../evil.py
            --- a/candidates/../evil.py
            +++ b/candidates/../evil.py
            @@ -1 +1 @@
            -old
            +new
            """
        ),
        cwd=Path.cwd(),
    )

    assert_patch_rejected(result, "path traversal")


def test_apply_candidate_patch_rejects_symlink_patch() -> None:
    result = apply_candidate_patch(
        dedent(
            """\
            diff --git a/candidates/link b/candidates/link
            new file mode 120000
            index 0000000..e69de29
            --- /dev/null
            +++ b/candidates/link
            @@ -0,0 +1 @@
            +../outside
            """
        ),
        cwd=Path.cwd(),
    )

    assert_patch_rejected(result, "unsupported patch marker")


def test_apply_candidate_patch_rejects_existing_symlink_path(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (candidate_dir / "link").symlink_to(outside, target_is_directory=True)

    result = apply_candidate_patch(
        dedent(
            """\
            diff --git a/candidates/link/seed.py b/candidates/link/seed.py
            --- a/candidates/link/seed.py
            +++ b/candidates/link/seed.py
            @@ -1 +1 @@
            -old
            +new
            """
        ),
        cwd=tmp_path,
    )

    assert_patch_rejected(result, "existing symlink")


def test_apply_candidate_patch_rejects_binary_patch() -> None:
    result = apply_candidate_patch(
        dedent(
            """\
            diff --git a/candidates/blob.bin b/candidates/blob.bin
            new file mode 100644
            index 0000000..1234567
            GIT binary patch
            literal 0
            HcmV?d00001
            """
        ),
        cwd=Path.cwd(),
    )

    assert_patch_rejected(result, "unsupported patch marker")


def test_apply_candidate_patch_rejects_delete_patch() -> None:
    result = apply_candidate_patch(
        dedent(
            """\
            diff --git a/candidates/seed.py b/candidates/seed.py
            deleted file mode 100644
            index 1234567..0000000
            --- a/candidates/seed.py
            +++ /dev/null
            @@ -1 +0,0 @@
            -VALUE = 1
            """
        ),
        cwd=Path.cwd(),
    )

    assert_patch_rejected(result, "unsupported patch marker")


def test_apply_candidate_patch_rejects_empty_patch() -> None:
    result = apply_candidate_patch("", cwd=Path.cwd())

    assert_patch_rejected(result, "at least one diff")


def test_apply_candidate_patch_rejects_non_git_unified_diff() -> None:
    result = apply_candidate_patch(
        dedent(
            """\
            --- a/candidates/seed.py
            +++ b/candidates/seed.py
            @@ -1 +1 @@
            -VALUE = 1
            +VALUE = 2
            """
        ),
        cwd=Path.cwd(),
    )

    assert_patch_rejected(result, "diff --git")


def test_run_decision_command_executes_allowed_command() -> None:
    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0"),
        cwd=Path.cwd(),
        timeout_s=10,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )

    assert attempt.command_result.ok
    assert "AVO_RESULT_JSON" in attempt.command_result.stdout_tail
    assert attempt.patch_result is None


def test_run_decision_command_applies_candidate_patch_before_command(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0", candidate_patch=candidate_value_patch()),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )

    assert attempt.patch_result is not None
    assert attempt.patch_result.ok
    assert attempt.command_result.ok
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_run_decision_command_materializes_transform_before_command(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    attempt = run_decision_command(
        decision(
            "avo worker-sleep --seconds 0",
            candidate_transform={
                "op": "replace_once",
                "path": "candidates/seed.py",
                "find": "VALUE = 1",
                "replace": "VALUE = 2",
            },
        ),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )

    assert attempt.patch_result is not None
    assert attempt.patch_result.ok
    assert attempt.command_result.ok
    assert attempt.decision.candidate_transform is not None
    assert attempt.decision.candidate_patch == ""
    assert attempt.materialized_patch is not None
    assert attempt.materialized_patch.startswith("diff --git")
    assert attempt.as_dict()["decision"]["candidate_patch"] == ""
    assert attempt.as_dict()["materialized_patch"].startswith("diff --git")
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_run_decision_command_materializes_block_transform_before_command(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "candidates" / "seed.py"
    seed.parent.mkdir(parents=True)
    seed.write_text(
        "def helper():\n"
        "    value = 1\n"
        "    return value\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    attempt = run_decision_command(
        decision(
            "avo worker-sleep --seconds 0",
            candidate_transform={
                "op": "replace_block_once",
                "path": "candidates/seed.py",
                "find": "def helper():\n    value = 1\n    return value\n",
                "replace": "def helper():\n    value = 2\n    return value\n",
            },
        ),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )

    assert attempt.patch_result is not None
    assert attempt.patch_result.ok
    assert attempt.command_result.ok
    assert attempt.materialized_patch is not None
    assert "value = 2" in attempt.materialized_patch
    assert seed.read_text(encoding="utf-8") == (
        "def helper():\n"
        "    value = 2\n"
        "    return value\n"
    )


def test_run_decision_command_preflights_materialized_cuda_transform(tmp_path: Path) -> None:
    kernel = tmp_path / "candidates" / "kernel.cu"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("old\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    attempt = run_decision_command(
        decision(
            "avo worker-sleep --seconds 0",
            candidate_transform={
                "op": "replace_once",
                "path": "candidates/kernel.cu",
                "find": "old",
                "replace": "wmma::fragment<wmma::accumulator, 32, 16, 16, float> frag;\n",
            },
        ),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )

    assert attempt.patch_result is not None
    assert not attempt.patch_result.ok
    assert "structural preflight track wmma_fragment_shape" in str(
        attempt.patch_result.rejected_reason
    )
    assert attempt.command_result.returncode is None
    assert kernel.read_text(encoding="utf-8") == "old\n"


def test_run_decision_command_records_soft_cuda_advisories(tmp_path: Path) -> None:
    kernel = tmp_path / "candidates" / "kernel.cu"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("old\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    attempt = run_decision_command(
        decision(
            "avo worker-sleep --seconds 0",
            candidate_transform={
                "op": "replace_once",
                "path": "candidates/kernel.cu",
                "find": "old",
                "replace": "__pipeline_memcpy_async(dst, src, sizeof(__nv_bfloat16));",
            },
        ),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )

    assert attempt.patch_result is not None
    assert attempt.patch_result.ok
    assert attempt.command_result.ok
    assert any(
        "async_copy_granularity_preference" in advisory
        for advisory in attempt.patch_result.advisories
    )
    assert attempt.as_dict()["patch_result"]["advisories"]


def test_run_decision_command_rejects_materialized_unused_shared_buffer(
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "candidates" / "kernel.cu"
    kernel.parent.mkdir(parents=True)
    kernel.write_text(
        "__global__ void kernel() {\n"
        "  __shared__ float scores[kScoreElements];\n"
        "}\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    attempt = run_decision_command(
        decision(
            "avo worker-sleep --seconds 0",
            candidate_transform={
                "op": "replace_once",
                "path": "candidates/kernel.cu",
                "find": (
                    "__global__ void kernel() {\n"
                    "  __shared__ float scores[kScoreElements];\n"
                    "}\n"
                ),
                "replace": (
                    "__global__ void kernel() {\n"
                    "  __shared__ float scores[kScoreElements];\n"
                    "  __shared__ __nv_bfloat16 k_shared[2][kTile * kHeadDim];\n"
                    "}\n"
                ),
            },
        ),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )

    assert attempt.patch_result is not None
    assert not attempt.patch_result.ok
    assert "no_effect_shared_staging_buffer" in str(attempt.patch_result.rejected_reason)
    assert attempt.command_result.returncode is None
    assert "__nv_bfloat16 k_shared" not in kernel.read_text(encoding="utf-8")


def test_run_decision_command_applies_promoted_preflight_to_transform(
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "candidates" / "kernel.cu"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("float old_scale = 1.0f;\nacc *= old_scale;\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    attempt = run_decision_command(
        decision(
            "avo worker-sleep --seconds 0",
            candidate_transform={
                "op": "replace_once",
                "path": "candidates/kernel.cu",
                "find": "float old_scale = 1.0f;\nacc *= old_scale;\n",
                "replace": "float new_scale = 1.0f;\nacc += old_scale;\n",
            },
        ),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
        promoted_preflight_classes=frozenset({"stale_or_undefined_symbol"}),
    )

    assert attempt.patch_result is not None
    assert not attempt.patch_result.ok
    assert "structural promoted preflight" in str(attempt.patch_result.rejected_reason)
    assert "promoted_symbol_lifecycle_removed_declaration" in str(
        attempt.patch_result.rejected_reason
    )
    assert attempt.command_result.returncode is None
    assert kernel.read_text(encoding="utf-8") == "float old_scale = 1.0f;\nacc *= old_scale;\n"


def test_run_decision_command_stops_when_candidate_patch_is_rejected(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    attempt = run_decision_command(
        decision(
            "avo worker-sleep --seconds 0",
            candidate_patch=dedent(
                """\
                diff --git a/README.md b/README.md
                --- a/README.md
                +++ b/README.md
                @@ -1 +1 @@
                -old
                +new
                """
            ),
        ),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )

    assert attempt.patch_result is not None
    assert not attempt.patch_result.ok
    assert not attempt.command_result.ok
    assert attempt.command_result.returncode is None
    assert "candidate patch rejected" in attempt.command_result.stderr_tail


def test_cleanup_rejected_candidate_patch_reverts_nonaccepted_patch(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0", candidate_patch=candidate_value_patch()),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )
    step = EvolutionStep(attempt=attempt, gate_decision=None)

    cleaned = cleanup_rejected_candidate_patch(step, cwd=tmp_path)

    assert cleaned.patch_cleanup_result is not None
    assert cleaned.patch_cleanup_result.ok
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_cleanup_rejected_transform_uses_materialized_patch(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    attempt = run_decision_command(
        decision(
            "avo worker-sleep --seconds 0",
            candidate_transform={
                "op": "replace_once",
                "path": "candidates/seed.py",
                "find": "VALUE = 1",
                "replace": "VALUE = 2",
            },
        ),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )
    step = EvolutionStep(attempt=attempt, gate_decision=None)

    cleaned = cleanup_rejected_candidate_patch(step, cwd=tmp_path)

    assert attempt.decision.candidate_patch == ""
    assert attempt.materialized_patch is not None
    assert cleaned.patch_cleanup_result is not None
    assert cleaned.patch_cleanup_result.ok
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_cleanup_rejected_candidate_patch_rejects_dirty_patch_path(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    seed.write_text("VALUE = 1\nOTHER = 1\n", encoding="utf-8")
    init_git_repo(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    patch = candidate_value_patch_with_context()
    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0", candidate_patch=patch),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )
    seed.write_text("VALUE = 2\nOTHER = 1\nEXTRA = 3\n", encoding="utf-8")
    step = EvolutionStep(attempt=attempt, gate_decision=None)

    cleaned = cleanup_rejected_candidate_patch(step, cwd=tmp_path)

    assert cleaned.patch_cleanup_result is not None
    assert not cleaned.patch_cleanup_result.ok
    assert cleaned.patch_cleanup_result.rejected_reason is not None
    assert "left paths dirty" in cleaned.patch_cleanup_result.rejected_reason
    assert seed.read_text(encoding="utf-8") == "VALUE = 1\nOTHER = 1\nEXTRA = 3\n"


def test_cleanup_rejected_candidate_patch_keeps_accepted_patch(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    score = {"backend": "mock", "all_correct": True, "geomean_tflops": 3.0, "cases": [{}]}
    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0", candidate_patch=candidate_value_patch()),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )
    accepted_attempt = VariationAttempt(
        decision=attempt.decision,
        command_result=attempt.command_result,
        started_at=attempt.started_at,
        completed_at=attempt.completed_at,
        score_payload=score,
        patch_result=attempt.patch_result,
    )
    step = finalize_attempt(tmp_path / "lineage", accepted_attempt)

    cleaned = cleanup_rejected_candidate_patch(step, cwd=tmp_path)

    assert cleaned.accepted
    assert cleaned.patch_cleanup_result is None
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_finalize_attempt_snapshots_accepted_patch_sources(tmp_path: Path) -> None:
    seed = write_seed_candidate(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0", candidate_patch=candidate_value_patch()),
        cwd=tmp_path,
        timeout_s=10,
        env=env,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )
    accepted_attempt = VariationAttempt(
        decision=attempt.decision,
        command_result=attempt.command_result,
        started_at=attempt.started_at,
        completed_at=attempt.completed_at,
        score_payload={
            "backend": "mock",
            "all_correct": True,
            "geomean_tflops": 3.0,
            "cases": [{}],
        },
        patch_result=attempt.patch_result,
    )

    step = finalize_attempt(tmp_path / "lineage", accepted_attempt, source_root=tmp_path)

    assert step.accepted
    assert seed.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (tmp_path / "lineage" / "sources" / "latest" / "candidates" / "seed.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 2\n"
    assert (tmp_path / "lineage" / "patches" / "latest.patch").read_text(
        encoding="utf-8"
    ) == candidate_value_patch()


def test_finalize_attempt_snapshots_scored_candidate_sources_without_patch(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates"
    companion_dir = candidate_dir / "cuda_demo"
    companion_dir.mkdir(parents=True)
    seed = candidate_dir / "cuda_demo_seed.py"
    seed.write_text("from candidates.cuda_demo import attention\n", encoding="utf-8")
    (companion_dir / "attention.cpp").write_text("// cpp binding\n", encoding="utf-8")
    (companion_dir / "attention_kernel.cu").write_text("// cuda kernel\n", encoding="utf-8")
    (companion_dir / "compiled.so").write_bytes(b"not source")
    (candidate_dir / "__pycache__").mkdir()
    (candidate_dir / "__pycache__" / "cuda_demo_seed.pyc").write_bytes(b"cache")
    attempt = VariationAttempt(
        decision=decision("avo score --backend candidate --candidate candidates/cuda_demo_seed.py"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        score_payload={
            "backend": "candidate",
            "candidate_path": "candidates/cuda_demo_seed.py",
            "all_correct": True,
            "geomean_tflops": 3.0,
            "cases": [{}],
        },
    )

    step = finalize_attempt(tmp_path / "lineage", attempt, source_root=tmp_path)

    assert step.accepted
    source_root = tmp_path / "lineage" / "sources" / "latest"
    assert (source_root / "candidates" / "cuda_demo_seed.py").read_text(
        encoding="utf-8"
    ) == seed.read_text(encoding="utf-8")
    assert (source_root / "candidates" / "cuda_demo" / "attention.cpp").read_text(
        encoding="utf-8"
    ) == "// cpp binding\n"
    assert (source_root / "candidates" / "cuda_demo" / "attention_kernel.cu").read_text(
        encoding="utf-8"
    ) == "// cuda kernel\n"
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    manifest_paths = {entry["path"] for entry in manifest["files"]}
    assert "candidates/cuda_demo_seed.py" in manifest_paths
    assert "candidates/cuda_demo/attention.cpp" in manifest_paths
    assert "candidates/cuda_demo/attention_kernel.cu" in manifest_paths
    assert not (source_root / "candidates" / "cuda_demo" / "compiled.so").exists()
    assert not (source_root / "candidates" / "__pycache__").exists()
    assert not (tmp_path / "lineage" / "patches" / "latest.patch").exists()


def test_finalize_attempt_snapshots_local_python_import_dependencies(
    tmp_path: Path,
) -> None:
    candidate_dir = tmp_path / "candidates"
    shared_pkg = candidate_dir / "shared_pkg"
    shared_pkg.mkdir(parents=True)
    seed = candidate_dir / "cuda_demo_seed.py"
    seed.write_text(
        "from candidates import shared_helper\n"
        "from candidates.shared_pkg import helper\n",
        encoding="utf-8",
    )
    (candidate_dir / "shared_helper.py").write_text("SCALE = 1\n", encoding="utf-8")
    (shared_pkg / "__init__.py").write_text("from .helper import VALUE\n", encoding="utf-8")
    (shared_pkg / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    (shared_pkg / "helper_kernel.cu").write_text("// package cuda helper\n", encoding="utf-8")
    attempt = VariationAttempt(
        decision=decision("avo score --backend candidate --candidate candidates/cuda_demo_seed.py"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        score_payload={
            "backend": "candidate",
            "candidate_path": "candidates/cuda_demo_seed.py",
            "all_correct": True,
            "geomean_tflops": 3.0,
            "cases": [{}],
        },
    )

    step = finalize_attempt(tmp_path / "lineage", attempt, source_root=tmp_path)

    assert step.accepted
    source_root = tmp_path / "lineage" / "sources" / "latest"
    assert (source_root / "candidates" / "shared_helper.py").read_text(
        encoding="utf-8"
    ) == "SCALE = 1\n"
    assert (source_root / "candidates" / "shared_pkg" / "__init__.py").read_text(
        encoding="utf-8"
    ) == "from .helper import VALUE\n"
    assert (source_root / "candidates" / "shared_pkg" / "helper.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 2\n"
    assert (source_root / "candidates" / "shared_pkg" / "helper_kernel.cu").read_text(
        encoding="utf-8"
    ) == "// package cuda helper\n"


def test_finalize_attempt_snapshots_static_extension_sources_outside_companion(
    tmp_path: Path,
) -> None:
    candidate_dir = tmp_path / "candidates"
    extension_dir = candidate_dir / "shared_extension"
    extension_dir.mkdir(parents=True)
    seed = candidate_dir / "cuda_demo_seed.py"
    seed.write_text(
        "from pathlib import Path\n"
        "from torch.utils.cpp_extension import load\n"
        "EXTENSION_DIR = Path(__file__).resolve().parent / 'shared_extension'\n"
        "EXTENSION_SOURCES = [\n"
        "    str(EXTENSION_DIR / 'attention.cpp'),\n"
        "    str(EXTENSION_DIR / 'attention_kernel.cu'),\n"
        "]\n"
        "def _extension():\n"
        "    return load(name='demo', sources=EXTENSION_SOURCES)\n",
        encoding="utf-8",
    )
    (extension_dir / "attention.cpp").write_text("// cpp binding\n", encoding="utf-8")
    (extension_dir / "attention_kernel.cu").write_text("// cuda kernel\n", encoding="utf-8")
    attempt = VariationAttempt(
        decision=decision("avo score --backend candidate --candidate candidates/cuda_demo_seed.py"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        score_payload={
            "backend": "candidate",
            "candidate_path": "candidates/cuda_demo_seed.py",
            "all_correct": True,
            "geomean_tflops": 3.0,
            "cases": [{}],
        },
    )

    step = finalize_attempt(tmp_path / "lineage", attempt, source_root=tmp_path)

    assert step.accepted
    source_root = tmp_path / "lineage" / "sources" / "latest"
    assert (source_root / "candidates" / "shared_extension" / "attention.cpp").read_text(
        encoding="utf-8"
    ) == "// cpp binding\n"
    assert (source_root / "candidates" / "shared_extension" / "attention_kernel.cu").read_text(
        encoding="utf-8"
    ) == "// cuda kernel\n"


def test_finalize_attempt_snapshots_declared_runtime_source_files(
    tmp_path: Path,
) -> None:
    candidate_dir = tmp_path / "candidates"
    runtime_dir = candidate_dir / "runtime_extension"
    runtime_dir.mkdir(parents=True)
    seed = candidate_dir / "runtime_candidate.py"
    seed.write_text("def attention(q, k, v, causal):\n    return q\n", encoding="utf-8")
    kernel = runtime_dir / "generated_kernel.cu"
    header = runtime_dir / "generated_kernel.cuh"
    kernel.write_text("// generated cuda kernel\n", encoding="utf-8")
    header.write_text("// generated cuda header\n", encoding="utf-8")
    attempt = VariationAttempt(
        decision=decision(
            "avo score --backend candidate --candidate candidates/runtime_candidate.py"
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        score_payload={
            "backend": "candidate",
            "candidate_path": "candidates/runtime_candidate.py",
            "candidate_source_files": [
                kernel.as_posix(),
                "candidates/runtime_extension/generated_kernel.cuh",
            ],
            "all_correct": True,
            "geomean_tflops": 3.0,
            "cases": [{}],
        },
    )

    step = finalize_attempt(tmp_path / "lineage", attempt, source_root=tmp_path)

    assert step.accepted
    source_root = tmp_path / "lineage" / "sources" / "latest"
    assert (source_root / "candidates" / "runtime_candidate.py").read_text(
        encoding="utf-8"
    ) == seed.read_text(encoding="utf-8")
    assert (source_root / "candidates" / "runtime_extension" / "generated_kernel.cu").read_text(
        encoding="utf-8"
    ) == "// generated cuda kernel\n"
    assert (source_root / "candidates" / "runtime_extension" / "generated_kernel.cuh").read_text(
        encoding="utf-8"
    ) == "// generated cuda header\n"


def test_write_attempt_records_json(tmp_path: Path) -> None:
    attempt = run_decision_command(
        decision("avo worker-sleep --seconds 0"),
        cwd=Path.cwd(),
        timeout_s=10,
        allowed_subcommands=frozenset({"worker-sleep"}),
    )
    path = tmp_path / "attempt.json"

    write_attempt(path, attempt)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["decision"]["next_command"] == "avo worker-sleep --seconds 0"
    assert payload["command_result"]["ok"] is True


def test_extract_score_payload_from_score_wrapper_json() -> None:
    score = {"backend": "mock", "all_correct": True, "geomean_tflops": 3.0, "cases": []}
    stdout = json.dumps({"ok": True, "payload": score})

    assert _extract_score_payload(stdout) == score


def test_extract_score_payload_from_worker_result_line() -> None:
    score = {"backend": "mock", "all_correct": True, "geomean_tflops": 3.0, "cases": []}
    stdout = f"noise\nAVO_RESULT_JSON={json.dumps(score)}\n"

    assert _extract_score_payload(stdout) == score


def test_finalize_attempt_commits_score_payload(tmp_path: Path) -> None:
    score = {"backend": "mock", "all_correct": True, "geomean_tflops": 3.0, "cases": [{}]}
    attempt = VariationAttempt(
        decision=decision("avo score --backend torch-sdpa"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        score_payload=score,
    )

    step = finalize_attempt(tmp_path / "lineage", attempt)

    assert step.accepted
    assert best_geomean(tmp_path / "lineage") == 3.0


def test_finalize_attempt_without_score_payload_does_not_commit(tmp_path: Path) -> None:
    attempt = VariationAttempt(
        decision=decision("avo env"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "env"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )

    step = finalize_attempt(tmp_path / "lineage", attempt)

    assert step.gate_decision is None
    assert best_geomean(tmp_path / "lineage") == 0.0


def test_write_step_records_gate_decision(tmp_path: Path) -> None:
    score = {"backend": "mock", "all_correct": True, "geomean_tflops": 3.0, "cases": [{}]}
    attempt = VariationAttempt(
        decision=decision("avo score --backend torch-sdpa"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        score_payload=score,
    )
    step = finalize_attempt(tmp_path / "lineage", attempt)
    path = tmp_path / "step.json"

    write_step(path, step)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["gate_decision"]["accepted"] is True
    assert payload["patch_cleanup_result"] is None


def test_write_step_record_uses_timestamped_file(tmp_path: Path) -> None:
    attempt = VariationAttempt(
        decision=decision("avo env"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "env"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    step = EvolutionStep(attempt=attempt, gate_decision=None)

    first = write_step_record(tmp_path / "attempts", step)
    second = write_step_record(tmp_path / "attempts", step)

    assert first.exists()
    assert second.exists()
    assert first != second
    assert first.name.startswith("2026-05-08T00-00-01-00-00")


def test_summarize_attempt_history_reports_recent_steps(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    accepted_score = {"backend": "mock", "all_correct": True, "geomean_tflops": 3.0, "cases": [{}]}
    accepted_attempt = VariationAttempt(
        decision=decision("avo score --backend torch-sdpa"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        score_payload=accepted_score,
    )
    rejected_attempt = VariationAttempt(
        decision=decision("avo score --backend candidate"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=2,
            timed_out=False,
            stdout_tail="",
            stderr_tail="failed",
        ),
        started_at="2026-05-08T00:00:02+00:00",
        completed_at="2026-05-08T00:00:03+00:00",
    )
    patch_cleanup = PatchResult(
        ok=True,
        patch_paths=["candidates/seed.py"],
        returncode=0,
        stdout_tail="",
        stderr_tail="",
    )

    write_step_record(attempts, finalize_attempt(tmp_path / "lineage", accepted_attempt))
    write_step_record(
        attempts,
        EvolutionStep(
            attempt=rejected_attempt,
            gate_decision=None,
            patch_cleanup_result=patch_cleanup,
        ),
    )
    (attempts / "bad.json").write_text("{not-json", encoding="utf-8")

    summary = summarize_attempt_history(attempts, limit=5)

    assert "Recent attempts" in summary
    assert "avo score --backend torch-sdpa" in summary
    assert "gate accepted=True" in summary
    assert "geomean_tflops=3.0" in summary
    assert "command returncode=2" in summary
    assert "patch cleanup ok" in summary
    assert "bad.json" not in summary


def test_summarize_attempt_history_marks_reverted_acceptance_stale(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    accepted_score = {
        "backend": "mock",
        "all_correct": True,
        "geomean_tflops": 9.54,
        "cases": [{}],
    }
    accepted_attempt = VariationAttempt(
        decision=decision("avo score --backend candidate"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        score_payload=accepted_score,
    )
    write_step_record(attempts, finalize_attempt(tmp_path / "lineage", accepted_attempt))

    summary = summarize_attempt_history(attempts, limit=5, current_best_geomean=9.50)

    assert "class=stale_accepted" in summary
    assert "lineage status=stale accepted above current best 9.5" in summary
    assert "Lineage correction" in summary
    assert "reverted or noisy historical acceptances" in summary


def test_summarize_attempt_history_ignores_loop_and_score_json(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    attempt = VariationAttempt(
        decision=decision("avo compile --source candidates/seed.cu --out-dir build/seed"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))
    (attempts / "zz_loop_after_probe.json").write_text(
        json.dumps({"accepted": False, "steps": []}),
        encoding="utf-8",
    )
    (attempts / "zz_manual_score.json").write_text(
        json.dumps({"ok": True, "payload": {"geomean_tflops": 1.0}}),
        encoding="utf-8",
    )

    summary = summarize_attempt_history(attempts, limit=1)

    assert "avo compile --source candidates/seed.cu" in summary
    assert "zz_loop_after_probe.json" not in summary
    assert "zz_manual_score.json" not in summary
    assert "<missing command>" not in summary


def test_summarize_attempt_history_flags_repeated_unaccepted_attempts(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    for index in range(3):
        attempt = VariationAttempt(
            decision=decision("avo score --backend candidate"),
            command_result=CommandResult(
                command=[sys.executable, "-m", "avo", "score"],
                returncode=2,
                timed_out=False,
                stdout_tail="",
                stderr_tail="failed",
            ),
            started_at=f"2026-05-08T00:00:0{index}+00:00",
            completed_at=f"2026-05-08T00:00:0{index + 1}+00:00",
        )
        write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "Supervisor signal" in summary
    assert "share command/edit fingerprint" in summary
    assert "materially different optimization direction" in summary


def test_summarize_attempt_history_includes_patch_failure_detail(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    patch_result = PatchResult(
        ok=False,
        patch_paths=["candidates/seed.py"],
        returncode=1,
        stdout_tail="",
        stderr_tail="error: corrupt patch at line 53\n",
        rejected_reason="git apply --check failed",
    )
    attempt = VariationAttempt(
        decision=decision("avo compile --source candidates/seed.cu --out-dir build/seed"),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=None,
            timed_out=False,
            stdout_tail="",
            stderr_tail="candidate patch rejected: git apply --check failed",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        patch_result=patch_result,
    )
    write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "patch rejected reason=git apply --check failed" in summary
    assert "error: corrupt patch at line 53" in summary
    assert "class=raw_diff_preflight" in summary


def test_summarize_attempt_history_classifies_nonfinite_score(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    attempt = VariationAttempt(
        decision=decision(
            "avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py"
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        score_payload={
            "all_correct": False,
            "geomean_tflops": 0.0,
            "cases": [
                {
                    "correct": False,
                    "error": "RuntimeError: candidate output contains non-finite values",
                }
            ],
        },
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "class=correctness_nonfinite_output" in summary
    assert "non-finite values" in summary


def test_summarize_attempt_history_classifies_score_environment_error(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    attempt = VariationAttempt(
        decision=decision(
            "avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py"
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        score_payload={
            "all_correct": False,
            "geomean_tflops": 0.0,
            "cases": [
                {
                    "correct": False,
                    "error": "RuntimeError: Ninja is required to load C++ extensions",
                }
            ],
        },
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "class=score_environment_error" in summary
    assert "Ninja is required" in summary


def test_score_environment_error_is_not_repairable_correctness_failure() -> None:
    attempt = VariationAttempt(
        decision=decision(
            "avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py",
            candidate_transform={
                "op": "set_constexpr_int",
                "path": "candidates/kernel.cu",
                "name": "kThreads",
                "value": 64,
            },
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        score_payload={
            "all_correct": False,
            "geomean_tflops": 0.0,
            "cases": [
                {
                    "correct": False,
                    "error": "RuntimeError: CUDA is not available",
                }
            ],
        },
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )

    assert not attempt_has_repairable_correctness_failure(attempt)


def test_summarize_attempt_history_classifies_profile_unavailable(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    attempt = VariationAttempt(
        decision=decision(
            "avo profile --backend candidate --candidate candidates/cuda_mma_attention_seed.py"
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "profile"],
            returncode=2,
            timed_out=False,
            stdout_tail=json.dumps(
                {
                    "ok": False,
                    "profiler": {
                        "profiled": False,
                        "error": "profiler_unsupported_runtime",
                    },
                }
            ),
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "class=profiler_unsupported_runtime" in summary
    assert "avo profile" in summary


def test_score_time_nvcc_failure_is_compile_repairable(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    attempt = VariationAttempt(
        decision=decision(
            "avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py",
            candidate_patch=dedent(
                """\
                diff --git a/candidates/seed.py b/candidates/seed.py
                --- a/candidates/seed.py
                +++ b/candidates/seed.py
                @@ -1 +1 @@
                -VALUE = 1
                +VALUE = bad
                """
            ),
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=1,
            timed_out=False,
            stdout_tail="",
            stderr_tail="ninja: build stopped: nvcc failed compiling attention_kernel.cu",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/seed.py"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
        ),
    )

    assert attempt_has_repairable_compile_failure(attempt)
    write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "class=compile_failed" in summary


def test_score_payload_extension_build_failure_is_compile_class(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    attempt = VariationAttempt(
        decision=decision(
            "avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py",
            candidate_patch=dedent(
                """\
                diff --git a/candidates/seed.py b/candidates/seed.py
                --- a/candidates/seed.py
                +++ b/candidates/seed.py
                @@ -1 +1 @@
                -VALUE = 1
                +VALUE = bad
                """
            ),
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="{}",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
        score_payload={
            "backend": "candidate",
            "all_correct": False,
            "geomean_tflops": 0.0,
            "candidate_source_files": [
                "candidates/dynamic_extension/attention.cpp",
                "candidates/dynamic_extension/attention_kernel.cu",
            ],
            "cases": [
                {
                    "correct": False,
                    "error": (
                        "RuntimeError: Error building extension 'runtime_demo': "
                        "ninja: build stopped: nvcc failed compiling attention_kernel.cu"
                    ),
                }
            ],
        },
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/seed.py"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
        ),
    )

    assert attempt_has_repairable_correctness_failure(attempt)
    assert correctness_failure_class_for_attempt(attempt) == "score_time_compile_failure"
    write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "class=score_time_compile_failure" in summary


def test_summarize_attempt_history_classifies_planning_validation_failure(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    step = planning_failure_step(
        ValueError(
            "agent returned invalid variation decision after 3 attempts: "
            "next_command repeats a recorded no-patch compile diagnostic; include "
            "candidate_transform/candidate_patch to build-check a change"
        )
    )
    write_step_record(attempts, step)

    summary = summarize_attempt_history(attempts, limit=5)

    assert "class=planning_no_patch_compile" in summary


def test_summarize_attempt_history_classifies_missing_pending_transform(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    step = planning_failure_step(
        ValueError(
            "next_command scores without the pending compile-only candidate_transform; "
            "include the exact candidate_transform JSON from the follow-up signal"
        )
    )
    write_step_record(attempts, step)

    summary = summarize_attempt_history(attempts, limit=5)

    assert "class=planning_missing_pending_transform" in summary


def test_summarize_attempt_history_classifies_support_only_transform(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    step = planning_failure_step(
        ValueError(
            "candidate_transform is support-only; make the smallest coherent semantic "
            "transformation that preserves invariants"
        )
    )
    write_step_record(attempts, step)

    summary = summarize_attempt_history(attempts, limit=5)

    assert "class=planning_support_only_transform" in summary


def test_summarize_attempt_history_classifies_transform_semantic_mismatch(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    step = planning_failure_step(
        ValueError(
            "candidate_transform semantic mismatch: contract-only transforms do not "
            "implement dataflow"
        )
    )
    write_step_record(attempts, step)

    summary = summarize_attempt_history(attempts, limit=5)

    assert "class=planning_transform_semantic_mismatch" in summary
    assert "planning_feedback=ValueError: candidate_transform semantic mismatch" in summary


def test_summarize_attempt_history_classifies_predicted_correctness_planning_failure(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    step = planning_failure_step(
        ValueError(
            "candidate_patch is described as known invalid by the decision itself; "
            "planning risk class 'predicted_correctness_failure' matched "
            "'out of bounds key accesses will produce incorrect results'"
        )
    )
    write_step_record(attempts, step)

    summary = summarize_attempt_history(attempts, limit=5)

    assert "class=planning_predicted_correctness_failure" in summary
    assert "planning_feedback=ValueError: candidate_patch is described as known invalid" in summary


def test_summarize_attempt_history_classifies_provider_failure_without_supervisor_signal(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    for _ in range(3):
        step = planning_failure_step(
            RuntimeError("provider unavailable: Anthropic BadRequestError credit balance")
        )
        write_step_record(attempts, step)

    state = update_promoted_preflight_tracks(attempts)
    summary = summarize_attempt_history(attempts, limit=5)

    assert "class=planner_provider_error" in summary
    assert "Supervisor signal:" not in summary
    assert "planner_provider_error" not in state["tracks"]


def test_planning_feedback_class_is_not_promoted_without_concrete_preflight(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    for _ in range(3):
        step = planning_failure_step(
            ValueError(
                "candidate_patch is described as known invalid by the decision itself; "
                "planning risk class 'predicted_correctness_failure' matched "
                "'incorrect results'"
            )
        )
        write_step_record(attempts, step)

    state = update_promoted_preflight_tracks(attempts)
    summary = summarize_attempt_history(attempts, limit=5)

    assert "planning_predicted_correctness_failure" not in state["tracks"]
    assert "no concrete hard preflight track exists" in summary
    assert "eligible for hard preflight promotion" not in summary


def test_summarize_attempt_history_does_not_fingerprint_different_planning_errors(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    errors = [
        "candidate_patch and candidate_transform are mutually exclusive; use one edit channel",
        "candidate_transform or candidate_patch must be provided when candidate_edit "
        "describes a code change",
        "next_command repeats a recorded no-patch compile diagnostic; include candidate_transform",
    ]
    for error in errors:
        write_step_record(attempts, planning_failure_step(ValueError(error)))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "share command/edit fingerprint" not in summary
    assert "class=planning_edit_channel" in summary
    assert "class=planning_missing_edit_payload" in summary
    assert "class=planning_no_patch_compile" in summary


def test_summarize_attempt_history_requests_score_after_compile_only_transform(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kMaxSeqLen",
        "value": 512,
    }
    attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "compiled successfully but has not been scored" in summary
    assert "score the same candidate_transform" in summary
    assert "Compile-only patches are cleaned up" in summary
    assert "no_edit score would score the unmodified seed" in summary
    assert "executable edit payload must still be the exact candidate_transform" in summary
    assert "do not return a prose-only score decision" in summary
    assert '"op":"set_constexpr_int"' in summary
    assert '"value":512' in summary


def test_summarize_attempt_history_does_not_request_score_for_support_only_transform(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "add_include",
        "path": "candidates/kernel.cu",
        "header": "cuda_pipeline_primitives.h",
    }
    attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "compiled successfully but has not been scored" not in summary
    assert "score the same candidate_transform" not in summary
    assert "class=compile_only_diagnostic" in summary


def test_summarize_attempt_history_keeps_compile_followup_after_planning_failure(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kMaxSeqLen",
        "value": 1024,
    }
    compile_attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=compile_attempt, gate_decision=None))
    write_step_record(attempts, planning_failure_step(ValueError("invalid planner compile")))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "compiled successfully but has not been scored" in summary
    assert "score the same candidate_transform" in summary
    assert '"op":"set_constexpr_int"' in summary
    assert '"value":1024' in summary


def test_summarize_attempt_history_drops_compile_followup_after_preflight_rejection(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "batch",
        "steps": [
            {
                "op": "add_include",
                "path": "candidates/kernel.cu",
                "header": "cooperative_groups.h",
            },
            {
                "op": "replace_once",
                "path": "candidates/kernel.cu",
                "find": "__shared__ float scores[kScoreElements];",
                "replace": (
                    "__shared__ float scores[kScoreElements];\n"
                    "__shared__ __nv_bfloat16 k_shared[2][kTile * kHeadDim];"
                ),
            },
        ],
    }
    compile_attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    rejected_attempt = VariationAttempt(
        decision=decision(
            "avo score --backend candidate --candidate candidates/seed.py --seq-lens 4096",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=None,
            timed_out=False,
            stdout_tail="",
            stderr_tail=(
                "candidate patch rejected: candidate structural preflight rejected: "
                "structural preflight track no_effect_shared_staging_buffer"
            ),
        ),
        patch_result=PatchResult(
            ok=False,
            patch_paths=[],
            returncode=None,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=(
                "candidate transform rejected: candidate structural preflight rejected: "
                "structural preflight track no_effect_shared_staging_buffer"
            ),
        ),
        started_at="2026-05-08T00:00:02+00:00",
        completed_at="2026-05-08T00:00:03+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=compile_attempt, gate_decision=None))
    write_step_record(attempts, EvolutionStep(attempt=rejected_attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "compiled successfully but has not been scored" not in summary
    assert "score the same candidate_transform" not in summary
    validate_decision_against_attempt_history(
        decision("avo score --backend candidate --candidate candidates/seed.py --seq-lens 4096"),
        attempts,
    )


def test_summarize_attempt_history_requests_transform_anchor_repair(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "batch",
        "steps": [
            {
                "op": "insert_after_once",
                "path": "candidates/kernel.cu",
                "anchor": "  __syncthreads();",
                "text": "  staged();",
            },
            {
                "op": "replace_once",
                "path": "candidates/kernel.cu",
                "find": "load_global_q();",
                "replace": "load_shared_q();",
            },
        ],
    }
    rejected_attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=None,
            timed_out=False,
            stdout_tail="",
            stderr_tail=(
                "candidate patch rejected: candidate transform rejected: insert transform "
                "expected exactly one anchor, found 5; matching start lines: "
                "54, 84, 121, 127, 155. Use a larger unique anchor including surrounding code."
            ),
        ),
        patch_result=PatchResult(
            ok=False,
            patch_paths=[],
            returncode=None,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=(
                "candidate transform rejected: insert transform expected exactly one anchor, "
                "found 5; matching start lines: 54, 84, 121, 127, 155. Use a larger unique "
                "anchor including surrounding code."
            ),
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=rejected_attempt, gate_decision=None))
    write_step_record(
        attempts,
        planning_failure_step(
            ValueError(
                "edit_mode transform requires candidate_transform; candidate_transform or "
                "candidate_patch must be provided when candidate_edit describes a code change"
            )
        ),
    )

    summary = summarize_attempt_history(attempts, limit=5)

    assert "failed materialization before compile" in summary
    assert "repair the candidate_transform anchors/matches" in summary
    assert "matching start lines: 54, 84, 121, 127, 155" in summary
    assert "do not restate the CUDA edit in prose" in summary
    assert '"anchor":"  __syncthreads();"' in summary
    assert '"replace":"load_shared_q();"' in summary


def test_summarize_attempt_history_scope_hints_loop_local_insert(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "batch",
        "steps": [
            {
                "op": "insert_after_once",
                "path": "candidates/kernel.cu",
                "anchor": "  __syncthreads();",
                "text": (
                    "  for (int linear = threadIdx.x; linear < kTile; linear += blockDim.x) {\n"
                    "    k_shared[linear] = k[key_start + linear];\n"
                    "  }"
                ),
            },
        ],
    }
    rejected_attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=None,
            timed_out=False,
            stdout_tail="",
            stderr_tail="candidate patch rejected",
        ),
        patch_result=PatchResult(
            ok=False,
            patch_paths=[],
            returncode=None,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=(
                "candidate transform rejected: insert transform expected exactly one anchor, "
                "found 5; matching start lines: 54, 84, 121, 127, 155."
            ),
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=rejected_attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "Scope hint: inserted text references key_start" in summary
    assert "inside the key_start loop" in summary
    assert "loop header or surrounding body context" in summary


def test_summarize_attempt_history_drops_stale_compile_followup_after_later_score(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    old_transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kMaxSeqLen",
        "value": 1024,
    }
    scored_transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kMaxSeqLen",
        "value": 2048,
    }
    compile_attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=old_transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    score_attempt = VariationAttempt(
        decision=decision(
            "avo score --backend candidate --candidate candidates/seed.py --seq-lens 2048",
            candidate_transform=scored_transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        score_payload={"all_correct": False, "geomean_tflops": 0.0, "cases": []},
        started_at="2026-05-08T00:00:02+00:00",
        completed_at="2026-05-08T00:00:03+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=compile_attempt, gate_decision=None))
    write_step_record(attempts, EvolutionStep(attempt=score_attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "compiled successfully but has not been scored" not in summary
    validate_decision_against_attempt_history(
        decision("avo score --backend candidate --candidate candidates/seed.py --seq-lens 4096"),
        attempts,
    )


def test_summarize_attempt_history_drops_compile_followup_after_repair_score(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kMaxSeqLen",
        "value": 1024,
    }
    compile_attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    score_attempt = VariationAttempt(
        decision=decision(
            "avo score --backend candidate --candidate candidates/seed.py --seq-lens 1024",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        score_payload={"all_correct": False, "geomean_tflops": 0.0, "cases": []},
        started_at="2026-05-08T00:00:02+00:00",
        completed_at="2026-05-08T00:00:03+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=compile_attempt, gate_decision=None))
    write_step_record(
        attempts,
        planning_failure_step(
            ValueError("correctness repair decision repeated payload"),
            repair_attempts=(score_attempt,),
        ),
    )

    summary = summarize_attempt_history(attempts, limit=5)

    assert "compiled successfully but has not been scored" not in summary
    assert "score the same candidate_transform" not in summary
    validate_decision_against_attempt_history(
        decision("avo score --backend candidate --candidate candidates/seed.py --seq-lens 4096"),
        attempts,
    )


def test_summarize_attempt_history_includes_repair_attempt_failures(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    repair_attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_transform={
                "op": "insert_after_once",
                "path": "candidates/kernel.cu",
                "anchor": "__syncthreads();",
                "text": "int stage = 0;",
            },
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=None,
            timed_out=False,
            stdout_tail="",
            stderr_tail=(
                "candidate patch rejected: candidate transform rejected: insert transform "
                "expected exactly one anchor, found 5; matching start lines: 54, 90, 127"
            ),
        ),
        patch_result=PatchResult(
            ok=False,
            patch_paths=[],
            returncode=None,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=(
                "candidate transform rejected: insert transform expected exactly one "
                "anchor, found 5; matching start lines: 54, 90, 127"
            ),
        ),
        started_at="2026-05-08T00:00:02+00:00",
        completed_at="2026-05-08T00:00:03+00:00",
    )
    write_step_record(
        attempts,
        planning_failure_step(
            ValueError("repair planner failed"),
            repair_attempts=(repair_attempt,),
        ),
    )

    summary = summarize_attempt_history(attempts, limit=5)

    assert "repairs=1" in summary
    assert "repair_details=" in summary
    assert "class=structured_transform_preflight" in summary
    assert "expected exactly one anchor" in summary
    assert "matching start lines: 54, 90, 127" in summary


def test_attempt_history_rejects_repeated_compile_only_transform(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kMaxSeqLen",
        "value": 512,
    }
    attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    with pytest.raises(ValueError, match="repeats a successful compile-only"):
        validate_decision_against_attempt_history(
            decision(
                "avo compile --source candidates/kernel.cu --out-dir build/kernel",
                candidate_transform=transform,
            ),
            attempts,
        )

    validate_decision_against_attempt_history(
        decision(
            "avo score --backend candidate --candidate candidates/seed.py --seq-lens 512",
            candidate_transform=transform,
        ),
        attempts,
    )


def test_pending_compile_transform_uses_payload_timestamps_not_filename_order(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kMaxSeqLen",
        "value": 512,
    }
    old_score_attempt = VariationAttempt(
        decision=decision(
            "avo score --backend candidate --candidate candidates/seed.py --seq-lens 512",
            candidate_transform={
                "op": "set_constexpr_int",
                "path": "candidates/kernel.cu",
                "name": "kThreads",
                "value": 80,
            },
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        score_payload={
            "all_correct": True,
            "geomean_tflops": 9.0,
            "cases": [],
        },
        started_at="2026-05-07T00:00:00+00:00",
        completed_at="2026-05-07T00:00:01+00:00",
    )
    attempts.mkdir()
    (attempts / "zz_old_score.json").write_text(
        json.dumps(
            EvolutionStep(
                attempt=old_score_attempt,
                gate_decision=GateDecision(
                    accepted=False,
                    reason="candidate regressed geomean throughput",
                    candidate_geomean=9.0,
                    best_geomean=9.5,
                ),
            ).as_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    compile_attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=compile_attempt, gate_decision=None))

    assert pending_compile_only_transform(attempts) == transform


def test_pending_compile_transform_survives_older_repair_score_in_same_step(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "replace_once",
        "path": "candidates/kernel.cu",
        "find": "wmma::load_matrix_sync(v_frag, old_ptr, stride);",
        "replace": "wmma::load_matrix_sync(v_frag, new_ptr, stride);",
    }
    old_repair_score = VariationAttempt(
        decision=decision(
            "avo score --backend candidate --candidate candidates/seed.py --seq-lens 512",
            candidate_transform={
                "op": "replace_once",
                "path": "candidates/kernel.cu",
                "find": "__syncthreads();",
                "replace": "__syncwarp();",
            },
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        score_payload={
            "all_correct": False,
            "geomean_tflops": 0.0,
            "cases": [{"correct": False}],
        },
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    compile_attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:01:00+00:00",
        completed_at="2026-05-08T00:01:01+00:00",
    )
    write_step_record(
        attempts,
        EvolutionStep(
            attempt=compile_attempt,
            gate_decision=None,
            repair_attempts=(old_repair_score,),
        ),
    )

    assert pending_compile_only_transform(attempts) == transform

    validate_decision_against_attempt_history(
        decision(
            "avo score --backend candidate --candidate candidates/seed.py --seq-lens 512",
            candidate_transform=transform,
        ),
        attempts,
    )


def test_attempt_history_rejects_repeated_scored_unaccepted_transform(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "replace_once",
        "path": "candidates/kernel.cu",
        "find": "constexpr int kThreads = 96;",
        "replace": "constexpr int kThreads = 80;",
    }
    attempt = VariationAttempt(
        decision=decision(
            "avo score --backend candidate --candidate candidates/seed.py --seq-lens 4096",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        score_payload={
            "all_correct": True,
            "geomean_tflops": 9.10,
            "cases": [],
        },
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(
        attempts,
        EvolutionStep(
            attempt=attempt,
            gate_decision=GateDecision(
                accepted=False,
                reason="candidate regressed geomean throughput",
                candidate_geomean=9.10,
                best_geomean=9.50,
            ),
        ),
    )

    with pytest.raises(ValueError, match="previously scored unaccepted transform"):
        validate_decision_against_attempt_history(
            decision(
                "avo score --backend candidate --candidate candidates/seed.py --seq-lens 4096",
                candidate_transform=transform,
            ),
            attempts,
        )


def test_attempt_history_allows_repeated_transform_after_score_environment_error(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "replace_once",
        "path": "candidates/kernel.cu",
        "find": "constexpr int kThreads = 96;",
        "replace": "constexpr int kThreads = 80;",
    }
    attempt = VariationAttempt(
        decision=decision(
            "avo score --backend candidate --candidate candidates/seed.py --seq-lens 4096",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        score_payload={
            "all_correct": False,
            "geomean_tflops": 0.0,
            "cases": [{"error": "RuntimeError: CUDA is not available"}],
        },
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    validate_decision_against_attempt_history(
        decision(
            "avo score --backend candidate --candidate candidates/seed.py --seq-lens 4096",
            candidate_transform=transform,
        ),
        attempts,
    )


def test_attempt_history_rejects_new_compile_while_transform_score_pending(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    pending_transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kMaxSeqLen",
        "value": 512,
    }
    next_transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kThreads",
        "value": 80,
    }
    attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=pending_transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    with pytest.raises(ValueError, match="must be scored before compiling"):
        validate_decision_against_attempt_history(
            decision(
                "avo compile --source candidates/kernel.cu --out-dir build/kernel2",
                candidate_transform=next_transform,
            ),
            attempts,
        )


def test_attempt_history_rejects_different_score_while_transform_score_pending(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    pending_transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kMaxSeqLen",
        "value": 512,
    }
    other_transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kThreads",
        "value": 80,
    }
    attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=pending_transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    with pytest.raises(ValueError, match="must be scored before compiling or scoring"):
        validate_decision_against_attempt_history(
            decision(
                "avo score --backend candidate --candidate candidates/seed.py --seq-lens 512",
                candidate_transform=other_transform,
            ),
            attempts,
        )


def test_attempt_history_rejects_score_without_pending_transform(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kMaxSeqLen",
        "value": 512,
    }
    attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    with pytest.raises(ValueError, match="pending compile-only candidate_transform"):
        validate_decision_against_attempt_history(
            decision("avo score --backend candidate --candidate candidates/seed.py --seq-lens 512"),
            attempts,
        )


def test_cleanup_failed_compile_transform_is_not_pending(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kMaxSeqLen",
        "value": 512,
    }
    attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    cleanup_failed = PatchResult(
        ok=False,
        patch_paths=["candidates/kernel.cu"],
        returncode=0,
        stdout_tail="",
        stderr_tail="candidate patch cleanup left paths dirty",
        rejected_reason="candidate patch cleanup left paths dirty",
    )
    write_step_record(
        attempts,
        EvolutionStep(attempt=attempt, gate_decision=None, patch_cleanup_result=cleanup_failed),
    )

    validate_decision_against_attempt_history(
        decision("avo score --backend candidate --candidate candidates/seed.py --seq-lens 512"),
        attempts,
    )


def test_attempt_history_rejects_repeated_compile_after_planning_failure(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kMaxSeqLen",
        "value": 1024,
    }
    compile_attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=compile_attempt, gate_decision=None))
    write_step_record(attempts, planning_failure_step(ValueError("invalid planner compile")))

    with pytest.raises(ValueError, match="repeats a successful compile-only"):
        validate_decision_against_attempt_history(
            decision(
                "avo compile --source candidates/kernel.cu --out-dir build/kernel",
                candidate_transform=transform,
            ),
            attempts,
        )


def test_attempt_history_rejects_repeated_compile_after_transform_score(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "set_constexpr_int",
        "path": "candidates/kernel.cu",
        "name": "kMaxSeqLen",
        "value": 1024,
    }
    compile_attempt = VariationAttempt(
        decision=decision(
            "avo compile --source candidates/kernel.cu --out-dir build/kernel",
            candidate_patch="diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "compile"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
            rejected_reason=None,
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    score_attempt = VariationAttempt(
        decision=decision(
            "avo score --backend candidate --candidate candidates/seed.py --seq-lens 1024",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        score_payload={"all_correct": False, "geomean_tflops": 0.0, "cases": []},
        started_at="2026-05-08T00:00:02+00:00",
        completed_at="2026-05-08T00:00:03+00:00",
    )
    write_step_record(attempts, EvolutionStep(attempt=compile_attempt, gate_decision=None))
    write_step_record(attempts, EvolutionStep(attempt=score_attempt, gate_decision=None))

    with pytest.raises(ValueError, match="repeats a successful compile-only"):
        validate_decision_against_attempt_history(
            decision(
                "avo compile --source candidates/kernel.cu --out-dir build/kernel_again",
                candidate_transform=transform,
            ),
            attempts,
        )


def test_summarize_attempt_history_promotes_recurring_failure_class(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    for index in range(3):
        attempt = VariationAttempt(
            decision=decision(
                f"avo compile --source candidates/kernel_{index}.cu --out-dir build/kernel_{index}"
            ),
            command_result=CommandResult(
                command=[sys.executable, "-m", "avo", "compile"],
                returncode=2,
                timed_out=False,
                stdout_tail="",
                stderr_tail="error: identifier old_scale is undefined\n",
            ),
            started_at=f"2026-05-08T00:00:0{index}+00:00",
            completed_at=f"2026-05-08T00:00:0{index + 1}+00:00",
        )
        write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "share failure class 'stale_or_undefined_symbol'" in summary
    assert "eligible for hard preflight promotion" in summary


def test_update_promoted_preflight_tracks_persists_recurring_class(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    for index in range(3):
        attempt = VariationAttempt(
            decision=decision(
                f"avo compile --source candidates/kernel_{index}.cu --out-dir build/kernel_{index}"
            ),
            command_result=CommandResult(
                command=[sys.executable, "-m", "avo", "compile"],
                returncode=2,
                timed_out=False,
                stdout_tail="",
                stderr_tail="error: identifier old_scale is undefined\n",
            ),
            started_at=f"2026-05-08T00:00:0{index}+00:00",
            completed_at=f"2026-05-08T00:00:0{index + 1}+00:00",
        )
        write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    state = update_promoted_preflight_tracks(attempts)
    summary = summarize_attempt_history(attempts, limit=5)

    assert "stale_or_undefined_symbol" in state["tracks"]
    assert state["tracks"]["stale_or_undefined_symbol"]["track_names"] == [
        "promoted_symbol_lifecycle_duplicate_declaration",
        "promoted_symbol_lifecycle_removed_declaration",
    ]
    assert load_promoted_preflight_classes(attempts) == frozenset({"stale_or_undefined_symbol"})
    assert "Active hard preflight tracks:" in summary
    assert "track=symbol_lifecycle" in summary
    assert "checks=promoted_symbol_lifecycle_duplicate_declaration" in summary


def test_summarize_attempt_history_counts_mixed_recurring_failure_classes(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    errors = [
        "error: identifier old_scale is undefined\n",
        "error: expected a declaration\n",
        "error: identifier row_sum is undefined\n",
        "error: expected a ';'\n",
        "error: identifier probability_frag is undefined\n",
        "error: expected a type specifier\n",
    ]
    for index, stderr in enumerate(errors):
        attempt = VariationAttempt(
            decision=decision(
                f"avo compile --source candidates/kernel_{index}.cu --out-dir build/kernel_{index}"
            ),
            command_result=CommandResult(
                command=[sys.executable, "-m", "avo", "compile"],
                returncode=2,
                timed_out=False,
                stdout_tail="",
                stderr_tail=stderr,
            ),
            started_at=f"2026-05-08T00:00:{index:02d}+00:00",
            completed_at=f"2026-05-08T00:00:{index + 1:02d}+00:00",
        )
        write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=6)

    assert "recurring failure classes" in summary
    assert "cuda_syntax_error(count=3)" in summary
    assert "stale_or_undefined_symbol(count=3)" in summary


def test_summarize_attempt_history_flags_recurring_transform_family(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    for index, operand in enumerate(("Q", "K", "V")):
        transform = {
            "op": "batch",
            "steps": [
                {
                    "op": "insert_after_once",
                    "path": "candidates/kernel.cu",
                    "anchor": "__shared__ float old_scale[kTile];",
                    "text": f"__shared__ __nv_bfloat16 {operand.lower()}_tile[kTile * kHeadDim];",
                },
                {
                    "op": "replace_once",
                    "path": "candidates/kernel.cu",
                    "find": f"load {operand} from global",
                    "replace": f"stage {operand} tile into shared memory",
                },
            ],
        }
        attempt = VariationAttempt(
            decision=VariationDecision(
                hypothesis=f"stage {operand} through shared memory",
                files_to_inspect=["candidates/kernel.cu"],
                candidate_edit=f"Add cooperative shared-memory staging for {operand} tiles.",
                expected_effect="reduce global memory traffic",
                risk="barrier overhead may dominate",
                next_command="avo score --backend candidate --candidate candidates/seed.py",
                edit_mode="transform",
                candidate_transform=transform,
            ),
            command_result=CommandResult(
                command=[sys.executable, "-m", "avo", "score"],
                returncode=0,
                timed_out=False,
                stdout_tail="",
                stderr_tail="",
            ),
            score_payload={"all_correct": True, "geomean_tflops": 1.0, "cases": []},
            patch_result=PatchResult(
                ok=True,
                patch_paths=["candidates/kernel.cu"],
                returncode=0,
                stdout_tail="",
                stderr_tail="",
            ),
            started_at=f"2026-05-08T00:00:{index:02d}+00:00",
            completed_at=f"2026-05-08T00:00:{index + 1:02d}+00:00",
        )
        write_step_record(
            attempts,
            EvolutionStep(
                attempt=attempt,
                gate_decision=GateDecision(
                    accepted=False,
                    reason="candidate regressed geomean throughput",
                    candidate_geomean=1.0,
                    best_geomean=2.0,
                ),
            ),
        )

    summary = summarize_attempt_history(attempts, limit=5)

    assert "family=shared_memory_staging" in summary
    assert "Semantic-family signal" in summary
    assert "shared_memory_staging(count=3)" in summary
    assert "Choose a materially different optimization family" in summary


def test_summarize_attempt_history_flags_thread_count_family(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    for index, value in enumerate((32, 128, 256)):
        transform = {
            "op": "set_constexpr_int",
            "path": "candidates/kernel.cu",
            "name": "kThreads",
            "value": value,
        }
        attempt = VariationAttempt(
            decision=VariationDecision(
                hypothesis="retune block thread count",
                files_to_inspect=["candidates/kernel.cu"],
                candidate_edit=f"Set kThreads to {value} and score the retune.",
                expected_effect="may improve occupancy",
                risk="may reduce useful parallelism",
                next_command="avo score --backend candidate --candidate candidates/seed.py",
                edit_mode="transform",
                candidate_transform=transform,
            ),
            command_result=CommandResult(
                command=[sys.executable, "-m", "avo", "score"],
                returncode=0,
                timed_out=False,
                stdout_tail="",
                stderr_tail="",
            ),
            score_payload={"all_correct": True, "geomean_tflops": 1.0, "cases": []},
            materialized_patch=(
                "diff --git a/candidates/kernel.cu b/candidates/kernel.cu\n"
                "@@ -1,4 +1,4 @@\n"
                " __shared__ float scores[kTile * kTile];\n"
                f"-constexpr int kThreads = {value - 1};\n"
                f"+constexpr int kThreads = {value};\n"
            ),
            patch_result=PatchResult(
                ok=True,
                patch_paths=["candidates/kernel.cu"],
                returncode=0,
                stdout_tail="",
                stderr_tail="",
            ),
            started_at=f"2026-05-08T00:00:{index:02d}+00:00",
            completed_at=f"2026-05-08T00:00:{index + 1:02d}+00:00",
        )
        write_step_record(
            attempts,
            EvolutionStep(
                attempt=attempt,
                gate_decision=GateDecision(
                    accepted=False,
                    reason="candidate regressed geomean throughput",
                    candidate_geomean=1.0,
                    best_geomean=2.0,
                ),
            ),
        )

    summary = summarize_attempt_history(attempts, limit=5)

    assert "family=thread_count_or_warp_mapping" in summary
    assert "thread_count_or_warp_mapping(count=3)" in summary
    assert "Choose a materially different optimization family" in summary


def test_summarize_attempt_history_keeps_blockdim_shared_staging_family(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "batch",
        "steps": [
            {
                "op": "insert_after_once",
                "path": "candidates/kernel.cu",
                "anchor": "for (int key_start = 0;",
                "text": (
                    "for (int linear = threadIdx.x; linear < kTile * kHeadDim; "
                    "linear += blockDim.x) { k_shared[linear] = k[linear]; }\n"
                    "__syncthreads();"
                ),
            },
            {
                "op": "replace_once",
                "path": "candidates/kernel.cu",
                "find": "load K from global",
                "replace": "load K from shared staging buffer",
            },
        ],
    }
    attempt = VariationAttempt(
        decision=VariationDecision(
            hypothesis="current accepted state uses kThreads=96",
            files_to_inspect=["candidates/kernel.cu"],
            candidate_edit=(
                "Use all 96 threads for cooperative shared-memory K tile staging."
            ),
            expected_effect="reduce global memory traffic",
            risk="barrier overhead may dominate",
            next_command="avo score --backend candidate --candidate candidates/seed.py",
            edit_mode="transform",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        score_payload={"all_correct": True, "geomean_tflops": 1.0, "cases": []},
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(
        attempts,
        EvolutionStep(
            attempt=attempt,
            gate_decision=GateDecision(
                accepted=False,
                reason="candidate regressed geomean throughput",
                candidate_geomean=1.0,
                best_geomean=2.0,
            ),
        ),
    )

    summary = summarize_attempt_history(attempts, limit=2)

    assert "family=shared_memory_staging" in summary
    assert "family=thread_count_or_warp_mapping" not in summary


def test_summarize_attempt_history_classifies_syncwarp_family(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    transform = {
        "op": "replace_once",
        "path": "candidates/kernel.cu",
        "find": "wmma::store_matrix_sync(scores, score_frag, kTile);\n__syncthreads();",
        "replace": (
            "wmma::store_matrix_sync(scores, score_frag, kTile);\n"
            "__syncwarp();\n"
            "__syncthreads();"
        ),
    }
    attempt = VariationAttempt(
        decision=VariationDecision(
            hypothesis="score-store synchronization may affect scheduler behavior",
            files_to_inspect=["candidates/kernel.cu"],
            candidate_edit="Add __syncwarp before the score-store block barrier.",
            expected_effect="may improve warp convergence before block synchronization",
            risk="extra synchronization may regress",
            next_command="avo score --backend candidate --candidate candidates/seed.py",
            edit_mode="transform",
            candidate_transform=transform,
        ),
        command_result=CommandResult(
            command=[sys.executable, "-m", "avo", "score"],
            returncode=0,
            timed_out=False,
            stdout_tail="",
            stderr_tail="",
        ),
        score_payload={"all_correct": True, "geomean_tflops": 1.0, "cases": []},
        patch_result=PatchResult(
            ok=True,
            patch_paths=["candidates/kernel.cu"],
            returncode=0,
            stdout_tail="",
            stderr_tail="",
        ),
        started_at="2026-05-08T00:00:00+00:00",
        completed_at="2026-05-08T00:00:01+00:00",
    )
    write_step_record(
        attempts,
        EvolutionStep(
            attempt=attempt,
            gate_decision=GateDecision(
                accepted=False,
                reason="candidate regressed geomean throughput",
                candidate_geomean=1.0,
                best_geomean=2.0,
            ),
        ),
    )

    summary = summarize_attempt_history(attempts, limit=2)

    assert "family=synchronization_or_barrier" in summary
    assert "family=thread_count_or_warp_mapping" not in summary


def test_summarize_attempt_history_flags_query_tile_work_mapping_family(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    for index, value in enumerate((2, 3, 4)):
        transform = {
            "op": "batch",
            "steps": [
                {
                    "op": "insert_after_once",
                    "path": "candidates/kernel.cu",
                    "anchor": "constexpr int kTile = 16;",
                    "text": f"constexpr int kQueryTilesPerBlock = {value};",
                },
                {
                    "op": "replace_once",
                    "path": "candidates/kernel.cu",
                    "find": "one query tile per block",
                    "replace": "query tiles per block loop",
                },
            ],
        }
        attempt = VariationAttempt(
            decision=VariationDecision(
                hypothesis="process multiple query tiles per block",
                files_to_inspect=["candidates/kernel.cu"],
                candidate_edit=(
                    f"Set kQueryTilesPerBlock={value} and wrap query tiles per block."
                ),
                expected_effect="may amortize K/V work",
                risk="may serialize useful parallelism",
                next_command="avo score --backend candidate --candidate candidates/seed.py",
                edit_mode="transform",
                candidate_transform=transform,
            ),
            command_result=CommandResult(
                command=[sys.executable, "-m", "avo", "score"],
                returncode=0,
                timed_out=False,
                stdout_tail="",
                stderr_tail="",
            ),
            score_payload={"all_correct": True, "geomean_tflops": 1.0, "cases": []},
            patch_result=PatchResult(
                ok=True,
                patch_paths=["candidates/kernel.cu"],
                returncode=0,
                stdout_tail="",
                stderr_tail="",
            ),
            started_at=f"2026-05-08T00:00:{index:02d}+00:00",
            completed_at=f"2026-05-08T00:00:{index + 1:02d}+00:00",
        )
        write_step_record(
            attempts,
            EvolutionStep(
                attempt=attempt,
                gate_decision=GateDecision(
                    accepted=False,
                    reason="candidate regressed geomean throughput",
                    candidate_geomean=1.0,
                    best_geomean=2.0,
                ),
            ),
        )

    summary = summarize_attempt_history(attempts, limit=5)

    assert "family=query_tile_work_mapping" in summary
    assert "query_tile_work_mapping(count=3)" in summary
    assert "Choose a materially different optimization family" in summary


def test_update_promoted_preflight_tracks_persists_mixed_recurring_classes(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    errors = [
        "error: identifier old_scale is undefined\n",
        "error: expected a declaration\n",
        "error: identifier row_sum is undefined\n",
        "error: expected a ';'\n",
        "error: identifier probability_frag is undefined\n",
        "error: expected a type specifier\n",
    ]
    for index, stderr in enumerate(errors):
        attempt = VariationAttempt(
            decision=decision(
                f"avo compile --source candidates/kernel_{index}.cu --out-dir build/kernel_{index}"
            ),
            command_result=CommandResult(
                command=[sys.executable, "-m", "avo", "compile"],
                returncode=2,
                timed_out=False,
                stdout_tail="",
                stderr_tail=stderr,
            ),
            started_at=f"2026-05-08T00:00:{index:02d}+00:00",
            completed_at=f"2026-05-08T00:00:{index + 1:02d}+00:00",
        )
        write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    state = update_promoted_preflight_tracks(attempts)

    assert state["tracks"]["cuda_syntax_error"]["recent_count"] == 3
    assert state["tracks"]["cuda_syntax_error"]["track_names"] == [
        "promoted_cuda_delimiter_balance"
    ]
    assert state["tracks"]["stale_or_undefined_symbol"]["recent_count"] == 3
    assert load_promoted_preflight_classes(attempts) == frozenset(
        {"cuda_syntax_error", "stale_or_undefined_symbol"}
    )


def test_summarize_attempt_history_normalizes_compile_out_dir(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    source = "candidates/cuda_tiled_attention/attention_kernel.cu"
    for index in range(3):
        attempt = VariationAttempt(
            decision=decision(f"avo compile --source {source} --out-dir build/tiled_{index}"),
            command_result=CommandResult(
                command=[sys.executable, "-m", "avo", "compile"],
                returncode=0,
                timed_out=False,
                stdout_tail="",
                stderr_tail="",
            ),
            started_at=f"2026-05-08T00:00:0{index}+00:00",
            completed_at=f"2026-05-08T00:00:0{index + 1}+00:00",
        )
        write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "Supervisor signal" in summary
    assert "share command/edit fingerprint" in summary


def test_summarize_attempt_history_flags_unaccepted_exhaustion(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    for index in range(5):
        attempt = VariationAttempt(
            decision=decision(
                f"avo score --backend candidate --candidate candidates/seed_{index}.py"
            ),
            command_result=CommandResult(
                command=[sys.executable, "-m", "avo", "score"],
                returncode=2,
                timed_out=False,
                stdout_tail="",
                stderr_tail="failed",
            ),
            started_at=f"2026-05-08T00:00:0{index}+00:00",
            completed_at=f"2026-05-08T00:00:0{index + 1}+00:00",
        )
        write_step_record(attempts, EvolutionStep(attempt=attempt, gate_decision=None))

    summary = summarize_attempt_history(attempts, limit=5)

    assert "Supervisor signal" in summary
    assert "last 5 attempts produced no accepted candidate" in summary
    assert "reset strategy" in summary


def write_seed_candidate(root: Path) -> Path:
    candidate_dir = root / "candidates"
    candidate_dir.mkdir()
    seed = candidate_dir / "seed.py"
    seed.write_text("VALUE = 1\n", encoding="utf-8")
    return seed


def init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "candidates/seed.py"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AVO Test",
            "-c",
            "user.email=avo-test@example.com",
            "commit",
            "-m",
            "seed candidate",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


def candidate_value_patch() -> str:
    return dedent(
        """\
        diff --git a/candidates/seed.py b/candidates/seed.py
        --- a/candidates/seed.py
        +++ b/candidates/seed.py
        @@ -1 +1 @@
        -VALUE = 1
        +VALUE = 2
        """
    )


def candidate_value_patch_with_context() -> str:
    return dedent(
        """\
        diff --git a/candidates/seed.py b/candidates/seed.py
        --- a/candidates/seed.py
        +++ b/candidates/seed.py
        @@ -1,2 +1,2 @@
        -VALUE = 1
        +VALUE = 2
         OTHER = 1
        """
    )


def assert_patch_rejected(result: PatchResult, reason: str) -> None:
    assert not result.ok
    assert result.rejected_reason is not None
    assert reason in result.rejected_reason
