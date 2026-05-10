import re
from pathlib import Path

import pytest

from avo.cli import _knowledge_query
from avo.knowledge import build_knowledge_context, search_knowledge


def test_search_knowledge_ranks_relevant_chunks(tmp_path: Path) -> None:
    corpus = tmp_path / "knowledge"
    corpus.mkdir()
    (corpus / "ampere.md").write_text(
        "# Ampere\n"
        "cp.async copies should use 16 byte aligned groups for shared memory staging.\n\n"
        "# Other\n"
        "This paragraph is about unrelated Python packaging.\n",
        encoding="utf-8",
    )

    results = search_knowledge(corpus, query="Ampere cp.async shared staging alignment")

    assert results
    assert results[0].chunk.path == "ampere.md"
    assert "cp.async" in results[0].chunk.text


def test_build_knowledge_context_indexes_sibling_knowledge_files_from_file(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "knowledge"
    references = corpus / "references"
    references.mkdir(parents=True)
    (corpus / "ampere.md").write_text("Ampere baseline notes.\n", encoding="utf-8")
    (references / "fa2.md").write_text(
        "FlashAttention-2 on sm86 uses K/V tiling and tensor core MMA.\n",
        encoding="utf-8",
    )

    context = build_knowledge_context(
        corpus / "ampere.md",
        query="FlashAttention-2 sm86 K/V tiling",
    )

    assert "Retrieved knowledge context" in context
    assert "references/fa2.md" in context
    assert "K/V tiling" in context


def test_build_knowledge_context_falls_back_for_missing_source(tmp_path: Path) -> None:
    context = build_knowledge_context(tmp_path / "missing", query="ampere")

    assert "No supported knowledge files were found" in context


@pytest.mark.parametrize(
    ("query", "expected_terms"),
    [
        (
            "Ampere cp.async 16-byte aligned groups scalar BF16 async copy real dataflow",
            ("cp.async", "16-byte", "bf16", "real overlap"),
        ),
        (
            "compile score torch extension CUDA_NO_BFLOAT16_CONVERSIONS __nv_bfloat16 constructor",
            ("compile", "__cuda_no_bfloat16_conversions__", "__nv_bfloat16", "torch extension"),
        ),
        (
            "CUTLASS Ampere FlashAttention v2 128x128 128 threads swizzled online softmax",
            ("cutlass", "128x128", "128 threads", "online softmax"),
        ),
        (
            "CUTLASS Ampere FlashAttention cp.async ldmatrix register pipeline 128-bit K V staging",
            ("cp.async", "ldmatrix", "register pipeline", "128-bit"),
        ),
        (
            "FlashAttention SM80 Q in regs Share_Q_K_smem cp_async K V "
            "pipeline q_shared regression",
            ("q-in-register", "share q/k", "cp.async", "q_shared"),
        ),
        (
            "semantic transform mismatch contract-only set_constexpr_int dataflow staging",
            ("set_constexpr_int", "contract-only", "dataflow", "semantic"),
        ),
        (
            "candidate_transform structural preflight rejection invalidates pending "
            "compile-only score followup",
            ("structural preflight rejection", "compile-only", "score follow-up"),
        ),
        (
            "candidate_transform ambiguous anchor matching start lines larger unique anchor",
            ("ambiguous", "matching start line", "unique anchor"),
        ),
        (
            "candidate_transform replace_block_once coherent loop body helper replacement "
            "semantic transform raw diff",
            ("replace_block_once", "coherent loop/body/helper", "raw CUDA diffs"),
        ),
        (
            "compile repair episode failed edit payload replay rejected",
            ("repair", "failed edit payload", "execution"),
        ),
        (
            "repair validation feedback invalid repair decision not executed retry "
            "same repair episode",
            ("repair validation feedback", "invalid repair decision", "not executed"),
        ),
        (
            "transcript compaction recent messages verbatim structured breadcrumbs "
            "durable files lineage attempts",
            ("recent turns verbatim", "structured breadcrumbs", "durable files"),
        ),
        (
            "stagnation supervisor strategy reset candidates Ampere work "
            "decomposition memory layout register softmax diagnostic",
            ("strategy-reset", "work decomposition", "source-verifiable transform"),
        ),
        (
            "score_time_compile_failure torch extension build repair candidate_source_files",
            ("score_time_compile_failure", "compile-style repair", "candidate_source_files"),
        ),
        (
            "candidate_transform batch native steps array steps_json legacy fallback",
            ("native `steps`", "steps_json", "legacy fallback"),
        ),
        (
            "synchronous Q shared memory staging regression geomean 6.722112165053056",
            ("q shared-memory staging", "6.722112165053056", "regressed"),
        ),
        (
            "FlashAttention-2 baseline comparison lineage acceptance threshold",
            ("flashattention-2", "baseline", "lineage acceptance threshold"),
        ),
        (
            "lineage baseline_comparisons candidate_vs_baseline FA2 gap_tflops",
            ("baseline_comparisons", "candidate-vs-baseline", "gap"),
        ),
        (
            "candidate companion source directory generated CUDA helper runtime "
            "torch cpp_extension load AVO_SOURCE_FILES lineage snapshot",
            (
                "companion source directories",
                "generated companion-directory helpers",
                "python modules dynamically",
                "torch.utils.cpp_extension.load",
                "AVO_SOURCE_FILES",
                "lineage snapshots",
            ),
        ),
        (
            "wmma_fragment_shape Ampere BF16 fragment dimension outside supported 16x16x16",
            ("wmma", "bf16", "16x16x16"),
        ),
        (
            "kTile 32 query tile second 16-row subtile MMA dataflow rows uncovered",
            ("ktile=32", "second 16-row", "sub-tile", "rows uncovered"),
        ),
        (
            "kThreads 64 rejected geomean 7.587127963961811 occupancy",
            ("kthreads=64", "7.587127963961811", "occupancy"),
        ),
        (
            "WMMA chunk unroll by 2 rejected geomean 7.758592599549404",
            ("chunk-loop unroll-by-2", "7.758592599549404", "regressed"),
        ),
        (
            "q_shared Q staging repaired anchors geomean 6.686302249012325 shared memory 14016",
            ("q-staging", "6.686302249012325", "14016 bytes", "regressed"),
        ),
        (
            "synchronous K shared memory staging regression geomean "
            "4.16538030902376 k_shared key_start",
            ("k shared-memory staging", "4.16538030902376", "k_shared", "key_start"),
        ),
        (
            "CUDA execution model grid block thread warp SIMT divergence",
            ("grid", "blocks", "warps", "simt"),
        ),
        (
            "CUDA memory spaces global shared register local constant cache",
            ("global memory", "shared memory", "registers", "local memory"),
        ),
        (
            "CUDA global memory coalescing warp consecutive lanes transactions",
            ("coalescing", "warp", "consecutive lanes", "transactions"),
        ),
        (
            "CUDA shared memory synchronization bank conflicts tiling",
            ("shared memory", "__syncthreads", "bank", "tiling"),
        ),
        (
            "CUDA shared staging buffer no effect must be loaded stored consumed",
            ("shared-memory staging buffer", "executable dataflow", "no-effect"),
        ),
        (
            "CUDA occupancy registers shared memory spills ptxas",
            ("occupancy", "registers", "shared memory", "ptxas"),
        ),
        (
            "CUDA optimization workflow hypothesis measure profile transform",
            ("hypothesis", "correctness", "profile", "negative results"),
        ),
        (
            "CUDA work decomposition thread block tile warp row mapping coalesced layout",
            ("work decomposition", "warp", "tile", "coalesced"),
        ),
        (
            "CUDA indexing threadIdx x fastest linearization data layout tail predicates",
            ("threadidx.x", "fastest", "linearization", "tail"),
        ),
        (
            "CUDA memory movement shared tile global load sync reuse bank conflict padding",
            ("shared-memory tile", "global load", "reuse", "bank"),
        ),
        (
            "CUDA synchronization __syncthreads warp shuffle atomics cross block reduction",
            ("__syncthreads", "warp shuffles", "atomics", "global barrier"),
        ),
        (
            "CUDA tiling double buffering cp.async memcpy_async pipeline producer consumer overlap",
            ("double buffering", "cp.async", "producer/consumer", "overlap"),
        ),
        (
            "CUDA tensor core WMMA fragments opaque shape layout leading dimension",
            ("tensor cores", "wmma fragments", "opaque", "leading dimension"),
        ),
        (
            "CUDA streams events asynchronous host device default stream synchronization",
            ("streams", "events", "default stream", "synchronization"),
        ),
        (
            "CUDA profiling realistic workloads SpeedOfLight occupancy memory workload roofline",
            ("realistic workloads", "speedoflight", "occupancy", "memory workload"),
        ),
        (
            "CUDA semantic transform smallest coherent transformation invariants hypothesis",
            ("smallest coherent transformation", "invariants", "hypothesis"),
        ),
    ],
)
def test_real_ampere_corpus_retrieves_useful_claims(
    query: str,
    expected_terms: tuple[str, ...],
) -> None:
    context = build_knowledge_context(
        Path("knowledge/ampere.md"),
        query=query,
        max_chunks=8,
        max_chars=20_000,
    ).lower()

    assert "retrieved knowledge context" in context
    assert "#chunk-" in context
    for term in expected_terms:
        assert term.lower() in context


def test_real_ampere_corpus_retrieval_is_bounded_and_auditable() -> None:
    context = build_knowledge_context(
        Path("knowledge/ampere.md"),
        query="Ampere FlashAttention-2 cp.async WMMA semantic transform regression",
        max_chunks=4,
        max_chars=5_000,
    )

    assert len(context) <= 5_100
    assert "Knowledge source: knowledge/ampere.md" in context
    assert "lines " in context
    assert "score=" in context
    assert "No supported knowledge files were found" not in context


def test_real_ampere_corpus_query_changes_top_result() -> None:
    async_results = search_knowledge(
        Path("knowledge/ampere.md"),
        query="cp.async 16-byte scalar BF16 aligned copy",
        max_chunks=1,
    )
    q_stage_results = search_knowledge(
        Path("knowledge/ampere.md"),
        query="synchronous Q shared memory staging regression 6.722112165053056",
        max_chunks=1,
    )

    assert async_results
    assert q_stage_results
    assert async_results[0].chunk.text != q_stage_results[0].chunk.text
    assert "cp.async" in async_results[0].chunk.text
    assert "6.722112165053056" in q_stage_results[0].chunk.text


def test_retrieval_claim_manifest_is_indexed_from_ampere_entrypoint() -> None:
    context = build_knowledge_context(
        Path("knowledge/ampere.md"),
        query="retrieved knowledge claims why useful kThreads 64 regression",
        max_chunks=4,
        max_chars=10_000,
    )

    assert "retrieval_claims.md" in context
    assert "Why useful" in context
    assert "kThreads=64" in context


def test_ampere_knowledge_prefers_native_batch_steps() -> None:
    context = build_knowledge_context(
        Path("knowledge/ampere.md"),
        query="candidate_transform batch native steps array steps_json legacy fallback",
        max_chunks=8,
        max_chars=16_000,
    )

    assert "native `steps` array" in context
    assert "steps_json" in context
    assert "legacy fallback" in context
    assert "schema-complexity errors" not in context
    assert "still advertises compact `steps_json`" not in context


def test_general_cuda_grounding_is_indexed_from_ampere_entrypoint() -> None:
    context = build_knowledge_context(
        Path("knowledge/ampere.md"),
        query="CUDA execution model memory spaces occupancy profiling workflow",
        max_chunks=8,
        max_chars=16_000,
    )

    assert "b/cuda_general.md" in context
    assert "b/cuda_programming_practice.md" in context
    assert "General CUDA Working Knowledge" in context
    assert "Profiling And Optimization Workflow" in context
    assert "smallest coherent transformation" in context
    assert "Execution Model" in context
    assert "How CUDA Programmers Usually Approach Optimization" in context


def test_agent_planning_query_retrieves_cuda_practice_context() -> None:
    query = _knowledge_query(
        lineage_summary="Current lineage is an sm86 BF16 WMMA attention seed.",
        attempt_history=(
            "Recent attempts include K shared-memory staging and chunk-loop unrolling."
        ),
        repo_context=(
            "candidate_transform should make the smallest coherent semantic "
            "transformation under candidates/."
        ),
    )

    context = build_knowledge_context(
        Path("knowledge/ampere.md"),
        query=query,
        max_chunks=10,
        max_chars=24_000,
    )

    assert "b/cuda_programming_practice.md" in context
    assert "CUDA Kernel Design Practice" in context
    assert "work decomposition" in context.lower()
    assert "smallest coherent transformation" in context


def test_every_claim_manifest_query_retrieves_useful_manifest_context() -> None:
    manifest = Path("knowledge/retrieval_claims.md")
    queries = re.findall(r"Retrieval query: `([^`]+)`", manifest.read_text(encoding="utf-8"))

    assert len(queries) >= 16
    for query in queries:
        context = build_knowledge_context(
            Path("knowledge/ampere.md"),
            query=query,
            max_chunks=12,
            max_chars=24_000,
        )
        assert "retrieval_claims.md" in context, query
        assert "Why useful" in context, query
        assert "No supported knowledge files were found" not in context
