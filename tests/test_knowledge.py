from pathlib import Path

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
