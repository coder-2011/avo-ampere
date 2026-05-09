from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KNOWLEDGE_MAX_CHARS = 24_000
DEFAULT_KNOWLEDGE_MAX_CHUNKS = 10
DEFAULT_KNOWLEDGE_CHUNK_CHARS = 2_600
KNOWLEDGE_SUFFIXES = frozenset({".cu", ".cuh", ".cpp", ".h", ".hpp", ".md", ".py", ".txt"})
SKIPPED_KNOWLEDGE_PARTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "attempts",
        "build",
        "lineage",
    }
)
STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "against",
        "also",
        "and",
        "are",
        "before",
        "both",
        "but",
        "can",
        "cannot",
        "current",
        "does",
        "for",
        "from",
        "has",
        "have",
        "into",
        "not",
        "only",
        "that",
        "the",
        "then",
        "this",
        "through",
        "use",
        "with",
        "without",
    }
)


@dataclass(frozen=True)
class KnowledgeChunk:
    path: str
    index: int
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class KnowledgeSearchResult:
    chunk: KnowledgeChunk
    score: float


def build_knowledge_context(
    source: Path,
    *,
    query: str,
    max_chars: int = DEFAULT_KNOWLEDGE_MAX_CHARS,
    max_chunks: int = DEFAULT_KNOWLEDGE_MAX_CHUNKS,
) -> str:
    results = search_knowledge(source, query=query, max_chunks=max_chunks)
    if not results:
        return _fallback_knowledge_context(source)

    lines = [
        "Retrieved knowledge context from local corpus.",
        f"Knowledge source: {source.as_posix()}",
        "Retrieval: deterministic lexical chunk search over local docs/source notes.",
        "Use snippets as planning context; inspect exact files before relying on "
        "line-level details.",
        "",
    ]
    for result in results:
        chunk = result.chunk
        lines.extend(
            [
                (
                    f"-- {chunk.path}#chunk-{chunk.index} "
                    f"lines {chunk.start_line}-{chunk.end_line} score={result.score:.3f} --"
                ),
                chunk.text.rstrip(),
                f"-- end {chunk.path}#chunk-{chunk.index} --",
                "",
            ]
        )
        if len("\n".join(lines)) >= max_chars:
            break
    context = "\n".join(lines).rstrip()
    if len(context) <= max_chars:
        return context
    return context[:max_chars].rstrip() + "\n... truncated retrieved knowledge context ..."


def search_knowledge(
    source: Path,
    *,
    query: str,
    max_chunks: int = DEFAULT_KNOWLEDGE_MAX_CHUNKS,
) -> list[KnowledgeSearchResult]:
    chunks = _load_chunks(source)
    if not chunks or max_chunks <= 0:
        return []
    query_terms = _tokens(query)
    if not query_terms:
        return [KnowledgeSearchResult(chunk=chunk, score=0.0) for chunk in chunks[:max_chunks]]

    query_counts = Counter(query_terms)
    document_frequency: Counter[str] = Counter()
    chunk_terms: list[Counter[str]] = []
    for chunk in chunks:
        counts = Counter(_tokens(f"{chunk.path}\n{chunk.text}"))
        chunk_terms.append(counts)
        document_frequency.update(counts)

    results: list[KnowledgeSearchResult] = []
    corpus_size = len(chunks)
    for chunk, counts in zip(chunks, chunk_terms, strict=True):
        score = 0.0
        for term, query_count in query_counts.items():
            frequency = counts.get(term, 0)
            if frequency <= 0:
                continue
            idf = math.log(1.0 + corpus_size / (1.0 + document_frequency[term]))
            score += query_count * min(frequency, 5) * idf
        if score > 0.0:
            score /= math.sqrt(max(1, sum(counts.values())))
        results.append(KnowledgeSearchResult(chunk=chunk, score=score))

    return sorted(
        results,
        key=lambda result: (-result.score, result.chunk.path, result.chunk.index),
    )[:max_chunks]


def _fallback_knowledge_context(source: Path) -> str:
    if source.is_file():
        content = source.read_text(encoding="utf-8", errors="replace").strip()
        if content:
            return content[:DEFAULT_KNOWLEDGE_MAX_CHARS]
    return (
        "Retrieved knowledge context from local corpus.\n"
        f"Knowledge source: {source.as_posix()}\n"
        "No supported knowledge files were found."
    )


def _load_chunks(source: Path) -> list[KnowledgeChunk]:
    files = _knowledge_files(source)
    chunks: list[KnowledgeChunk] = []
    for path in files:
        chunks.extend(_chunk_file(path, label=_knowledge_label(path, source)))
    return chunks


def _knowledge_files(source: Path) -> list[Path]:
    if source.is_file():
        if source.parent.name == "knowledge":
            root = source.parent
            files = [
                path
                for path in root.rglob("*")
                if _is_supported_knowledge_file(path) and _is_not_skipped(path, root)
            ]
            if source not in files and _is_supported_knowledge_file(source):
                files.append(source)
            return sorted(dict.fromkeys(files))
        return [source] if _is_supported_knowledge_file(source) else []
    if not source.is_dir():
        return []
    return sorted(
        path
        for path in source.rglob("*")
        if _is_supported_knowledge_file(path) and _is_not_skipped(path, source)
    )


def _is_supported_knowledge_file(path: Path) -> bool:
    return path.is_file() and path.suffix in KNOWLEDGE_SUFFIXES


def _is_not_skipped(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return not any(
        part in SKIPPED_KNOWLEDGE_PARTS or part.startswith(".")
        for part in relative.parts
    )


def _knowledge_label(path: Path, source: Path) -> str:
    base = source.parent if source.is_file() and source.parent.name == "knowledge" else source
    if source.is_file() and source.parent.name != "knowledge":
        base = source.parent
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _chunk_file(
    path: Path,
    *,
    label: str,
    max_chunk_chars: int = DEFAULT_KNOWLEDGE_CHUNK_CHARS,
) -> list[KnowledgeChunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    chunks: list[KnowledgeChunk] = []
    start = 0
    while start < len(lines):
        current: list[str] = []
        end = start
        while end < len(lines):
            candidate = [*current, lines[end]]
            if current and len("\n".join(candidate)) > max_chunk_chars:
                break
            current = candidate
            end += 1
        if not current:
            current = [lines[start]]
            end = start + 1
        chunks.append(
            KnowledgeChunk(
                path=label,
                index=len(chunks),
                start_line=start + 1,
                end_line=end,
                text="\n".join(current).strip(),
            )
        )
        start = end
    return [chunk for chunk in chunks if chunk.text]


def _tokens(text: str) -> list[str]:
    normalized = text.lower().replace("cp.async", "cp_async").replace("sm_86", "sm86")
    raw_tokens = re.findall(r"[a-z][a-z0-9_]*|\d{3,}", normalized)
    return [
        token
        for token in raw_tokens
        if token not in STOPWORDS and (len(token) >= 3 or token in {"q", "k", "v"})
    ]
