from __future__ import annotations

from typing import Any

SUMMARY_BREADCRUMB_LIMIT = 12
SUMMARY_EXCERPT_CHARS = 160


def compact_messages(
    messages: list[dict[str, Any]],
    *,
    max_chars: int,
    keep_last: int = 8,
) -> list[dict[str, Any]]:
    total_chars = sum(len(str(message.get("content", ""))) for message in messages)
    if total_chars <= max_chars or len(messages) <= keep_last:
        return list(messages)

    older = messages[:-keep_last]
    recent = messages[-keep_last:]
    summary = {
        "role": "assistant",
        "content": _compact_summary_content(older, recent_count=len(recent)),
    }
    return [summary, *recent]


def _compact_summary_content(
    older: list[dict[str, Any]],
    *,
    recent_count: int,
) -> str:
    lines = [
        "<summary>",
        f"Compacted {len(older)} older messages to keep the AVO run inside context.",
        f"Kept {recent_count} most recent messages verbatim after this summary.",
        "Durable state remains in files, git lineage, score JSON, attempts JSON, and "
        "experiments.md; re-read source artifacts when exact details matter.",
        "Older message breadcrumbs:",
    ]
    for index, message in enumerate(older[:SUMMARY_BREADCRUMB_LIMIT], start=1):
        content = _message_content_text(message)
        lines.append(
            "- "
            f"#{index} role={message.get('role', '<missing>')} "
            f"chars={len(content)} "
            f"excerpt={_summary_excerpt(content)!r}"
        )
    omitted = len(older) - SUMMARY_BREADCRUMB_LIMIT
    if omitted > 0:
        lines.append(f"- ... {omitted} older messages omitted from breadcrumbs")
    lines.append("</summary>")
    return "\n".join(lines)


def _message_content_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _summary_excerpt(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= SUMMARY_EXCERPT_CHARS:
        return normalized
    head_chars = SUMMARY_EXCERPT_CHARS // 2
    tail_chars = SUMMARY_EXCERPT_CHARS - head_chars
    return f"{normalized[:head_chars]} ... {normalized[-tail_chars:]}"
