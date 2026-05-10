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
    recent_chars = sum(len(_message_content_text(message)) for message in recent)
    summary_budget = max(0, max_chars - recent_chars)
    summary = {
        "role": "assistant",
        "content": _compact_summary_content(
            older,
            recent_count=len(recent),
            max_chars=summary_budget,
        ),
    }
    return [summary, *recent]


def _compact_summary_content(
    older: list[dict[str, Any]],
    *,
    recent_count: int,
    max_chars: int | None = None,
) -> str:
    lines = _summary_header(
        older_count=len(older),
        recent_count=recent_count,
        compact=False,
    )
    lines.extend(_summary_breadcrumb_lines(older, limit=SUMMARY_BREADCRUMB_LIMIT))
    lines.append("</summary>")
    content = "\n".join(lines)
    if max_chars is None or len(content) <= max_chars:
        return content
    return _bounded_compact_summary_content(
        older,
        recent_count=recent_count,
        max_chars=max_chars,
    )


def _bounded_compact_summary_content(
    older: list[dict[str, Any]],
    *,
    recent_count: int,
    max_chars: int,
) -> str:
    if max_chars <= 0:
        return ""
    lines = _summary_header(
        older_count=len(older),
        recent_count=recent_count,
        compact=True,
    )
    if len("\n".join([*lines, "</summary>"])) > max_chars:
        return _clip_text(
            (
                "<summary>\n"
                f"Compacted {len(older)} older messages. Kept {recent_count} recent. "
                "Recover exact state from files, lineage, attempts, scores, and "
                "experiments.md.\n"
                "</summary>"
            ),
            max_chars,
        )

    kept = 0
    for line in _summary_breadcrumb_lines(
        older,
        limit=SUMMARY_BREADCRUMB_LIMIT,
        excerpt_chars=64,
        include_omitted=False,
    ):
        candidate = [*lines, line]
        omitted = len(older) - kept - 1
        if omitted > 0:
            candidate.append(f"- ... {omitted} older messages omitted from breadcrumbs")
        candidate.append("</summary>")
        if len("\n".join(candidate)) > max_chars:
            break
        lines.append(line)
        kept += 1

    omitted = len(older) - kept
    omitted_line = f"- ... {omitted} older messages omitted from breadcrumbs"
    if omitted > 0 and len("\n".join([*lines, omitted_line, "</summary>"])) <= max_chars:
        lines.append(omitted_line)
    lines.append("</summary>")
    return "\n".join(lines)


def _summary_header(
    *,
    older_count: int,
    recent_count: int,
    compact: bool,
) -> list[str]:
    durable_line = (
        "Durable state: files, git lineage, score JSON, attempts JSON, and experiments.md."
        if compact
        else (
            "Durable state remains in files, git lineage, score JSON, attempts JSON, and "
            "experiments.md; re-read source artifacts when exact details matter."
        )
    )
    return [
        "<summary>",
        f"Compacted {older_count} older messages to keep the AVO run inside context.",
        f"Kept {recent_count} most recent messages verbatim after this summary.",
        durable_line,
        "Older message breadcrumbs:",
    ]


def _summary_breadcrumb_lines(
    older: list[dict[str, Any]],
    *,
    limit: int,
    excerpt_chars: int = SUMMARY_EXCERPT_CHARS,
    include_omitted: bool = True,
) -> list[str]:
    lines = []
    for index, message in enumerate(older[:limit], start=1):
        content = _message_content_text(message)
        lines.append(
            "- "
            f"#{index} role={message.get('role', '<missing>')} "
            f"chars={len(content)} "
            f"excerpt={_summary_excerpt(content, max_chars=excerpt_chars)!r}"
        )
    omitted = len(older) - limit
    if include_omitted and omitted > 0:
        lines.append(f"- ... {omitted} older messages omitted from breadcrumbs")
    return lines


def _message_content_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _summary_excerpt(text: str, *, max_chars: int = SUMMARY_EXCERPT_CHARS) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return f"{normalized[:head_chars]} ... {normalized[-tail_chars:]}"


def _clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."
