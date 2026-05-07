from __future__ import annotations

from typing import Any


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
        "content": (
            "<summary>\n"
            f"Compacted {len(older)} older messages to keep the AVO run inside context.\n"
            "Preserve durable state from files, git lineage, score JSON, and experiments.md. "
            "Re-read source artifacts when exact details matter.\n"
            "</summary>"
        ),
    }
    return [summary, *recent]
