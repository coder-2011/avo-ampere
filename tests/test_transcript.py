from avo.transcript import compact_messages


def test_compact_messages_keeps_recent_context() -> None:
    messages = [{"role": "user", "content": f"message {index} " * 20} for index in range(12)]
    compacted = compact_messages(messages, max_chars=100, keep_last=3)
    assert compacted[0]["role"] == "assistant"
    assert "Compacted 9 older messages" in compacted[0]["content"]
    assert "Kept 3 most recent messages verbatim" in compacted[0]["content"]
    assert "Older message breadcrumbs:" in compacted[0]["content"]
    assert "#1 role=user" in compacted[0]["content"]
    assert compacted[1:] == messages[-3:]


def test_compact_messages_noops_under_threshold() -> None:
    messages = [{"role": "user", "content": "small"}]
    assert compact_messages(messages, max_chars=100) == messages


def test_compact_messages_bounds_older_breadcrumbs() -> None:
    messages = [
        {"role": "tool", "content": f"tool output {index} " * 30} for index in range(20)
    ]

    compacted = compact_messages(messages, max_chars=100, keep_last=2)

    summary = compacted[0]["content"]
    assert "#12 role=tool" in summary
    assert "#13 role=tool" not in summary
    assert "... 6 older messages omitted from breadcrumbs" in summary
    assert compacted[1:] == messages[-2:]


def test_compact_messages_summarizes_long_content_with_head_and_tail() -> None:
    messages = [
        {"role": "tool", "content": "alpha " + ("middle " * 80) + "omega"},
        {"role": "assistant", "content": "recent"},
    ]

    compacted = compact_messages(messages, max_chars=100, keep_last=1)

    summary = compacted[0]["content"]
    assert "alpha" in summary
    assert "omega" in summary
    assert "middle middle" in summary
    assert compacted[1:] == messages[-1:]
