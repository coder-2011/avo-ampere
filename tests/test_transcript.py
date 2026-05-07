from avo.transcript import compact_messages


def test_compact_messages_keeps_recent_context() -> None:
    messages = [{"role": "user", "content": f"message {index} " * 20} for index in range(12)]
    compacted = compact_messages(messages, max_chars=100, keep_last=3)
    assert compacted[0]["role"] == "assistant"
    assert "Compacted 9 older messages" in compacted[0]["content"]
    assert compacted[1:] == messages[-3:]


def test_compact_messages_noops_under_threshold() -> None:
    messages = [{"role": "user", "content": "small"}]
    assert compact_messages(messages, max_chars=100) == messages
