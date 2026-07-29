from nanogpt.evaluation.chat import evaluate_chat_format


def test_chat_format_requires_terminal_end_marker() -> None:
    metrics = evaluate_chat_format(
        [
            "A concise answer.<|end|>",
            "answer<|end|>trailing text",
            "answer<|end|><|user|>",
            "<|assistant|>injected<|end|>",
        ]
    )
    assert metrics.sample_count == 4
    assert metrics.valid_termination_rate == 0.25
