from nanogpt.evaluate import build_report, evaluate_chat_format, evaluate_reasoning


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


def test_reasoning_metrics_and_release_gates() -> None:
    metrics = evaluate_reasoning([["<final>4</final>"], ["no answer"]], ["4", "5"])
    assert metrics.exact_match == 0.5
    assert metrics.malformed_answer_rate == 0.5
    report = build_report(
        base={"loss": 3.0},
        sft={"format_valid_rate": 0.96, "exact_match": 0.2},
        grpo={"exact_match": 0.35},
        run={"estimated_cost_usd": 90.0, "budget_usd": 100.0},
    )
    assert report["release_gates"]["passed"] is True  # type: ignore[index]
