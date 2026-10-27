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


def test_rl_may_not_regress_the_easiest_bucket() -> None:
    def gates(grpo_easiest: float) -> dict[str, bool]:
        report = build_report(
            base={"loss": 3.0},
            sft={"format_valid_rate": 1.0, "exact_match": 0.3, "easiest_exact_match": 1.0},
            grpo={"exact_match": 0.5, "easiest_exact_match": grpo_easiest},
            run={"estimated_cost_usd": 1.0, "budget_usd": 100.0},
        )
        return report["release_gates"]  # type: ignore[return-value]

    assert gates(0.99)["passed"] is True
    assert gates(0.9)["rl_no_regression_on_easiest_bucket"] is False
    assert gates(0.9)["passed"] is False
