# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from benchmarks.watermarking.analyze import (
    context_recurrence_rate,
    distinct_n,
    majority_correct,
    repetition_rate,
    select_key_averaged_records,
    unique_completion_rate,
)
from benchmarks.watermarking.generate import coarsened_entropy
from benchmarks.watermarking.run_matrix import (
    generation_commands,
    performance_commands,
    selected_variants,
)


def test_diversity_metrics_distinguish_repeated_outputs():
    repeated = [[1, 2, 1, 2], [1, 2, 1, 2]]
    diverse = [[1, 2, 3, 4], [5, 6, 7, 8]]

    assert distinct_n(repeated, 2) < distinct_n(diverse, 2)
    assert repetition_rate(repeated[0], 2) > repetition_rate(diverse[0], 2)
    assert context_recurrence_rate(repeated[0], 1) > 0
    assert unique_completion_rate(["same", "same"]) == 0.5


def test_majority_correct_requires_unique_winner():
    records = [
        {
            "sample_index": index,
            "completion": completion,
            "reference": 42,
        }
        for index, completion in enumerate(["42", "42", "7"])
    ]

    assert majority_correct(records, 3) == 1


def test_coarsened_entropy_includes_unreported_probability_mass():
    entropy = coarsened_entropy([{"a": -0.6931471805599453}])

    assert entropy == 0.6931471805599453


def test_key_average_pairs_keys_with_sampling_seeds():
    records = [
        {"key": key, "seed": seed} for key in (None, 10, 20) for seed in (100, 200, 300)
    ]
    selected = select_key_averaged_records(
        records,
        {"keys": [10, 20], "sampling_seeds": [100, 200, 300]},
    )

    assert {(record["key"], record["seed"]) for record in selected} == {
        (None, 100),
        (None, 200),
        (10, 100),
        (20, 200),
    }


def test_performance_matrix_enables_watermark_only_for_enabled_variant():
    config = {
        "model": "model",
        "keys": [42],
        "server_args": [],
        "watermark": {
            "algorithm": "gumbel",
            "prf": "philox",
            "context_width": 4,
        },
        "performance": {
            "repetitions": 1,
            "input_length": 8,
            "output_length": 16,
            "batch_sizes": [1],
            "latency_warmups": 1,
            "latency_iterations": 1,
            "throughput_prompts": 2,
        },
    }
    disabled = {
        "name": "disabled",
        "command": ["vllm"],
        "watermarked": False,
    }
    enabled = {**disabled, "name": "enabled", "watermarked": True}

    disabled_commands = list(performance_commands(config, disabled, Path("disabled")))
    enabled_commands = list(performance_commands(config, enabled, Path("enabled")))

    assert all(
        "watermark-config" not in " ".join(command) for command in disabled_commands
    )
    assert all("watermark-config" in " ".join(command) for command in enabled_commands)


def test_generation_matrix_uses_full_sweep_for_one_fixed_key():
    config = {
        "model": "model",
        "results_dir": "results",
        "keys": [10, 20],
        "sampling_seeds": [100, 200, 300],
        "generation": {
            "datasets": ["gsm8k"],
            "num_prompts": {"gsm8k": 2},
            "num_fewshot": 0,
            "temperatures": [1.0],
            "max_tokens": 16,
            "max_concurrency": 1,
            "request_timeout_seconds": 10,
            "top_logprobs": 2,
            "fixed_key_diversity_keys": 1,
        },
    }
    variant = {"name": "enabled", "command": ["vllm"], "watermarked": True}

    first = " ".join(next(iter(generation_commands(config, variant, 10))))
    second = " ".join(next(iter(generation_commands(config, variant, 20))))

    assert "--seeds 100 200 300" in first
    assert "--seeds 200" in second


def test_select_variants_preserves_requested_order():
    config = {"variants": [{"name": "a"}, {"name": "b"}]}

    assert selected_variants(config, ["b", "a"]) == [
        {"name": "b"},
        {"name": "a"},
    ]
