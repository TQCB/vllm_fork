# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Aggregate watermark performance, quality, diversity, and detection results."""

import argparse
import collections
import json
import math
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tests.evals.gsm8k.gsm8k_eval import get_answer_value
from vllm.tokenizers import cached_get_tokenizer
from vllm.v1.watermarking import GumbelWatermarkDetector


def ngrams(tokens: Sequence[Any], order: int) -> list[tuple[Any, ...]]:
    return [
        tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1)
    ]


def unique_completion_rate(completions: Sequence[str]) -> float:
    normalized = {" ".join(completion.split()) for completion in completions}
    return len(normalized) / len(completions) if completions else 0.0


def distinct_n(token_sequences: Sequence[Sequence[int]], order: int) -> float:
    grams = [gram for tokens in token_sequences for gram in ngrams(tokens, order)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repetition_rate(tokens: Sequence[int], order: int) -> float:
    grams = ngrams(tokens, order)
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def context_recurrence_rate(tokens: Sequence[int], width: int) -> float:
    contexts = [
        tuple(tokens[max(0, index - width) : index]) for index in range(len(tokens))
    ]
    seen: set[tuple[int, ...]] = set()
    repeated = 0
    for context in contexts:
        if context in seen:
            repeated += 1
        seen.add(context)
    return repeated / len(contexts) if contexts else 0.0


def bleu(candidate: Sequence[int], references: Sequence[Sequence[int]]) -> float:
    if not candidate or not references:
        return 0.0
    log_precisions = []
    for order in range(1, 5):
        candidate_counts = collections.Counter(ngrams(candidate, order))
        reference_counts: collections.Counter[tuple[int, ...]] = collections.Counter()
        for reference in references:
            counts = collections.Counter(ngrams(reference, order))
            for gram, count in counts.items():
                reference_counts[gram] = max(reference_counts[gram], count)
        matches = sum(
            min(count, reference_counts[gram])
            for gram, count in candidate_counts.items()
        )
        total = sum(candidate_counts.values())
        log_precisions.append(math.log((matches + 1) / (total + 1)))

    reference_length = min(
        (len(reference) for reference in references),
        key=lambda length: (abs(length - len(candidate)), length),
    )
    brevity_penalty = (
        1.0
        if len(candidate) >= reference_length
        else math.exp(1.0 - reference_length / len(candidate))
    )
    return brevity_penalty * math.exp(sum(log_precisions) / len(log_precisions))


def self_bleu(token_sequences: Sequence[Sequence[int]]) -> float:
    if len(token_sequences) < 2:
        return 1.0
    values = [
        bleu(
            candidate,
            [
                reference
                for other_index, reference in enumerate(token_sequences)
                if other_index != index
            ],
        )
        for index, candidate in enumerate(token_sequences)
    ]
    return statistics.mean(values)


def majority_correct(records: Sequence[dict[str, Any]], k: int) -> float:
    selected = sorted(records, key=lambda record: record["sample_index"])[:k]
    predictions = [get_answer_value(record["completion"]) for record in selected]
    counts = collections.Counter(predictions)
    most_common = counts.most_common()
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return 0.0
    return float(most_common[0][0] == selected[0]["reference"])


def load_generation_records(results_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((results_dir / "generation").glob("**/*.jsonl")):
        with path.open() as input_file:
            records.extend(json.loads(line) for line in input_file)
    return records


def group_records(
    records: Iterable[dict[str, Any]], fields: Sequence[str]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        groups[tuple(record[field] for field in fields)].append(record)
    return groups


def entropy_buckets(
    records: Sequence[dict[str, Any]], reference_variant: str, num_buckets: int
) -> dict[tuple[str, float], int]:
    reference = [
        record
        for record in records
        if record["variant"] == reference_variant
        and record["mean_coarsened_entropy"] is not None
    ]
    per_prompt = group_records(reference, ("task_id", "temperature"))
    means = {
        key: statistics.mean(
            record["mean_coarsened_entropy"] for record in prompt_records
        )
        for key, prompt_records in per_prompt.items()
    }
    buckets = {}
    by_temperature: dict[float, list[tuple[tuple[str, float], float]]] = (
        collections.defaultdict(list)
    )
    for key, value in means.items():
        by_temperature[key[1]].append((key, value))
    for values in by_temperature.values():
        values.sort(key=lambda item: item[1])
        for rank, (key, _) in enumerate(values):
            buckets[key] = min(num_buckets - 1, rank * num_buckets // len(values))
    return buckets


def analyze_generation(
    config: dict[str, Any], results_dir: Path
) -> list[dict[str, Any]]:
    records = load_generation_records(results_dir)
    if not records:
        return []
    tokenizer = cached_get_tokenizer(config["model"])
    token_ids = {
        id(record): tokenizer.encode(record["completion"], add_special_tokens=False)
        for record in records
    }
    settings = config["analysis"]
    prompt_buckets = entropy_buckets(
        records,
        settings["entropy_reference_variant"],
        settings["entropy_buckets"],
    )
    prompt_groups = group_records(
        records, ("variant", "dataset", "key", "temperature", "task_id")
    )
    summaries = []
    for (
        variant,
        dataset,
        key,
        temperature,
        task_id,
    ), prompt_records in prompt_groups.items():
        sequences = [token_ids[id(record)] for record in prompt_records]
        summary: dict[str, Any] = {
            "variant": variant,
            "dataset": dataset,
            "key": key,
            "temperature": temperature,
            "task_id": task_id,
            "num_samples": len(prompt_records),
            "entropy_bucket": prompt_buckets.get((task_id, temperature)),
            "unique_completion_rate": unique_completion_rate(
                [record["completion"] for record in prompt_records]
            ),
            "self_bleu": self_bleu(sequences),
        }
        for order in settings["ngram_orders"]:
            summary[f"distinct_{order}"] = distinct_n(sequences, order)
            summary[f"rep_{order}"] = statistics.mean(
                repetition_rate(sequence, order) for sequence in sequences
            )
        for width in settings["context_widths"]:
            summary[f"context_recurrence_{width}"] = statistics.mean(
                context_recurrence_rate(sequence, width) for sequence in sequences
            )
        if dataset == "gsm8k":
            summary["sample_accuracy"] = statistics.mean(
                get_answer_value(record["completion"]) == record["reference"]
                for record in prompt_records
            )
            for k in settings["majority_k"]:
                if len(prompt_records) >= k:
                    summary[f"maj_at_{k}"] = majority_correct(prompt_records, k)
        summaries.append(summary)

    aggregate_fields = sorted(
        {
            field
            for summary in summaries
            for field, value in summary.items()
            if isinstance(value, float)
        }
    )
    summaries_with_overall = [
        *summaries,
        *[{**summary, "entropy_bucket": "all"} for summary in summaries],
    ]
    grouped = group_records(
        summaries_with_overall,
        ("variant", "dataset", "key", "temperature", "entropy_bucket"),
    )
    aggregates = []
    for group_key, values in grouped.items():
        aggregate = dict(
            zip(
                ("variant", "dataset", "key", "temperature", "entropy_bucket"),
                group_key,
                strict=True,
            )
        )
        aggregate["num_prompts"] = len(values)
        aggregate["samples_per_prompt"] = statistics.mean(
            value["num_samples"] for value in values
        )
        for field in aggregate_fields:
            field_values = [value[field] for value in values if field in value]
            if field_values:
                aggregate[field] = statistics.mean(field_values)
        aggregates.append(aggregate)

    detection_groups = group_records(
        records, ("variant", "dataset", "key", "temperature")
    )
    for group_key, values in detection_groups.items():
        variant, dataset, key, temperature = group_key
        scoring_keys = [key] if key is not None else config["keys"]
        z_scores = []
        positives = []
        for record in values:
            ids = token_ids[id(record)]
            for scoring_key in scoring_keys:
                detection = GumbelWatermarkDetector(
                    key=scoring_key,
                    context_width=config["watermark"]["context_width"],
                    p_value_threshold=settings["p_value_threshold"],
                    prf=config["watermark"]["prf"],
                ).detect(ids)
                if detection.num_scored_tokens:
                    z_scores.append(
                        (detection.score - detection.num_scored_tokens)
                        / math.sqrt(detection.num_scored_tokens)
                    )
                positives.append(detection.is_watermarked)
        aggregates.append(
            {
                "variant": variant,
                "dataset": dataset,
                "key": key,
                "temperature": temperature,
                "metric_group": "detection",
                "median_z_score": statistics.median(z_scores) if z_scores else None,
                "positive_rate": statistics.mean(positives) if positives else None,
            }
        )
    return aggregates


def analyze_key_averaged_quality(
    records: Sequence[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    selected = select_key_averaged_records(records, config)
    gsm8k = [record for record in selected if record["dataset"] == "gsm8k"]
    groups = group_records(gsm8k, ("variant", "temperature"))
    return [
        {
            "variant": variant,
            "temperature": temperature,
            "num_samples": len(values),
            "accuracy": statistics.mean(
                get_answer_value(value["completion"]) == value["reference"]
                for value in values
            ),
        }
        for (variant, temperature), values in groups.items()
    ]


def select_key_averaged_records(
    records: Sequence[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    key_indices = {key: index for index, key in enumerate(config["keys"])}
    selected = []
    matched_seeds = set(config["sampling_seeds"][: len(config["keys"])])
    for record in records:
        key = record["key"]
        if key is None:
            include = record["seed"] in matched_seeds
        else:
            include = record["seed"] == config["sampling_seeds"][key_indices[key]]
        if include:
            selected.append(record)
    return selected


def analyze_performance(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    root = results_dir / "performance"
    for variant_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        latency_groups: dict[str, list[float]] = collections.defaultdict(list)
        for path in variant_dir.glob("latency_*.json"):
            batch_size = path.stem.split("_")[1]
            with path.open() as input_file:
                latency_groups[batch_size].append(json.load(input_file)["avg_latency"])
        for batch_size, values in latency_groups.items():
            rows.append(
                {
                    "variant": variant_dir.name,
                    "benchmark": "latency",
                    "batch_size": int(batch_size.removeprefix("bs")),
                    "mean_seconds": statistics.mean(values),
                    "stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0,
                    "repetitions": len(values),
                }
            )
        throughput = []
        for path in variant_dir.glob("throughput_*.json"):
            with path.open() as input_file:
                throughput.append(json.load(input_file)["tokens_per_second"])
        if throughput:
            rows.append(
                {
                    "variant": variant_dir.name,
                    "benchmark": "throughput",
                    "mean_tokens_per_second": statistics.mean(throughput),
                    "stdev_tokens_per_second": (
                        statistics.stdev(throughput) if len(throughput) > 1 else 0
                    ),
                    "repetitions": len(throughput),
                }
            )
    return rows


def export_humaneval(records: Sequence[dict[str, Any]], output_dir: Path) -> None:
    groups = group_records(records, ("variant", "key", "temperature"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for (variant, key, temperature), values in groups.items():
        key_name = f"key_{key}" if key is not None else "unwatermarked"
        path = output_dir / f"{variant}_{key_name}_temperature_{temperature}.jsonl"
        with path.open("w") as output_file:
            for record in values:
                output_file.write(
                    json.dumps(
                        {
                            "task_id": record["task_id"],
                            "completion": record["completion"],
                        }
                    )
                    + "\n"
                )


def export_key_averaged_humaneval(
    records: Sequence[dict[str, Any]], config: dict[str, Any], output_dir: Path
) -> None:
    selected = select_key_averaged_records(records, config)
    groups = group_records(selected, ("variant", "temperature"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for (variant, temperature), values in groups.items():
        path = output_dir / f"{variant}_key_average_temperature_{temperature}.jsonl"
        with path.open("w") as output_file:
            for record in values:
                output_file.write(
                    json.dumps(
                        {
                            "task_id": record["task_id"],
                            "completion": record["completion"],
                        }
                    )
                    + "\n"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as config_file:
        config = json.load(config_file)
    results_dir = Path(config["results_dir"])
    if not results_dir.is_absolute():
        results_dir = (args.config.parent / results_dir).resolve()
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    generation_records = load_generation_records(results_dir)
    export_humaneval(
        [record for record in generation_records if record["dataset"] == "humaneval"],
        analysis_dir / "humaneval",
    )
    export_key_averaged_humaneval(
        [record for record in generation_records if record["dataset"] == "humaneval"],
        config,
        analysis_dir / "humaneval",
    )
    report = {
        "performance": analyze_performance(results_dir),
        "key_averaged_quality": analyze_key_averaged_quality(
            generation_records, config
        ),
        "generation": analyze_generation(config, results_dir),
    }
    with (analysis_dir / "report.json").open("w") as output_file:
        json.dump(report, output_file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
