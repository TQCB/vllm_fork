# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Collect stochastic completions from an OpenAI-compatible vLLM server."""

import argparse
import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm

from tests.evals.gsm8k.gsm8k_eval import _build_gsm8k_prompts


@dataclass(frozen=True)
class EvaluationPrompt:
    task_id: str
    prompt: str
    reference: int | None
    stop: list[str]


def load_prompts(
    dataset: str, num_prompts: int, num_fewshot: int
) -> list[EvaluationPrompt]:
    if dataset == "gsm8k":
        prompts, labels = _build_gsm8k_prompts(num_prompts, num_fewshot)
        return [
            EvaluationPrompt(
                task_id=f"gsm8k/{index}",
                prompt=prompt,
                reference=label,
                stop=["Question", "Assistant:", "<|separator|>"],
            )
            for index, (prompt, label) in enumerate(zip(prompts, labels, strict=True))
        ]
    if dataset == "humaneval":
        from datasets import load_dataset

        rows = load_dataset("openai/openai_humaneval", split="test")
        return [
            EvaluationPrompt(
                task_id=row["task_id"],
                prompt=row["prompt"],
                reference=None,
                stop=["\nclass", "\nif __name__", "\nprint"],
            )
            for row in rows.select(range(min(num_prompts, len(rows))))
        ]
    raise ValueError(f"Unsupported dataset: {dataset}")


def coarsened_entropy(top_logprobs: list[dict[str, float]] | None) -> float | None:
    if not top_logprobs:
        return None
    entropies = []
    for token_logprobs in top_logprobs:
        probabilities = [math.exp(value) for value in token_logprobs.values()]
        covered_probability = min(1.0, sum(probabilities))
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0
        )
        residual = 1.0 - covered_probability
        if residual > 0:
            entropy -= residual * math.log(residual)
        entropies.append(entropy)
    return sum(entropies) / len(entropies)


async def collect(
    args: argparse.Namespace,
    completed: set[tuple[str, int]],
    output_file: Any,
) -> None:
    prompts = load_prompts(args.dataset, args.num_prompts, args.num_fewshot)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_seconds)
    base_url = f"http://{args.host}:{args.port}"

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def generate_one(
            prompt: EvaluationPrompt, sample_index: int, seed: int
        ) -> dict[str, Any]:
            payload = {
                "model": args.model,
                "prompt": prompt.prompt,
                "temperature": args.temperature,
                "top_p": 1.0,
                "max_tokens": args.max_tokens,
                "stop": prompt.stop,
                "seed": seed,
                "logprobs": args.top_logprobs,
            }
            for attempt in range(args.request_retries + 1):
                try:
                    async with (
                        semaphore,
                        session.post(
                            f"{base_url}/v1/completions", json=payload
                        ) as response,
                    ):
                        response.raise_for_status()
                        body = await response.json()
                    break
                except (aiohttp.ClientError, TimeoutError):
                    if attempt == args.request_retries:
                        raise
                    await asyncio.sleep(2**attempt)
            choice = body["choices"][0]
            logprobs = choice.get("logprobs") or {}
            usage = body.get("usage") or {}
            return {
                "variant": args.variant,
                "dataset": args.dataset,
                "task_id": prompt.task_id,
                "reference": prompt.reference,
                "key": args.key,
                "sample_index": sample_index,
                "seed": seed,
                "temperature": args.temperature,
                "completion": choice["text"],
                "completion_tokens": usage.get("completion_tokens"),
                "mean_coarsened_entropy": coarsened_entropy(
                    logprobs.get("top_logprobs")
                ),
            }

        tasks = [
            generate_one(prompt, sample_index, seed)
            for prompt in prompts
            for sample_index, seed in enumerate(args.seeds)
            if (prompt.task_id, seed) not in completed
        ]
        for future in tqdm.as_completed(
            tasks, desc=f"{args.dataset} T={args.temperature}"
        ):
            record = await future
            output_file.write(json.dumps(record) + "\n")
            output_file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "humaneval"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--key", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--request-timeout-seconds", type=float, default=600)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, int]] = set()
    if args.resume and args.output.exists():
        with args.output.open() as input_file:
            completed = {
                (record["task_id"], record["seed"])
                for record in map(json.loads, input_file)
            }
    mode = "a" if args.resume else "w"
    with args.output.open(mode) as output_file:
        asyncio.run(collect(args, completed, output_file))


if __name__ == "__main__":
    main()
