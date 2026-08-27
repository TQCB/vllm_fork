# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Orchestrate watermark performance and generation experiments."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_HARNESS_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as config_file:
        config = json.load(config_file)
    required = {"model", "variants", "keys", "sampling_seeds"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    return config


def watermark_config(config: dict[str, Any], key: int) -> str:
    value = dict(config["watermark"])
    value["key"] = key
    return json.dumps(value, separators=(",", ":"))


def variant_command(variant: dict[str, Any]) -> list[str]:
    command = variant["command"]
    if not isinstance(command, list) or not command:
        raise ValueError(f"Invalid command for variant {variant['name']}")
    return [str(part) for part in command]


def print_command(
    command: Iterable[str],
    cwd: str | None = None,
    environment: dict[str, str] | None = None,
) -> None:
    environment_prefix = ""
    if environment:
        assignments = (f"{key}={value}" for key, value in environment.items())
        environment_prefix = f"env {shlex.join(assignments)} "
    prefix = f"(cd {shlex.quote(cwd)} && " if cwd else ""
    suffix = ")" if cwd else ""
    print(f"{prefix}{environment_prefix}{shlex.join(command)}{suffix}")


def run_command(
    command: list[str],
    cwd: str | None,
    environment: dict[str, str] | None,
    execute: bool,
) -> subprocess.CompletedProcess[str] | None:
    print_command(command, cwd, environment)
    if not execute:
        return None
    process_environment = os.environ.copy()
    process_environment.update(environment or {})
    return subprocess.run(
        command,
        cwd=cwd,
        env=process_environment,
        check=True,
        text=True,
    )


def performance_commands(
    config: dict[str, Any], variant: dict[str, Any], results_dir: Path
) -> Iterable[list[str]]:
    settings = config["performance"]
    base = variant_command(variant)
    engine_args = [f"--model={config['model']}", *config.get("server_args", [])]
    if variant["watermarked"]:
        engine_args.append(
            f"--watermark-config={watermark_config(config, config['keys'][0])}"
        )

    for repetition in range(settings["repetitions"]):
        for batch_size in settings["batch_sizes"]:
            output = results_dir / (f"latency_bs{batch_size}_repeat{repetition}.json")
            yield [
                *base,
                "bench",
                "latency",
                *engine_args,
                f"--input-len={settings['input_length']}",
                f"--output-len={settings['output_length']}",
                f"--batch-size={batch_size}",
                f"--num-iters-warmup={settings['latency_warmups']}",
                f"--num-iters={settings['latency_iterations']}",
                "--disable-detokenize",
                f"--output-json={output}",
            ]

        output = results_dir / f"throughput_repeat{repetition}.json"
        yield [
            *base,
            "bench",
            "throughput",
            *engine_args,
            "--backend=vllm",
            "--dataset-name=random",
            f"--input-len={settings['input_length']}",
            f"--output-len={settings['output_length']}",
            f"--num-prompts={settings['throughput_prompts']}",
            "--seed=0",
            "--disable-detokenize",
            f"--output-json={output}",
        ]


def selected_variants(
    config: dict[str, Any], names: list[str] | None
) -> list[dict[str, Any]]:
    variants = config["variants"]
    if not names:
        return variants
    by_name = {variant["name"]: variant for variant in variants}
    unknown = set(names).difference(by_name)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    return [by_name[name] for name in names]


def run_performance(
    config: dict[str, Any], execute: bool, variants: list[dict[str, Any]]
) -> None:
    root = Path(config["results_dir"]) / "performance"
    for variant in variants:
        results_dir = root / variant["name"]
        if execute:
            results_dir.mkdir(parents=True, exist_ok=True)
        for command in performance_commands(config, variant, results_dir):
            run_command(
                command,
                variant.get("cwd"),
                variant.get("env"),
                execute,
            )


def server_command(
    config: dict[str, Any], variant: dict[str, Any], key: int | None
) -> list[str]:
    command = [
        *variant_command(variant),
        "serve",
        config["model"],
        f"--host={config.get('host', '127.0.0.1')}",
        f"--port={config.get('port', 8000)}",
        *config.get("server_args", []),
    ]
    if key is not None:
        command.append(f"--watermark-config={watermark_config(config, key)}")
    return command


def generation_commands(
    config: dict[str, Any], variant: dict[str, Any], key: int | None
) -> Iterable[list[str]]:
    settings = config["generation"]
    if key is None:
        seeds = config["sampling_seeds"]
    else:
        key_index = config["keys"].index(key)
        if key_index < settings["fixed_key_diversity_keys"]:
            seeds = config["sampling_seeds"]
        else:
            seeds = [config["sampling_seeds"][key_index]]
    key_name = f"key_{key}" if key is not None else "unwatermarked"
    output_root = Path(config["results_dir"]) / "generation" / variant["name"]
    for dataset in settings["datasets"]:
        for temperature in settings["temperatures"]:
            output = output_root / (
                f"{dataset}_{key_name}_temperature_{temperature}.jsonl"
            )
            command = [
                sys.executable,
                str(_HARNESS_DIR / "generate.py"),
                f"--model={config['model']}",
                f"--variant={variant['name']}",
                f"--dataset={dataset}",
                f"--output={output}",
                f"--host={config.get('host', '127.0.0.1')}",
                f"--port={config.get('port', 8000)}",
                f"--num-prompts={settings['num_prompts'][dataset]}",
                f"--num-fewshot={settings['num_fewshot']}",
                f"--temperature={temperature}",
                f"--max-tokens={settings['max_tokens']}",
                f"--max-concurrency={settings['max_concurrency']}",
                f"--request-timeout-seconds={settings['request_timeout_seconds']}",
                f"--request-retries={settings.get('request_retries', 3)}",
                f"--top-logprobs={settings['top_logprobs']}",
                "--resume",
                "--seeds",
                *[str(seed) for seed in seeds],
            ]
            if key is not None:
                command.append(f"--key={key}")
            yield command


def wait_for_server(
    process: subprocess.Popen[str], host: str, port: int, timeout_seconds: int = 600
) -> None:
    deadline = time.monotonic() + timeout_seconds
    health_url = f"http://{host}:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=2):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"Server did not become healthy within {timeout_seconds}s")


def run_generation(
    config: dict[str, Any],
    execute: bool,
    variants: list[dict[str, Any]],
    key: int | None,
) -> None:
    results_root = Path(config["results_dir"])
    for variant in variants:
        keys = config["keys"] if variant["watermarked"] else [None]
        if key is not None:
            if not variant["watermarked"]:
                raise ValueError("--key requires a watermarked variant")
            if key not in keys:
                raise ValueError(f"Unknown watermark key: {key}")
            keys = [key]
        for key in keys:
            serve = server_command(config, variant, key)
            print_command(serve, variant.get("cwd"), variant.get("env"))
            commands = list(generation_commands(config, variant, key))
            for command in commands:
                print_command(command)
            if not execute:
                continue

            results_root.mkdir(parents=True, exist_ok=True)
            log_name = f"{variant['name']}_{key if key is not None else 'none'}.log"
            log_path = results_root / "server_logs" / log_name
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as log_file:
                process = subprocess.Popen(
                    serve,
                    cwd=variant.get("cwd"),
                    env={**os.environ, **variant.get("env", {})},
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                try:
                    wait_for_server(
                        process,
                        config.get("host", "127.0.0.1"),
                        config.get("port", 8000),
                    )
                    for command in commands:
                        subprocess.run(command, check=True)
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--variant", action="append")
    parser.add_argument("--key", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("phase", choices=("performance", "generation", "all"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.port is not None:
        config["port"] = args.port
    variants = selected_variants(config, args.variant)
    if args.key is not None and args.phase != "generation":
        raise ValueError("--key is only valid for the generation phase")
    results_dir = Path(config["results_dir"])
    if not results_dir.is_absolute():
        config["results_dir"] = str((args.config.parent / results_dir).resolve())
    if args.execute:
        results_dir = Path(config["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        config_output = results_dir / "config.json"
        temporary_output = results_dir / f".config.{os.getpid()}.json"
        with temporary_output.open("w") as output_file:
            json.dump(config, output_file, indent=2)
        temporary_output.replace(config_output)
    if args.phase in {"performance", "all"}:
        run_performance(config, args.execute, variants)
    if args.phase in {"generation", "all"}:
        run_generation(config, args.execute, variants, args.key)


if __name__ == "__main__":
    main()
