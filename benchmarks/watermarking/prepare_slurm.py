# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare headless SLURM jobs for the watermark evaluation matrix."""

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

from benchmarks.watermarking.run_matrix import load_config

_HARNESS_DIR = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _HARNESS_DIR.parents[1]


def generation_units(config: dict[str, Any]) -> list[tuple[str, int | None]]:
    units = []
    for variant in config["variants"]:
        keys = config["keys"] if variant["watermarked"] else [None]
        units.extend((variant["name"], key) for key in keys)
    return units


def directive(name: str, value: str | int) -> str:
    return f"#SBATCH --{name}={value}"


def common_header(
    *,
    job_name: str,
    log_path: Path,
    account: str,
    time_limit: str,
    cpus: int,
    memory: str,
    gpus: int,
    partition: str | None,
) -> list[str]:
    lines = [
        "#!/usr/bin/env bash",
        directive("job-name", job_name),
        directive("account", account),
        directive("nodes", 1),
        directive("ntasks", 1),
        directive("cpus-per-task", cpus),
        directive("mem", memory),
        directive("time", time_limit),
        directive("output", log_path),
    ]
    if gpus:
        lines.append(directive("gpus", gpus))
    if partition:
        lines.append(directive("partition", partition))
    lines.extend(
        [
            "",
            "source /etc/shell-config/shell-config.sh",
            "set -euo pipefail",
        ]
    )
    return lines


def run_matrix_command(
    python: Path, config: Path, phase: str, extra: list[str] | None = None
) -> str:
    command = [
        str(python),
        str(_HARNESS_DIR / "run_matrix.py"),
        "--config",
        str(config),
        "--execute",
        *(extra or []),
        phase,
    ]
    return shlex.join(command)


def write_script(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def prepare(args: argparse.Namespace) -> Path:
    config_path = args.config.resolve()
    config = load_config(config_path)
    results_dir = Path(config["results_dir"])
    if not results_dir.is_absolute():
        results_dir = (config_path.parent / results_dir).resolve()
    slurm_dir = results_dir / "slurm"
    logs_dir = slurm_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    python = (args.python or Path(sys.executable)).resolve()
    performance_path = slurm_dir / "performance.sbatch"
    generation_path = slurm_dir / "generation.sbatch"
    analysis_path = slurm_dir / "analysis.sbatch"

    performance_lines = common_header(
        job_name="wm-performance",
        log_path=logs_dir / "performance-%j.log",
        account=args.account,
        time_limit=args.gpu_time,
        cpus=args.cpus,
        memory=args.memory,
        gpus=1,
        partition=args.partition,
    )
    performance_lines.extend(
        [
            f"cd {shlex.quote(str(_REPOSITORY_ROOT))}",
            run_matrix_command(python, config_path, "performance"),
        ]
    )
    write_script(performance_path, performance_lines)

    generation_lines = common_header(
        job_name="wm-generation",
        log_path=logs_dir / "generation-%A_%a.log",
        account=args.account,
        time_limit=args.gpu_time,
        cpus=args.cpus,
        memory=args.memory,
        gpus=1,
        partition=args.partition,
    )
    generation_lines.extend(
        [
            f"cd {shlex.quote(str(_REPOSITORY_ROOT))}",
            'case "$SLURM_ARRAY_TASK_ID" in',
        ]
    )
    for index, (variant, key) in enumerate(generation_units(config)):
        extra = ["--variant", variant]
        if key is not None:
            extra.extend(["--key", str(key)])
        command = run_matrix_command(python, config_path, "generation", extra)
        generation_lines.append(f"  {index}) {command} ;;")
    generation_lines.extend(
        [
            '  *) echo "Unknown array index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;',
            "esac",
        ]
    )
    write_script(generation_path, generation_lines)

    analysis_lines = common_header(
        job_name="wm-analysis",
        log_path=logs_dir / "analysis-%j.log",
        account=args.account,
        time_limit=args.analysis_time,
        cpus=args.cpus,
        memory=args.memory,
        gpus=0,
        partition=args.partition,
    )
    analysis_lines.extend(
        [
            f"cd {shlex.quote(str(_REPOSITORY_ROOT))}",
            shlex.join(
                [
                    str(python),
                    str(_HARNESS_DIR / "analyze.py"),
                    "--config",
                    str(config_path),
                ]
            ),
        ]
    )
    write_script(analysis_path, analysis_lines)

    units = generation_units(config)
    submit_path = slurm_dir / "submit.sh"
    performance_script = shlex.quote(str(performance_path))
    generation_script = shlex.quote(str(generation_path))
    analysis_script = shlex.quote(str(analysis_path))
    array = f"0-{len(units) - 1}%{args.max_parallel}"
    write_script(
        submit_path,
        [
            "#!/usr/bin/env bash",
            "source /etc/shell-config/shell-config.sh",
            "set -euo pipefail",
            f"performance_submission=$(sbatch --parsable {performance_script})",
            (
                "generation_submission=$(sbatch --parsable "
                f"--array={array} {generation_script})"
            ),
            "performance_job=${performance_submission%%;*}",
            "generation_job=${generation_submission%%;*}",
            (
                "analysis_submission=$(sbatch --parsable "
                "--dependency=afterok:$performance_job:$generation_job "
                f"{analysis_script})"
            ),
            "analysis_job=${analysis_submission%%;*}",
            (
                'echo "performance=$performance_job generation=$generation_job '
                'analysis=$analysis_job"'
            ),
        ],
    )
    return submit_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--account", default="runtime")
    parser.add_argument("--partition")
    parser.add_argument("--max-parallel", default=4, type=int)
    parser.add_argument("--cpus", default=8, type=int)
    parser.add_argument("--memory", default="64G")
    parser.add_argument("--gpu-time", default="06:00:00")
    parser.add_argument("--analysis-time", default="01:00:00")
    return parser.parse_args()


def main() -> None:
    submit_path = prepare(parse_args())
    print(f"Prepared {submit_path}")
    print(f"Review it, then run: {submit_path}")


if __name__ == "__main__":
    main()
