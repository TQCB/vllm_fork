# Watermark evaluation

This harness compares three builds:

1. The parent of the watermarking change.
2. The watermarking change with watermarking disabled.
3. The watermarking change with Gumbel watermarking enabled.

It uses vLLM's existing latency and throughput benchmarks for performance and
an OpenAI-compatible server for output collection. No command runs unless
`run_matrix.py` receives `--execute`.

## Setup

Use this stacked evaluation worktree for the disabled and enabled variants, and
create one independent worktree for the parent commit.

```bash
git worktree add ../vllm-watermark-parent HEAD^

cd ../vllm-watermark-parent
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

cd ../vllm-watermark-evals
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

Copy `config.example.json`, then replace the command and working-directory
placeholders with the paths on the benchmark machine. Both PR variants should
point to this evaluation worktree. Keep every other engine and sampling setting
identical across variants.

## Inspect and run

Print the complete command matrix without starting a model:

```bash
.venv/bin/python benchmarks/watermarking/run_matrix.py \
  --config watermark-eval.json all
```

Run it only after reviewing the commands:

```bash
.venv/bin/python benchmarks/watermarking/run_matrix.py \
  --config watermark-eval.json --execute all
```

Individual phases are `performance` and `generation`. Generation starts one
server at a time, waits for `/health`, collects outputs, and stops the server.
The enabled variant is restarted for every key because the key is an engine
configuration. Use `generation_server_args` for engine settings that should
only apply while collecting outputs, such as a concurrency cap.

Use `--variant NAME` to run one variant and `--key KEY` to run one watermarked
generation key. These selectors make independent server configurations safe to
schedule on separate GPUs.

## Headless SLURM execution

After creating a cluster-specific config with absolute worktree, executable,
and result paths, prepare the batch scripts without submitting them:

```bash
.venv/bin/python benchmarks/watermarking/prepare_slurm.py \
  --config watermark-eval.json \
  --max-parallel 4
```

This creates three jobs under `RESULTS/slurm`: one GPU job that runs all
performance variants sequentially on the same device, a one-GPU-per-task
generation array, and a CPU analysis job. `submit.sh` gives analysis an
`afterok` dependency on both GPU jobs. Keeping the performance comparison on a
single device avoids confounding a small sampler overhead with device or node
variation.

Review the generated scripts and resource requests before running:

```bash
RESULTS/slurm/submit.sh
```

With the example matrix, generation has ten tasks: the two unwatermarked
variants and eight watermarked keys. `--max-parallel` caps simultaneous
generation GPUs; the performance job may occupy one additional GPU.

The first configured watermark key receives all sampling seeds for fixed-key
diversity. Each remaining key receives its index-matched seed for key-averaged
quality, avoiding a full samples-by-keys Cartesian product. Increase
`fixed_key_diversity_keys` to measure fixed-key diversity variability.

## Analysis

```bash
.venv/bin/python benchmarks/watermarking/analyze.py \
  --config watermark-eval.json
```

The report includes GSM8K accuracy, maj@k, lexical diversity, within-sequence
repetition, context recurrence, entropy buckets, and Gumbel detector scores.
Key-averaged quality uses one matched sample per key; fixed-key diversity uses
all samples generated under each key.

Entropy buckets use the disabled PR's generation paths as the common reference.
Entropy is coarsened from the requested top log-probabilities by treating the
unreported probability mass as one outcome; it is not full-vocabulary entropy.

HumanEval outputs are exported in the format consumed by OpenAI's
`human-eval` package. Executing generated code is deliberately separate from
this harness:

```bash
uv pip install human-eval
for samples in RESULTS/analysis/humaneval/*.jsonl; do
  evaluate_functional_correctness "$samples"
done
```

Review the generated programs and run code execution only in an appropriately
isolated environment.

## Interpretation

- The parent-versus-disabled comparison detects integration regressions.
- The disabled-versus-enabled comparison measures Philox/Gumbel overhead.
- Unwatermarked repetitions vary the request seed; watermarked repetitions
  use a full seed sweep for selected fixed keys and matched seeds elsewhere.
- Report the distribution across fixed keys as well as the pooled key average.
- Temperature-zero evaluations do not exercise watermarking and are excluded.
