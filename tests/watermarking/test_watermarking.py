# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking import create_watermarker
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.watermarking.watermarker import WatermarkSample
from vllm.v1.worker.gpu.sample.sampler import Sampler


@pytest.mark.parametrize("algorithm", ["gumbel"])
def test_watermarker_contract(algorithm: str):
    watermarker = create_watermarker(
        WatermarkConfig(algorithm=algorithm, key=42, context_width=4)
    )
    logits = torch.zeros(2, 128)
    contexts = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    random_sample = lambda sample_logits: sample_logits.argmax(dim=-1)

    first = watermarker.sample(logits, contexts, random_sample)
    second = watermarker.sample(logits, contexts, random_sample)

    assert first.token_ids.shape == (2,)
    assert first.logits.shape == logits.shape
    assert torch.equal(first.token_ids, second.token_ids)


def test_gumbel_config_warns_about_degenerate_generations(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(
        "vllm.config.watermarking.logger.warning_once",
        lambda message, *, scope: messages.append(message),
    )

    WatermarkConfig(key=42)

    assert messages == [
        "Single-key Gumbel-max watermarking may increase the frequency of "
        "degenerate generations, including repetition loops."
    ]


def test_large_context_width_warns_but_is_allowed():
    config = WatermarkConfig(key=42, context_width=17)

    with pytest.warns(UserWarning, match="reduce robustness to edits"):
        watermarker = create_watermarker(config)

    assert watermarker.context_width == 17


def test_sampling_params_can_disable_watermarking():
    assert SamplingParams().watermarking
    assert not SamplingParams.from_optional(watermarking=False).watermarking


def test_gpu_sampler_rejects_watermarking_for_greedy(monkeypatch):
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarking = SimpleNamespace(np=np.ones(1, dtype=bool))
    monkeypatch.setattr(Sampler, "add_request", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="Greedy decoding cannot be used"):
        sampler.add_request(0, 1, SamplingParams(temperature=0))

    sampler.add_request(0, 1, SamplingParams(temperature=1))
    sampler.add_request(0, 1, SamplingParams(temperature=0, watermarking=False))


def test_gpu_sampler_respects_mixed_request_watermarking(monkeypatch):
    class StubWatermarker:
        context_width = 1

        def sample(self, logits, contexts, random_sample):
            return WatermarkSample(torch.tensor([7, 7]), logits + 10)

    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.watermarking = SimpleNamespace(
        np=np.array([False, True, False, False]),
        gpu=torch.tensor([False, True, False, False]),
    )
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.ones(4), gpu=torch.ones(4)),
        seeds=SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int64)),
    )
    sampler.use_fp64_gumbel = False
    sampler._get_contexts = lambda expanded_idx_mapping: torch.zeros(
        2, 1, dtype=torch.int64
    )
    monkeypatch.setattr(
        "vllm.v1.watermarking.gpu_sampler.gumbel_sample",
        lambda *args, **kwargs: torch.tensor([3, 4]),
    )
    logits = torch.zeros(2, 8)

    sampled, output_logits = sampler._sample_random(
        logits,
        torch.tensor([3, 1]),
        np.array([3, 1]),
        torch.zeros(2, dtype=torch.int64),
        None,
        None,
        False,
    )

    assert torch.equal(sampled, torch.tensor([3, 7]))
    assert torch.equal(output_logits[0], logits[0])
    assert torch.equal(output_logits[1], torch.full((8,), 10.0))


def test_gpu_sampler_filters_top_k_top_p_before_watermarking(monkeypatch):
    captured_logits = None

    class StubWatermarker:
        context_width = 1

        def sample(self, logits, contexts, random_sample):
            nonlocal captured_logits
            captured_logits = logits
            return WatermarkSample(logits.argmax(dim=-1), logits)

    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.watermarking = SimpleNamespace(
        np=np.array([True]), gpu=torch.tensor([True])
    )
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.ones(1), gpu=torch.ones(1)),
        seeds=SimpleNamespace(gpu=torch.zeros(1, dtype=torch.int64)),
    )
    sampler.use_fp64_gumbel = False
    sampler._get_contexts = lambda expanded_idx_mapping: torch.zeros(
        1, 1, dtype=torch.int64
    )
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0]])

    sampled, _ = sampler._sample_random(
        logits,
        torch.tensor([0]),
        np.array([0]),
        torch.zeros(1, dtype=torch.int64),
        torch.tensor([2]),
        torch.tensor([0.8]),
        False,
    )

    assert captured_logits is not None
    assert torch.isneginf(captured_logits[0, 2:]).all()
    assert sampled.item() in (0, 1)


def test_watermark_context_ignores_prefix_cache_bookkeeping():
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = SimpleNamespace(context_width=3)
    sampler.req_states = SimpleNamespace(
        total_len=SimpleNamespace(gpu=torch.tensor([4, 0, 5])),
        prompt_len=SimpleNamespace(gpu=torch.tensor([3, 0, 2])),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor(
                [
                    [10, 11, 12, 40, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [20, 21, 50, 51, 52, 0],
                ]
            )
        ),
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([0, 0, 0])),
    )
    request_slots = torch.tensor([2, 0])

    cold_contexts = sampler._get_contexts(request_slots)
    sampler.req_states.num_computed_tokens.gpu[:] = torch.tensor([3, 0, 2])
    cached_contexts = sampler._get_contexts(request_slots)

    expected = torch.tensor([[50, 51, 52], [-1, -1, 40]])
    assert torch.equal(cold_contexts, expected)
    assert torch.equal(cached_contexts, expected)


def test_gpu_sampler_skips_watermarking_for_disabled_greedy_batch(monkeypatch):
    class StubWatermarker:
        context_width = 1

        def sample(self, logits, contexts, random_sample):
            raise AssertionError("watermarker should not run for greedy requests")

    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.watermarking = SimpleNamespace(
        np=np.array([False, False]), gpu=torch.tensor([False, False])
    )
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.zeros(2), gpu=torch.zeros(2)),
    )
    expected = (torch.tensor([3, 4]), torch.zeros(2, 8))
    monkeypatch.setattr(
        "vllm.v1.watermarking.gpu_sampler.Sampler._sample_random",
        lambda *args, **kwargs: expected,
    )

    actual = sampler._sample_random(
        torch.zeros(2, 8),
        torch.tensor([0, 1]),
        np.array([0, 1]),
        torch.zeros(2, dtype=torch.int64),
        None,
        None,
        False,
    )

    assert actual is expected
