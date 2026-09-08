# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

from vllm.config.watermarking import WatermarkConfig, derive_watermark_key
from vllm.v1.watermarking.gumbel import GumbelWatermarker
from vllm.v1.watermarking.watermarker import Watermarker


def _create_gumbel(config: WatermarkConfig, key: int) -> Watermarker:
    return GumbelWatermarker(key, config.context_width, config.prf)


_WATERMARKERS: dict[str, Callable[[WatermarkConfig, int], Watermarker]] = {
    "gumbel": _create_gumbel,
    "dual_key_gumbel": _create_gumbel,
}


def get_watermark_key(config: WatermarkConfig, *, is_drafting: bool = False) -> int:
    if config.algorithm == "dual_key_gumbel":
        domain = b"draft" if is_drafting else b"target"
        return derive_watermark_key(config.key, domain)
    return config.key


def create_watermarker(
    config: WatermarkConfig, *, is_drafting: bool = False
) -> Watermarker:
    watermarker_factory = _WATERMARKERS.get(config.algorithm)
    if watermarker_factory is None:
        raise ValueError(f"Unknown watermarking algorithm: {config.algorithm}")
    return watermarker_factory(
        config, get_watermark_key(config, is_drafting=is_drafting)
    )
