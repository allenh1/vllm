# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepseekV4 rotary embedding initialization."""

import torch

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


def build_deepseek_v4_rope(
    config,
    *,
    head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    compress_ratio: int,
    use_unscaled_rope: bool = False,
    use_compress_rope_theta: bool | None = None,
) -> RotaryEmbedding:
    # Copy so per-layer overrides cannot leak into the shared hf_config dict.
    rope_parameters = dict(config.rope_parameters)
    if use_compress_rope_theta is None:
        # V4's rule: only a real pooling ratio moves off the base theta. That
        # is the wrong test for DeepSeek-V4.1, where a raw ratio of 1 means
        # "one latent per token" and *does* use the compressed theta, while a
        # raw 0 means no compression at all -- so V4.1 callers pass the flag
        # explicitly, derived from the raw entry rather than the operational
        # ratio (which is clamped to >= 1 and so cannot tell them apart).
        use_compress_rope_theta = compress_ratio > 1
    rope_parameters["rope_theta"] = (
        config.compress_rope_theta if use_compress_rope_theta else config.rope_theta
    )
    if use_unscaled_rope:
        # The MTP draft layer of DSpark-style checkpoints (compress_ratios
        # entry 0) is trained with plain rope, not the yarn-scaled variant.
        rope_parameters["rope_type"] = "default"
    if rope_parameters["rope_type"] != "default":
        rope_parameters["rope_type"] = (
            "deepseek_yarn"
            if rope_parameters.get("apply_yarn_scaling", True)
            else "deepseek_llama_scaling"
        )
    rope_parameters["mscale"] = 0  # Disable mscale
    rope_parameters["mscale_all_dim"] = 0  # Disable mscale
    rope_parameters["is_deepseek_v4"] = True
    rope_parameters["rope_dim"] = rope_head_dim
    return get_rope(
        head_dim,
        max_position=max_position_embeddings,
        rope_parameters=rope_parameters,
        is_neox_style=False,
        # DeepSeek V4 kernels consume the cached cos/sin table directly and
        # require FP32 even when the draft/MTP model is initialized under a
        # lower default dtype.
        dtype=torch.float32,
    )
