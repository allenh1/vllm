# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DeepSeek V4 per-layer rope selection.

Covers the MTP draft layer's compress_ratios handling: checkpoints that
include their draft layer in compress_ratios with an entry of 0 get an
uncompressed-KV, plain-rope draft; the operational compress ratio is never
0 (KV-cache specs treat 1 as "no compression" and divide by the ratio).
"""

import types

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT, RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
    DeepseekV4ScalingRotaryEmbedding,
)
from vllm.models.deepseek_v4.attention import resolve_layer_compress_ratio
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope

NUM_HIDDEN_LAYERS = 4


@pytest.fixture(autouse=True)
def _clear_rope_cache():
    _ROPE_DICT.clear()
    yield
    _ROPE_DICT.clear()


def _config(compress_ratios: list[int]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        compress_ratios=compress_ratios,
        rope_theta=10000.0,
        compress_rope_theta=1000000.0,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 512,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    )


@pytest.mark.parametrize(
    "compress_ratios,layer_id,expected_ratio,expected_unscaled",
    [
        # Main layers: clamped, never unscaled (unchanged upstream behavior).
        ([1, 4, 4, 1], 0, 1, False),
        ([1, 4, 4, 1], 1, 4, False),
        ([0, 4, 4, 1], 0, 1, False),
        # Draft layer present in the list with 0: operational ratio 1,
        # unscaled rope selected.
        ([1, 4, 4, 1, 0], NUM_HIDDEN_LAYERS, 1, True),
        # Draft layer present with an explicit nonzero entry: honored.
        ([1, 4, 4, 1, 4], NUM_HIDDEN_LAYERS, 4, False),
        # Draft layer absent from the list: legacy fallback (ratio 1, yarn).
        ([1, 4, 4, 1], NUM_HIDDEN_LAYERS, 1, False),
        ([1, 4, 4, 1], NUM_HIDDEN_LAYERS + 1, 1, False),
    ],
)
def test_resolve_layer_compress_ratio(
    compress_ratios: list[int],
    layer_id: int,
    expected_ratio: int,
    expected_unscaled: bool,
):
    ratio, unscaled = resolve_layer_compress_ratio(_config(compress_ratios), layer_id)
    assert ratio == expected_ratio
    assert unscaled == expected_unscaled
    # The operational ratio is an invariant: never below 1.
    assert ratio >= 1


def test_unscaled_rope_selects_plain_rotary_embedding(default_vllm_config):
    config = _config([1, 4, 4, 1, 0])
    rope = build_deepseek_v4_rope(
        config,
        head_dim=64,
        rope_head_dim=64,
        max_position_embeddings=config.max_position_embeddings,
        compress_ratio=1,
        use_unscaled_rope=True,
    )
    assert type(rope) is RotaryEmbedding
    assert not isinstance(rope, DeepseekScalingRotaryEmbedding)


def test_scaled_rope_unchanged_without_flag(default_vllm_config):
    config = _config([1, 4, 4, 1])
    rope = build_deepseek_v4_rope(
        config,
        head_dim=64,
        rope_head_dim=64,
        max_position_embeddings=config.max_position_embeddings,
        compress_ratio=1,
    )
    # The V4 subclass specifically: losing is_deepseek_v4 would silently
    # downgrade to the base deepseek-yarn implementation.
    assert isinstance(rope, DeepseekV4ScalingRotaryEmbedding)


# ---------------------------------------------------------------------------
# DeepSeek-V4.1
# ---------------------------------------------------------------------------
#
# V4.1 keeps V4's classes but reads `compress_ratios` differently and names the
# layers that own the shared compressed cache. `_config` deliberately has none
# of the V4.1 keys, so every test above is also the regression check that a V4
# checkpoint still resolves exactly as it did.


def _v41_config() -> types.SimpleNamespace:
    """The real DeepSeek-V4.1-Flash text config, minus the fields we don't read."""
    config = _config([0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0])
    config.num_hidden_layers = 40
    config.kv_source_layer_ids = [2, 8, 14, 20]
    config.index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
    return config


def test_v4_configs_have_no_v41_roles():
    # The gate: everything V4.1-specific is off unless the config names sources.
    from vllm.models.deepseek_v4.attention import v41_layer_roles

    assert v41_layer_roles(_config([1, 4, 4, 1]), 1) is None


def test_v41_compression_is_keyed_on_the_raw_entry():
    from vllm.models.deepseek_v4.attention import v41_layer_roles

    config = _v41_config()
    # raw 0 (layers 0, 1) is plain sliding window -- and, unlike V4, it sits
    # inside the backbone rather than only on the draft layer.
    for layer_id in (0, 1):
        roles = v41_layer_roles(config, layer_id)
        assert not roles.compression_enabled
        assert roles.compress_ratio == 1  # spec-safe, but unused
        assert roles.use_unscaled_rope
        assert not roles.owns_compressed_kv
        assert not roles.runs_indexer
    # raw 1 (layer 20) compresses at ratio 1, and uses the compressed theta.
    roles = v41_layer_roles(config, 20)
    assert roles.compression_enabled
    assert roles.compress_ratio == 1
    assert not roles.use_unscaled_rope
    # raw 2 pools pairs.
    roles = v41_layer_roles(config, 2)
    assert roles.compress_ratio == 2
    assert not roles.use_unscaled_rope


def test_v41_only_the_configured_layers_own_each_cache():
    from vllm.models.deepseek_v4.attention import v41_layer_roles

    config = _v41_config()
    kv_owners = [i for i in range(40) if v41_layer_roles(config, i).owns_compressed_kv]
    assert kv_owners == [2, 8, 14, 20]
    indexers = [i for i in range(40) if v41_layer_roles(config, i).runs_indexer]
    assert indexers == [2, 8, 14, 20, 24, 28, 32, 36]


@pytest.mark.parametrize(
    "layer_id,expected_owner",
    [
        (0, None),  # raw 0, shares nothing
        (2, None),  # owns
        (3, 2),
        (7, 2),
        (8, None),
        (13, 8),
        (14, None),
        (19, 14),
        (20, None),
        (21, 20),
        (24, 20),  # index-only source: own indexer, layer 20's compressed KV
        (36, 20),
        (39, 20),
    ],
)
def test_v41_each_layer_reads_the_most_recent_source(layer_id: int, expected_owner):
    from vllm.models.deepseek_v4.attention import v41_layer_roles

    assert v41_layer_roles(_v41_config(), layer_id).kv_owner_layer == expected_owner


def test_v41_draft_layers_are_uncompressed():
    from vllm.models.deepseek_v4.attention import v41_layer_roles

    # compress_ratios[40:] is [0, 0, 0] and the DSpark layers share nothing.
    for layer_id in (40, 41, 42, 43):
        roles = v41_layer_roles(_v41_config(), layer_id)
        assert not roles.compression_enabled
        assert not roles.owns_compressed_kv
        assert roles.kv_owner_layer is None


def test_v41_owner_prefix_rewrites_only_the_layer_index():
    from vllm.models.deepseek_v4.attention import _v41_owner_prefix

    assert _v41_owner_prefix("model.layers.7.attn", 2) == "model.layers.2.attn"
    assert _v41_owner_prefix("layers.39.attn", 20) == "layers.20.attn"
    with pytest.raises(ValueError):
        _v41_owner_prefix("mtp.0.attn", 2)


def test_v41_ratio_one_uses_the_compressed_rope_theta(default_vllm_config):
    # The bug this guards: `compress_rope_theta if compress_ratio > 1` picks the
    # base theta for a raw ratio of 1, but V4.1 trained layer 20 on the
    # compressed one. The flag is what tells the two apart, since the
    # operational ratio is clamped to 1 for the uncompressed layers as well.
    config = _v41_config()

    def cache(*, use_compress_rope_theta: bool | None, compress_ratio: int = 1):
        _ROPE_DICT.clear()  # the builder caches by parameters; force a rebuild
        return build_deepseek_v4_rope(
            config,
            head_dim=64,
            rope_head_dim=64,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=compress_ratio,
            use_compress_rope_theta=use_compress_rope_theta,
        ).cos_sin_cache.clone()

    compressed = cache(use_compress_rope_theta=True)
    base = cache(use_compress_rope_theta=False)
    assert not torch.equal(compressed, base)
    # Omitted, the flag falls back to V4's `compress_ratio > 1` rule, which for
    # an operational ratio of 1 means the base theta.
    assert torch.equal(cache(use_compress_rope_theta=None), base)
    assert torch.equal(cache(use_compress_rope_theta=None, compress_ratio=4), compressed)


def test_attention_init_wires_the_resolver():
    # Guards the production wiring: the resolver and rope builder are unit
    # tested above, but a refactor that reverts DeepseekV4Attention.__init__
    # to inline forced-ratio logic (leaving the helper unused) must fail.
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    assert (
        "resolve_layer_compress_ratio" in DeepseekV4Attention.__init__.__code__.co_names
    )


def test_rope_parameters_dict_not_mutated(default_vllm_config):
    config = _config([1, 4, 4, 1, 0])
    snapshot = dict(config.rope_parameters)
    for compress_ratio, use_unscaled in ((1, True), (1, False), (4, False)):
        build_deepseek_v4_rope(
            config,
            head_dim=64,
            rope_head_dim=64,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=compress_ratio,
            use_unscaled_rope=use_unscaled,
        )
        assert config.rope_parameters == snapshot


def test_default_rope_cos_sin_cache_is_fp32_under_bf16_default(default_vllm_config):
    config = _config([1, 4, 4, 1, 0])
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        rope = build_deepseek_v4_rope(
            config,
            head_dim=64,
            rope_head_dim=64,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=1,
            use_unscaled_rope=True,
        )
    finally:
        torch.set_default_dtype(old_dtype)
    assert rope.cos_sin_cache.dtype == torch.float32
