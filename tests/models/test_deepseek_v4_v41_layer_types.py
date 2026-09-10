# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DeepSeek V4.1's decode-layer classification.

V4 picks its attention path off ``compress_ratio``: 4 means the indexer chose
the compressed positions, 128 means the metadata build did, anything <= 1 means
there is no compressed region at all. V4.1 breaks the first and last of those --
it compresses at raw ratio 1, and it uses the indexer at raw ratios 1 and 2 --
so every one of those comparisons has to be answered from the layer's *role*
instead. These tests pin both readings: that V4's answers are unchanged, and
that V4.1's are what its architecture says they should be.

The classification decides the FlashMLA tile-scheduler plan, the width of the
SM120 decode workspace, and whether a layer reads ``topk_indices_buffer`` or the
pre-computed C128A fields, so getting it wrong is a wrong-shaped kernel launch
rather than an exception.
"""

import types

import pytest

from vllm.models.deepseek_v4.attention import (
    DeepseekV4Attention,
    v41_layer_roles,
    resolve_layer_compress_ratio,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    _LAYER_TYPE_C1A,
    _LAYER_TYPE_C2A,
    _LAYER_TYPE_C4A,
    _LAYER_TYPE_C128A,
    _LAYER_TYPE_SWAONLY,
    _layer_type_for,
    is_deepseek_v41_config,
)

#: The real DeepSeek-V4.1-Flash text config, minus the fields nothing here reads.
V41_COMPRESS_RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]


def _v41_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=list(V41_COMPRESS_RATIOS),
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    )


def _v4_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        num_hidden_layers=4, compress_ratios=[1, 4, 4, 1]
    )


@pytest.mark.parametrize(
    "compress_ratio,expected",
    [
        # V4: unchanged. A raw 1 is "no compressed region", and the comments
        # above `_layer_type_for` call the type swaonly for 0 as well.
        (0, _LAYER_TYPE_SWAONLY),
        (1, _LAYER_TYPE_SWAONLY),
        (4, _LAYER_TYPE_C4A),
        (128, _LAYER_TYPE_C128A),
    ],
)
def test_v4_layer_types_are_unchanged(compress_ratio, expected):
    assert _layer_type_for(compress_ratio) == expected
    # ...and the same numbers still mean swaonly for V4 even when asked with the
    # V4.1 spelling explicitly off.
    assert _layer_type_for(compress_ratio, v41=False) == expected


@pytest.mark.parametrize(
    "compress_ratio,expected",
    [
        (0, _LAYER_TYPE_SWAONLY),
        # The two readings of 1 are opposite, which is why the flag exists.
        (1, _LAYER_TYPE_C1A),
        (2, _LAYER_TYPE_C2A),
    ],
)
def test_v41_layer_types_read_the_raw_entry_as_the_pooling_width(
    compress_ratio, expected
):
    assert _layer_type_for(compress_ratio, v41=True) == expected


@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_v41_rejects_v4_ratios(compress_ratio):
    # V4.1-Flash has no such layers. A checkpoint that claims one is not this
    # architecture, and a wrong tile-scheduler plan is a silent wrong answer,
    # so this fails loudly.
    with pytest.raises(ValueError, match="DeepseekV4.1"):
        _layer_type_for(compress_ratio, v41=True)


@pytest.mark.parametrize("compress_ratio", [2, 3, 8, 256])
def test_v4_rejects_unknown_ratios(compress_ratio):
    with pytest.raises(ValueError, match="DeepseekV4"):
        _layer_type_for(compress_ratio)


def test_the_v41_gate_matches_what_attention_uses():
    # `is_deepseek_v41` in the model package cannot be imported here (it would
    # pull the CUDA op registrations in), so the backend repeats the check.
    # Pin that the two agree on the thing that matters: which configs are V4.1.
    from vllm.models.deepseek_v4.attention import is_deepseek_v41

    assert is_deepseek_v41(_v41_config())
    assert not is_deepseek_v41(_v4_config())
    assert is_deepseek_v41_config(_v41_config())
    assert not is_deepseek_v41_config(_v4_config())


@pytest.mark.parametrize(
    "layer_id,expected_type",
    # Layers 2..19 pool pairs; 20..39 keep one latent per token. The three MTP
    # layers carry a raw 0 like layers 0 and 1.
    [(0, _LAYER_TYPE_SWAONLY), (1, _LAYER_TYPE_SWAONLY), (2, _LAYER_TYPE_C2A),
     (13, _LAYER_TYPE_C2A), (19, _LAYER_TYPE_C2A), (20, _LAYER_TYPE_C1A),
     (39, _LAYER_TYPE_C1A), (40, _LAYER_TYPE_SWAONLY), (42, _LAYER_TYPE_SWAONLY)],
)
def test_the_layer_type_a_real_v41_layer_resolves_to(layer_id, expected_type):
    """End to end through the builder's own route: raw entry -> type name."""
    config = _v41_config()
    # The builder walks the raw list, not the resolved ratio, so a raw 0 is
    # only classified as swaonly because `_layer_type_for` reads 0 as such.
    raw = config.compress_ratios[layer_id]
    assert _layer_type_for(raw, v41=True) == expected_type
    # And the operational ratio the attention module caches agrees on whether
    # the layer compresses.
    roles = v41_layer_roles(config, layer_id)
    assert roles.compression_enabled == (expected_type != _LAYER_TYPE_SWAONLY)


@pytest.mark.parametrize(
    "layer_id,expected_ratio",
    [
        # resolved (compress_ratio, use_unscaled_rope) for every distinct case
        (0, 1), (2, 2), (20, 1), (40, 1),
        # V4's own path for comparison
    ],
)
def test_operational_ratio_is_still_never_zero(layer_id, expected_ratio):
    """KV-cache specs divide by the ratio, so it can never be 0."""
    ratio, _ = resolve_layer_compress_ratio(_v41_config(), layer_id)
    assert ratio == expected_ratio and ratio >= 1
    v4_ratio, _ = resolve_layer_compress_ratio(_v4_config(), 0)
    assert v4_ratio == 1


class _RoleStub(types.SimpleNamespace):
    """A layer's roles, under the real properties that read them.

    `DeepseekV4Attention.compresses` and its siblings only ever touch
    `v41_roles` and `compress_ratio`, so binding the actual property objects to
    a stand-in evaluates the real code without building a CUDA layer -- and
    keeps them descriptors, which `uses_indexer_topk` relies on when it reads
    `self.compresses`.
    """

    compresses = DeepseekV4Attention.compresses
    uses_indexer_topk = DeepseekV4Attention.uses_indexer_topk
    tile_sched_layer_type = DeepseekV4Attention.tile_sched_layer_type


def _classify(config, layer_id):
    ratio, _ = resolve_layer_compress_ratio(config, layer_id)
    stub = _RoleStub(
        compress_ratio=ratio, v41_roles=v41_layer_roles(config, layer_id)
    )
    return stub.compresses, stub.uses_indexer_topk


def test_v41_ratio_one_compresses_and_uses_the_indexer():
    # The two facts the whole dispatch rests on: raw 1 is compressed (so it
    # gets a compressed cache, a workspace, and a non-SWA plan) and it is
    # indexer-driven (so it reads topk_indices_buffer, like V4's C4A).
    compresses, uses_indexer = _classify(_v41_config(), 20)
    assert compresses and uses_indexer


def test_v41_ratio_two_compresses_and_uses_the_indexer():
    compresses, uses_indexer = _classify(_v41_config(), 5)
    assert compresses and uses_indexer


def test_v41_uncompressed_layers_do_neither():
    for layer_id in (0, 1, 40, 41, 42):
        compresses, uses_indexer = _classify(_v41_config(), layer_id)
        assert not compresses and not uses_indexer


def test_v4_answers_are_unchanged():
    config = _v4_config()
    # ratio 1: SWA only, no indexer. ratio 4: both. Same as before the port.
    assert _classify(config, 0) == (False, False)
    assert _classify(config, 1) == (True, True)


def test_v4_c128a_is_compressed_but_not_indexer_driven():
    """The one V4 layer type that is compressed and does *not* use the indexer.

    Its sparse set is fixed when the metadata is built, so it reads the C128A
    fields instead of `topk_indices_buffer` -- the reason `uses_indexer_topk`
    is not simply `compresses`.
    """
    config = types.SimpleNamespace(num_hidden_layers=2, compress_ratios=[128, 4])
    compresses, uses_indexer = _classify(config, 0)
    assert compresses and not uses_indexer


def _tile_type(config, layer_id):
    ratio, _ = resolve_layer_compress_ratio(config, layer_id)
    return _RoleStub(
        compress_ratio=ratio, v41_roles=v41_layer_roles(config, layer_id)
    ).tile_sched_layer_type()


@pytest.mark.parametrize(
    "layer_id,expected_type",
    [
        (0, _LAYER_TYPE_SWAONLY),
        (2, _LAYER_TYPE_C2A),
        (20, _LAYER_TYPE_C1A),
        (42, _LAYER_TYPE_SWAONLY),
    ],
)
def test_tile_scheduler_plan_selected_for_a_v41_layer(layer_id, expected_type):
    """The plan name `flashmla.py` looks up on the SWA metadata.

    It used to be chosen by `compress_ratio`, which cannot separate V4.1's
    uncompressed ratio-0 layer from its compressed ratio-1 layer: both resolve
    to an operational ratio of 1.
    """
    ratio, _ = resolve_layer_compress_ratio(_v41_config(), layer_id)
    stub = _RoleStub(
        compress_ratio=ratio, v41_roles=v41_layer_roles(_v41_config(), layer_id)
    )
    assert stub.tile_sched_layer_type() == expected_type


def test_tile_scheduler_plan_selected_for_a_v4_layer():
    config = _v4_config()
    assert _tile_type(config, 0) == _LAYER_TYPE_SWAONLY
    assert _tile_type(config, 1) == _LAYER_TYPE_C4A
    assert _tile_type(
        types.SimpleNamespace(num_hidden_layers=2, compress_ratios=[128, 4]), 0
    ) == _LAYER_TYPE_C128A


def test_swa_only_ratio_is_not_read_as_compressed_by_accident():
    """A raw 0 must not be classified as compressing just because V4.1 is on.

    `compress_ratio` is clamped to 1 for the spec, and 1 <= 1 would have been
    the old test for "no compression" -- which V4.1's real ratio-1 layers also
    trip. The distinction has to come from the raw entry.
    """
    config = _v41_config()
    ratio_zero, _ = resolve_layer_compress_ratio(config, 0)
    ratio_one, _ = resolve_layer_compress_ratio(config, 20)
    # The hazard, stated: same operational ratio, opposite meanings.
    assert ratio_zero == ratio_one == 1
    assert _classify(config, 0) == (False, False)
    assert _classify(config, 20) == (True, True)


def test_every_declared_role_field_is_actually_assigned():
    """`__slots__` fields must all be set, or the miss hides until it is read.

    `candidate_source_layer` was declared in `__slots__`, accepted by
    `__init__`, and read by both `__repr__` and the indexer-cache sharing --
    but never assigned, so model construction died with
    `AttributeError: 'V41LayerRoles' object has no attribute
    'candidate_source_layer'` on the first boot that reached it. `__slots__`
    gives no class-level default to paper over it, which is the point: the
    attribute is simply absent until something asks.

    Reading every slot on every layer of the real config is what turns that
    into a caught bug. `repr()` reading all of them is why it is asserted too.
    """
    config = _v41_config()
    for layer_id in range(config.num_hidden_layers + 3):
        roles = v41_layer_roles(config, layer_id)
        assert roles is not None
        unset = [f for f in type(roles).__slots__ if not hasattr(roles, f)]
        assert not unset, f"layer {layer_id}: declared but never assigned: {unset}"
        assert "candidate_source" in repr(roles)
    # And the value is the source layer, not just present: this is what the
    # consumers rewrite into the source's module prefix to find the blocks.
    assert v41_layer_roles(config, 24).candidate_source_layer == 20
    assert v41_layer_roles(config, 36).candidate_source_layer == 20
