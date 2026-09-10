# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's compressor ratios 1 and 2.

V4.1 keeps V4's `DeepseekCompressor` but widens `compress_ratios` to include 1
and 2; the vendor's module branches on the width -- ratio 1 is a plain bf16
projection with no gate and no positional term, above 1 the pooling is an fp32
softmax gate. What the ratio changes in *this* fork is therefore all shape: the
submodules per ratio, the KV-cache spec the compressor publishes, and which
store kernel the forward dispatches to. That is what these tests pin down.

The kernels themselves are not exercised -- nothing here needs a GPU beyond the
device the module is built on -- because the ratio-1/2 numerics follow from
feeding a one-row softmax through the generic launcher, which the serving
recipes cover end to end. The V4 ratios (4 and 128) are asserted alongside as
the regression check that they did not move.
"""

import contextlib
import inspect
import types

import pytest
import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.models.deepseek_v4.compressor import (
    _CUTEDSL_COMPRESS_RATIOS,
    DeepseekCompressor,
    _checkpoint_has_ape,
    _compressor_store_uses_cutedsl,
    _prefer_two_stage_compressor,
)
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.core.kv_cache_utils import group_and_unify_kv_cache_specs
from vllm.v1.kv_cache_interface import MLAAttentionSpec, get_kv_quant_mode

HIDDEN_SIZE = 5120
HEAD_DIM = 512
ROPE_HEAD_DIM = 64

# Which checkpoint provides each ratio, and the module shape that checkpoint
# implies. V4.1 owns ratios 1 and 2 (`[2]*18 + [1]*20`), V4 the other two.
V41_RATIOS = (1, 2)
V4_RATIOS = (4, 128)

# ratio -> (coff, has_gate, has_ape, state block size, fused wkv/wgate shards)
EXPECTED = {
    1: (1, False, False, 8, [HEAD_DIM]),
    2: (1, True, False, 8, [HEAD_DIM, HEAD_DIM]),
    4: (2, True, True, 4, [2 * HEAD_DIM, 2 * HEAD_DIM]),
    128: (1, True, True, 8, [HEAD_DIM, HEAD_DIM]),
}


def _v41_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        qk_rope_head_dim=ROPE_HEAD_DIM,
        rms_norm_eps=1e-20,
        kv_source_layer_ids=[2, 8, 14, 20],
    )


def _v4_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(qk_rope_head_dim=ROPE_HEAD_DIM, rms_norm_eps=1e-20)


def _config_for(ratio: int) -> types.SimpleNamespace:
    return _v41_config() if ratio in V41_RATIOS else _v4_config()


_MP_READY = False


def _init_model_parallel() -> None:
    """One-rank gloo group: `ModelWeightParameter` reads the TP rank at build."""
    global _MP_READY
    if not dist.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29579",
            backend="gloo",
        )
    if not _MP_READY:
        initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        _MP_READY = True


@contextlib.contextmanager
def _vllm_config(config: types.SimpleNamespace, cache_dtype: str = "fp8_ds_mla"):
    vc = VllmConfig()
    vc.model_config = types.SimpleNamespace(
        hf_config=config,
        max_model_len=1024,
        dtype=torch.bfloat16,
        is_moe=True,
    )
    vc.scheduler_config = types.SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=64
    )
    vc.cache_config = types.SimpleNamespace(cache_dtype=cache_dtype, block_size=64)
    with set_current_vllm_config(vc):
        yield vc


@contextlib.contextmanager
def _compressor(ratio: int, cache_dtype: str = "fp8_ds_mla"):
    """One compressor, built the way the model builds it (bf16 default dtype)."""
    with _vllm_config(_config_for(ratio), cache_dtype) as vc:
        _init_model_parallel()
        with set_default_torch_dtype(torch.bfloat16):
            yield (
                DeepseekCompressor(
                    vllm_config=vc,
                    compress_ratio=ratio,
                    hidden_size=HIDDEN_SIZE,
                    head_dim=HEAD_DIM,
                    rotate=True,
                    prefix="model.layers.0.attn.compressor",
                    k_cache_prefix="model.layers.0.attn",
                ),
                vc,
            )


@pytest.mark.parametrize("ratio", [1, 2, 4, 128])
def test_compressor_shape_per_ratio(ratio: int):
    coff, has_gate, has_ape, block_size, output_sizes = EXPECTED[ratio]
    with _compressor(ratio) as (c, _):
        assert c.coff == coff
        assert c.has_gate is has_gate
        assert c.overlap is (ratio == 4)  # only C4's boundary window straddles
        assert (c.ape is not None) is has_ape
        assert c.fused_wkv_wgate.output_sizes == output_sizes
        assert c.fused_wkv_wgate.weight.shape == (sum(output_sizes), HIDDEN_SIZE)
        # Checkpoint dtype, not an fp32 promotion: the only fp32 state is the
        # gate accumulate (`kv_score` is an fp32 GEMM output) and `ape`.
        assert c.fused_wkv_wgate.weight.dtype == torch.bfloat16
        assert c.state_cache.block_size == block_size
        assert c.state_cache.sliding_window == coff * ratio


@pytest.mark.parametrize("ratio", [1, 2, 4, 128])
def test_state_cache_page_size_does_not_depend_on_the_ratio(ratio: int):
    """The page the compressor states share with KV blocks is a fixed 32768 B.

    The compressor state and the layer's KV cache are packed into one physical
    tensor by page, so a ratio whose state row did not land on that page would
    corrupt the packing. Ratios 1 and 2 reuse C128's 8-row page (their state row
    is C128's: no overlap, so `2 * head_dim * 4` bytes per row), and C4's 4-row
    page of 8192 B rows is the same 32768 B. The fp8_ds_mla layout then rounds
    every ratio up to the same 576 B alignment.
    """
    coff = EXPECTED[ratio][0]
    for cache_dtype, padded in (("auto", 32768), ("fp8_ds_mla", 32832)):
        with _compressor(ratio, cache_dtype) as (c, vc):
            spec = c.state_cache.get_kv_cache_spec(vc)
            assert spec.unpadded_page_size_bytes == 32768
            assert spec.page_size_bytes == padded
            assert spec.block_size == c.state_cache.block_size
            assert spec.sliding_window == coff * ratio
            assert spec.head_size == 2 * coff * HEAD_DIM
            assert spec.num_kv_heads == 1
            assert spec.dtype == torch.float32


def test_v41_compressor_drops_ape_for_zeros():
    """V4.1 ships no positional term; adding zeros is the exact same thing.

    `save_partial_states` adds `ape[position % compress_ratio]` to the gate score
    *before* the softmax, so its absence is a zero bias rather than a missing
    scale -- but the parameter must be absent rather than left uninitialised,
    which is what the loader would leave behind for a tensor no checkpoint
    provides.
    """
    assert _checkpoint_has_ape(_v4_config()) is True
    assert _checkpoint_has_ape(_v41_config()) is False
    for ratio in V41_RATIOS:
        with _compressor(ratio) as (c, _):
            assert c.ape is None
            assert c._zero_ape.shape == (ratio, EXPECTED[ratio][0] * HEAD_DIM)
            assert c._zero_ape.dtype == torch.float32
            assert not c._zero_ape.any()
    for ratio in V4_RATIOS:
        with _compressor(ratio) as (c, _):
            assert c.ape.shape == (ratio, EXPECTED[ratio][0] * HEAD_DIM)
            assert c.ape.dtype == torch.float32
            assert c._zero_ape is None


def test_ratio_one_is_a_gateless_single_shard_projection():
    """Ratio 1 has no gate to fuse, so `fused_wkv_wgate` holds one shard.

    The weight mapper still routes `compressor.wkv` to shard 0 and
    `compressor.wgate` to shard 1; a ratio-1 layer simply has no `wgate` tensor
    for the second entry to fire on, and `validate_shard_id` is what the loader
    consults, so shard 1 must not be addressable.
    """
    with _compressor(1) as (c, _):
        assert c.has_gate is False
        assert [n for n, _ in c.named_parameters()] == [
            "fused_wkv_wgate.weight",
            "norm.weight",
        ]
        assert c.fused_wkv_wgate.validate_shard_id(0) is True
        with pytest.raises(ValueError):
            c.fused_wkv_wgate.validate_shard_id(1)


def test_ratio_one_still_gets_a_state_cache():
    """Unlike the vendor's, this fork's ratio-1 layer keeps a state cache.

    The vendor's ratio 1 allocates no `kv_state`/`score_state` because it pools
    a group of one with no gate. Here the compressor's output reaches the store
    kernel only through the state: `save_partial_states` stages it and the store
    step gathers it back. Dropping the state for ratio 1 would therefore drop
    the store kernel's input tensor, its metadata (`CompressorMetadataBuilder`
    is bound to the state cache's spec) and the bound cache.
    """
    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        compress_norm_rope_store_triton,
    )

    # The state is the launcher's input tensor, not an optional extra.
    params = inspect.signature(compress_norm_rope_store_triton).parameters
    assert next(iter(params)) == "state_cache"
    with _compressor(1) as (c, vc):
        assert c.state_cache.state_dim == 2 * HEAD_DIM  # kv_state + score_state
        spec = c.state_cache.get_kv_cache_spec(vc)
        assert spec.sliding_window == 1  # the group is a single token
        assert spec.block_size == 8


def test_store_kernel_routing_keeps_ratios_four_and_128_on_cutedsl():
    """Ratios 1 and 2 take the generic triton launcher; 4 and 128 are untouched."""
    assert _CUTEDSL_COMPRESS_RATIOS == (4, 128)
    for ratio in V4_RATIOS:
        assert _compressor_store_uses_cutedsl(HEAD_DIM, ratio, True) is True
    for ratio in V41_RATIOS:
        assert _compressor_store_uses_cutedsl(HEAD_DIM, ratio, True) is False
    # The indexer's head and every non-CUDA platform were already off this path.
    assert _compressor_store_uses_cutedsl(128, 4, True) is False
    assert _compressor_store_uses_cutedsl(HEAD_DIM, 4, False) is False

    # ...and the kernel that would take the ratios the routing keeps away
    # agrees that it does not implement them, so the routing is not merely a
    # policy choice made in the compressor.
    from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
        compile_split_sparse_attn_cutedsl,
    )

    with pytest.raises(ValueError, match="only supports the real"):
        compile_split_sparse_attn_cutedsl(
            head_size=HEAD_DIM,
            state_width=HEAD_DIM,
            block_size=8,
            rope_head_dim=ROPE_HEAD_DIM,
            fp8_max=448.0,
            quant_block=64,
            token_stride=576,
            scale_dim=8,
            kv_cache_block_size=64,
            kv_block_stride=1,
            compress_ratio=2,
            overlap=False,
            rms_norm_weight_dtype=torch.bfloat16,
        )


def test_two_stage_split_serves_only_c128():
    with _compressor(1) as (c, _):
        assert c._use_two_stage_fused_compressor is False
    with _compressor(2) as (c, _):
        assert c._use_two_stage_fused_compressor is False
    with _compressor(4) as (c, _):
        assert c._use_two_stage_fused_compressor is False
    with _compressor(128) as (c, _):
        # Whatever the platform did for V4's C128 layers, it still does.
        assert c._use_two_stage_fused_compressor is _prefer_two_stage_compressor()


def test_kv_cache_groups_accept_the_v41_ratios():
    """The real consumer of the specs: the DSV4 KV-cache group builder.

    A state cache whose page no group can hold would fail at serve time, after
    the weights are loaded -- so group the four ratios' state specs together
    with the attention specs they would be packed with, as the engine does.
    """
    specs = {}
    for ratio in (1, 2, 4, 128):
        with _compressor(ratio) as (c, vc):
            specs[f"model.layers.{ratio}.attn.comp"] = c.state_cache.get_kv_cache_spec(
                vc
            )
            specs[f"model.layers.{ratio}.attn"] = MLAAttentionSpec(
                block_size=64,
                num_kv_heads=1,
                head_size=HEAD_DIM,
                dtype=torch.uint8,
                tokens_per_state=ratio,
                cache_dtype_str="fp8_ds_mla",
                alignment=576,
                model_version="deepseek_v4",
                kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"),
                state_content_bytes=584,
            )

    groups = group_and_unify_kv_cache_specs(specs)
    assert groups is not None
    state_groups = {
        name: spec.page_size_bytes
        for group in groups
        for name, spec in group.kv_cache_specs.items()
        if name.endswith(".comp")
    }
    assert set(state_groups) == {f"model.layers.{r}.attn.comp" for r in (1, 2, 4, 128)}
    assert set(state_groups.values()) == {32832}  # 32768 padded to 576B
