# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import (
    deepseek_v4_fp8_einsum,
    deepseek_v4_fp8_einsum_config,
)
from vllm.platforms import current_platform


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.
    SM12x (and every other arch, including SM110): RTX PRO / GB10 do not expose
    the same TMA/TCGEN05 path, so keep the legacy FP32 block-scale layout
    expected by DeepGEMM (this is the ``deepseek_v4_fp8_einsum_config`` else
    branch — only SM100 takes the packed path).

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    return deepseek_v4_fp8_einsum_config(cap.major)


#: Hidden dimension the fused inverse-RoPE/quant kernel emits scales over, and
#: the width the fp8 einsum validates wo_a's weight scales against.
EINSUM_SCALE_BLOCK = 128


def _wo_a_scale_block(wo_a: nn.Module) -> tuple[int, int] | None:
    """wo_a's weight scale block as ``(out, k)``, or ``None`` if not block-FP8."""
    scale = getattr(wo_a, "weight_scale_inv", None)
    if scale is None:
        scale = getattr(wo_a, "weight_scale", None)
    if scale is None or scale.dim() != 2:
        return None
    out_features, in_features = wo_a.weight.shape
    if scale.shape[0] == 0 or scale.shape[1] == 0:
        return None
    return out_features // scale.shape[0], in_features // scale.shape[1]


def _wo_a_bf16(wo_a: nn.Module, o_lora_rank: int) -> torch.Tensor:
    """wo_a dequantized to BF16 ``[groups, in_features, out_rank]``, cached.

    Block-FP8 dequantization is a plain multiply (the scales carry the amax
    normalization; they are not per-tensor divisors). E8M0 scales are
    exponent-only, so the upcast the linear kernels do anyway is exact. The
    result is stored transposed, ready for ``torch.bmm`` against the grouped
    activation layout the inverse-RoPE kernel produces.
    """
    cached = getattr(wo_a, "_dsv41_wo_a_bf16", None)
    if cached is not None:
        return cached
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        _upcast_e8m0_to_fp32,
    )

    weight = wo_a.weight
    scale = getattr(wo_a, "weight_scale_inv", None)
    if scale is None:
        scale = wo_a.weight_scale
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and scale.dtype == e8m0_dtype:
        scale = _upcast_e8m0_to_fp32(scale)
    scale = scale.to(torch.float32)

    out_features, in_features = weight.shape
    blk_out, blk_k = out_features // scale.shape[0], in_features // scale.shape[1]
    declared = getattr(wo_a, "weight_block_size", None)
    if declared is not None and tuple(declared) != (blk_out, blk_k):
        # Every block-FP8 CUDA kernel here keeps the weight and its scale as
        # loaded; a mismatch means some kernel repacked them into a layout this
        # dequantization does not know, and guessing would produce a plausible
        # wrong answer rather than an error.
        raise RuntimeError(
            "wo_a's weight/scale shapes imply a "
            f"{blk_out}x{blk_k} scale block but the quant config declares "
            f"{tuple(declared)}; the weight was repacked after loading and this "
            "CPU-free dequantization cannot read it"
        )
    dequant = (
        weight.to(torch.float32)
        .view(scale.shape[0], blk_out, scale.shape[1], blk_k)
        * scale[:, None, :, None]
    )
    dequant = dequant.view(out_features, in_features).to(torch.bfloat16)
    # [out_features, in_features] -> [groups, in_features, out_rank]: wo_a is
    # block-diagonal over groups, so its rows are `groups` runs of o_lora_rank
    # and the einsum's `hdr` operand wants the rank leading.
    if out_features % o_lora_rank != 0:
        raise RuntimeError(
            f"wo_a rows ({out_features}) are not a multiple of o_lora_rank "
            f"({o_lora_rank})"
        )
    dequant = dequant.view(-1, o_lora_rank, in_features)
    dequant = dequant.transpose(1, 2).contiguous()
    object.__setattr__(wo_a, "_dsv41_wo_a_bf16", dequant)
    return dequant


def _bf16_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """The same O projection in BF16, for checkpoints the fp8 path cannot take.

    The fp8 path pairs a fixed 128-wide activation scale block (the fused
    inverse-RoPE/quant kernel's per-head chunking) with wo_a's weight scale
    block, and the two must agree on where the K boundaries fall — the scales
    multiply *inside* the K loop, so a 32-wide weight block under a 128-wide
    activation block is not something that can be corrected afterwards.
    DeepSeek-V4-Flash ships 128x128, so they agree. DeepSeek-V4.1-Flash ships
    32x32 (its reference implementation hardcodes ``fp8_block_size = 32`` for
    every dense fp8 layer, activations included), and nothing in this tree can
    emit or consume 32-wide blocks through the einsum.

    So dequantize both operands and do the (tiny — under 1% of the layer's
    FLOPs) matmul in BF16. This is more accurate than the reference, which
    rounds both operands to fp8; what it is not is bit-identical to it.
    """
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.models.deepseek_v4.nvidia.dspark_triton import (
        dspark_inv_rope_bf16_layout,
    )

    # Same inverse RoPE (and same [tokens, groups, heads*head_dim] layout) as
    # `fused_inv_rope_fp8_quant`, minus the quantization. The fused kernel
    # takes strides; this one is written for a contiguous input, and the
    # attention backends do not all hand one over (`flashmla._o_proj` passes a
    # transposed view). Copying is a no-op when it already is.
    act = dspark_inv_rope_bf16_layout(
        o.contiguous(),
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    weight = _wo_a_bf16(wo_a, o_lora_rank)

    # wo_a is column-parallel over its groups, so a TP rank may hold a slice of
    # the group axis rather than all of it; take the same slice the fp8 einsum
    # would have. (Under TP the checkpoint already arrives pre-sliced, so this
    # is the identity in practice.)
    weight_groups = weight.shape[0]
    if weight_groups != n_groups:
        if weight_groups % n_groups != 0:
            raise RuntimeError(
                "DeepSeek V4.o wo_a weight groups must match the TP-local "
                "output groups or be an integer multiple of them, got "
                f"weight_groups={weight_groups}, output_groups={n_groups}"
            )
        partitions = weight_groups // n_groups
        start = (get_tensor_model_parallel_rank() % partitions) * n_groups
        weight = weight.narrow(0, start, n_groups)

    return wo_b(grouped_o_einsum(act, weight))


def grouped_o_einsum(
    act: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """``bhr,hdr->bhd`` as a batched matmul: [T, G, D] x [G, D, R] -> [T, G*R].

    ``torch.bmm`` wants the batch dimension first in *both* operands, so move
    the group axis there for the call. The alternative — ``torch.matmul`` (or
    an einsum spelled ``tgd,gdr->tgr``) — broadcasts the weight's group axis
    against the token axis, which for a prefill chunk of a few thousand tokens
    means materializing the weight once per token. That is the whole decoder's
    width of memory for a projection worth under 1% of its FLOPs.
    """
    out = torch.bmm(act.transpose(0, 1), weight).transpose(0, 1)
    # Spelled out rather than `-1`: an empty batch has no element to infer the
    # width from, and a profile run does exactly that.
    return out.reshape(act.shape[0], weight.shape[0] * weight.shape[2])


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
    """
    scale_block = _wo_a_scale_block(wo_a)
    if scale_block is not None and scale_block != (EINSUM_SCALE_BLOCK,) * 2:
        return _bf16_o_proj(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            wo_b,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
        )
    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    # MarlinFP8.process_weights_after_loading renames block-FP8 scales to
    # weight_scale_inv. Non-Marlin kernels keep the on-disk weight_scale name.
    wo_a_scale = getattr(wo_a, "weight_scale_inv", None)
    if wo_a_scale is None:
        wo_a_scale = wo_a.weight_scale
    deepseek_v4_fp8_einsum(
        o_fp8,
        o_scale,
        wo_a.weight,
        wo_a_scale,
        z,
        "bhr,hdr->bhd",
        list(einsum_recipe),
    )
    return wo_b(z.flatten(1))
