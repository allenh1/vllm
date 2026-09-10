# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os
from dataclasses import dataclass, field

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_mqa_topk_indices,
    fp8_fp4_paged_mqa_logits,
    fp8_fp4_paged_mqa_topk_indices,
    has_deep_gemm,
)
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    sparse_indexer_max_logits_bytes,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.ops.pcp import maybe_gather_indexer_k
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
SM120_SHORT_ROW_TOPK_ALWAYS_WIDTH = 4096
SM120_SHORT_ROW_TOPK_MAX_WIDTH = 12288

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32

_PERSISTENT_TOPK_SMEM_OK: bool | None = None


def _device_has_persistent_topk_smem() -> bool:
    """True when the active CUDA device can run persistent_topk's fallbacks.

    persistent_topk's oversubscription fallback (FilteredTopKRaggedTransform)
    requires >=128KB opt-in shared memory per block (topk.cu). GB10 and other
    consumer/edge parts expose only ~101KB, so persistent_topk hard-fails
    ("requires >=128KB smem per block (have 101376)") as soon as the decode grid
    oversubscribes the 48 SMs. Gate on smem CAPACITY (not capability family) so
    the same check also catches RTX 50-series (family-120 with <128KB smem).
    Queried once and cached; both values are fixed for the life of the process.
    """
    global _PERSISTENT_TOPK_SMEM_OK
    if _PERSISTENT_TOPK_SMEM_OK is None:
        if not torch.cuda.is_available():
            _PERSISTENT_TOPK_SMEM_OK = False
        else:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            max_smem = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
            num_sms = int(getattr(props, "multi_processor_count", 0) or 0)
            # A/B + diagnostics: force persistent_topk even on low-smem parts
            # (mirrors the CUDA side's VLLM_TOPK_DISABLE_NONCOOP escape hatch).
            force = os.environ.get("VLLM_SPARSE_IDX_FORCE_PERSISTENT_TOPK", "0") == "1"
            _PERSISTENT_TOPK_SMEM_OK = force or max_smem >= 128 * 1024
            logger.info(
                "Sparse indexer persistent_topk smem gate: "
                "num_sms=%d shared_memory_per_block_optin=%d -> persistent_topk=%s",
                num_sms, max_smem, _PERSISTENT_TOPK_SMEM_OK,
            )
    return _PERSISTENT_TOPK_SMEM_OK


def _assert_cutedsl_dcp_merge_supported(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    k: int,
) -> None:
    # The DCP merge only supports the CuteDSL path (Triton pack kernel + CuteDSL
    # stable-topk selector); there is no PyTorch fallback. The first cut targets
    # Blackwell/Hopper with index_topk in (512, 1024, 2048) (the selector's radix
    # sizing); the Triton pack itself has no shape/topk constraints.
    if not has_cutedsl():
        raise RuntimeError(
            "DCP sparse-indexer merge requires CuteDSL; install it or disable DCP."
        )
    if logits.device.type != "cuda":
        raise RuntimeError("DCP sparse-indexer merge requires CUDA tensors.")
    if logits.dtype != torch.float32 or topk_indices.dtype != torch.int32:
        raise RuntimeError(
            "DCP sparse-indexer merge requires fp32 logits and int32 indices."
        )
    if k not in (512, 1024, 2048):
        raise RuntimeError(
            f"DCP sparse-indexer merge requires index_topk in (512, 1024, 2048); "
            f"got {k}."
        )


def _merge_dcp_topk_global(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Merge each DCP rank's local top-K into the global top-K.

    ``topk_indices`` are this rank's local top-K positions into its 1/N KV
    shard. A token in the global top-K must also be in its owning rank's local
    top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at
    most that many on its own rank), so exchanging only the per-rank local
    candidates is exact -- equivalent to all-gathering the full logit matrix,
    but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole
    score row. Overwrites ``topk_indices`` with global token ids (``-1`` for
    padding); the attention backend localizes them back to physical slots per
    rank.
    """
    if dcp_world_size <= 1:
        return

    # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's
    # (score, global_id) candidates on-device, all-gather, then the CuteDSL
    # stable-topk selector.
    _assert_cutedsl_dcp_merge_supported(logits, topk_indices, topk_tokens)
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    packed = torch.empty(
        (*topk_indices.shape, 2),
        dtype=torch.float32,
        device=topk_indices.device,
    )
    pack_dcp_topk_candidates_cutedsl(
        logits,
        topk_indices,
        packed,
        dcp_rank,
        dcp_world_size,
        cp_interleave,
        row_starts,
    )
    gathered = get_dcp_group().all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered, topk_tokens, out=topk_indices
    )


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout, x0 = q[0:32], x1 = q[32:64]
        x0 = tl.load(q_base + offs32).to(tl.float32)
        x1 = tl.load(q_base + 32 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # x0 = q[0, 2, 4, ...], x1 = q[1, 3, 5, ...]
        x0 = tl.load(q_base + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + 64 + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + 64 + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
    )
    return q_fp8, weights_out


def _should_use_sm120_short_row_topk_decode(
    topk_tokens: int,
    logits_width: int,
    is_cuda_sm120: bool,
) -> bool:
    if not is_cuda_sm120 or topk_tokens != 512:
        return False
    if logits_width <= SM120_SHORT_ROW_TOPK_ALWAYS_WIDTH:
        return True
    return logits_width < SM120_SHORT_ROW_TOPK_MAX_WIDTH


def _use_sm120_short_row_topk_decode(
    logits: torch.Tensor,
    topk_tokens: int,
) -> bool:
    return _should_use_sm120_short_row_topk_decode(
        topk_tokens,
        logits.shape[1],
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120),
    )


def _decode_logits_width(max_model_len: int, max_seq_len: int) -> int:
    if max_model_len <= 0:
        return 0
    if max_seq_len <= 0:
        return max_model_len
    return min(max_model_len, max_seq_len)


def _decode_topk_logits_width(
    max_model_len: int, max_seq_len: int, topk_tokens: int
) -> int:
    logits_width = _decode_logits_width(max_model_len, max_seq_len)
    return min(max_model_len, max(logits_width, topk_tokens))


def _sparse_indexer_requires_deep_gemm(use_fp4_cache: bool = False) -> bool:
    if not current_platform.is_cuda():
        return False
    if current_platform.is_device_capability_family(120):
        # The SM120 fallback path covers FP8-Q sparse indexer calls. FP4-Q
        # indexer calls still route through DeepGEMM's fp8_fp4 kernels, so
        # fail during construction instead of letting the first forward hit
        # the generic DeepGEMM missing-dependency error.
        return use_fp4_cache
    return True


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


# ---------------------------------------------------------------------------
# DeepSeek-V4.1's two-level candidate block selection
# ---------------------------------------------------------------------------
#
# V4.1 restricts every index source after the candidate source (layer 20 in the
# released model) to the blocks of compressed positions that source selected, so
# most of the top-k work runs against a fraction of the positions. Level one is
# the vendor's `select_candidate_blocks`, which already lives in the model
# package (`vllm.models.deepseek_v4.attention`) -- it is imported late below
# because that package imports this module.
#
# The one difference is the frame: the vendor scores one request per logits
# tensor, so a query's blocks are groups of `block_size` columns starting at
# column 0. Here a prefill chunk packs several requests into one tensor, and a
# row owns the columns `[cu_seqlen_ks, cu_seqlen_ke)` -- its request's compressed
# positions in the gathered key buffer -- so its blocks start at its own `ks`.
# A row cannot reach another row's columns, so those stay masked out.


def _v41_decode_candidate_mask(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one for decode, where the frame is the index cache itself.

    Every row's blocks start at column 0 -- there is no packing to anchor them
    against, so this is the vendor's call unchanged -- and a row can see the
    first `seq_lens` columns, which are already in compressed positions
    (`DeepseekV32IndexerMetadataBuilder` divides by the compress ratio).

    Returns a bool mask shaped like `logits` (one row per decode token, padded
    tokens included; a padded row has `seq_lens == 0` and comes back all False).
    """
    # Deferred: `vllm.models.deepseek_v4.attention` imports this module.
    from vllm.models.deepseek_v4.attention import select_candidate_blocks

    rows, width = logits.shape
    reach = seq_lens.reshape(-1).clamp(min=0)
    assert reach.numel() == rows, (
        f"V4.1 candidate selection needs one context length per logits row; got "
        f"{reach.numel()} for {rows} rows."
    )
    # The paged logits kernel is asked not to clean what a row cannot reach, so
    # those columns hold whatever was in the (uninitialized) output, and they
    # have to leave the block scores before the max. The vendor's decode tensor
    # is already truncated to the reach, which is the same thing.
    scores = logits.masked_fill(
        torch.arange(width, device=logits.device) >= reach.unsqueeze(-1),
        float("-inf"),
    )
    return select_candidate_blocks(scores, reach, topk_blocks, block_size)


def _v41_prefill_candidate_mask(
    logits: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one for one packed prefill chunk.

    `ks`/`ke` are the chunk's per-row key-buffer bounds
    (`DeepseekV32IndexerPrefillChunkMetadata.cu_seqlen_ks/ke`): a row can see the
    columns `[ks, ke)`, and `ks` is its request's first compressed position in
    the gathered key buffer -- so a row's blocks are groups of `block_size`
    columns *starting at its `ks`*, which is the vendor's grid, anchored by the
    packing rather than by the tensor.

    Returns a bool mask over the columns of `logits` (see the width note below).
    """
    # Deferred: `vllm.models.deepseek_v4.attention` imports this module.
    from vllm.models.deepseek_v4.attention import select_candidate_blocks

    rows, width = logits.shape
    device = logits.device
    # The metadata builds these in int32 (`BuildPrefillChunkMetadataKernel`) and
    # they index `logits` here, which torch takes in int64 and nothing else.
    ks = ks.to(torch.int64)
    ke = ke.to(torch.int64)
    reach = (ke - ks).clamp(min=0)
    assert reach.numel() == rows and ks.numel() == rows, (
        f"V4.1 candidate selection needs one key range per logits row; got "
        f"{ks.numel()}/{reach.numel()} for {rows} rows."
    )
    # One block grid for the whole chunk's tensor: a row reads its own columns
    # through a gather, and a gather has one width for every row. Sizing it needs
    # two host reads -- the widest reach and the last row's column base -- and
    # the metadata does not carry a host-side bound for either (`cu_seqlen_ke` is
    # device-built in `BuildPrefillChunkMetadataKernel`).
    max_reach = int(reach.max().item()) if rows else 0
    block_pad = -(-max(max_reach, 1) // block_size) * block_size
    out_width = (int(ks.max().item()) if rows else 0) + block_pad
    cols = ks.unsqueeze(-1) + torch.arange(
        block_pad, device=device, dtype=ks.dtype
    )
    # Reads past the frame land in the padding the block grid runs into, which no
    # row can reach (a row's reach ends at or before the frame's last column), so
    # what they hold cannot score: the reach mask below takes them out.
    scores = logits.gather(1, cols.clamp(max=width - 1))
    scores = scores.masked_fill(
        torch.arange(block_pad, device=device) >= reach.unsqueeze(-1),
        float("-inf"),
    )
    local = select_candidate_blocks(scores, reach, topk_blocks, block_size)
    # Written back where each row's grid put the blocks. The mask is `out_width`
    # wide rather than `width`: a row's last block is padded out to `block_size`
    # and can run up to `block_size - 1` columns past the frame, exactly as the
    # vendor's `F.pad` tail does before its `[..., :width]` truncation -- a reader
    # trims it back to its own logits' width.
    mask = torch.zeros((rows, out_width), dtype=torch.bool, device=device)
    mask.scatter_(1, cols, local)
    return mask


def _v41_candidate_holder(candidate_source_prefix: str) -> "V41CandidateBlocks":
    """The candidate blocks published under a layer prefix.

    Level one runs on the logits this op materializes, so its result has to
    leave the op and reach the *later* layers that consume it -- and an op is
    handed a layer's name, not the module. `ForwardContext.no_compile_layers` is
    the same `static_forward_context` mapping the model shares its caches
    through, so both ends agree on where the blocks live without this op having
    to learn the model's module tree.
    """
    holder = get_forward_context().no_compile_layers.get(candidate_source_prefix)
    blocks = getattr(holder, "v41_candidate_blocks", None) if holder is not None else None
    if blocks is None:
        raise ValueError(
            f"DeepSeek-V4.1: {candidate_source_prefix or '<unnamed>'} is not a "
            "candidate source: it publishes no candidate blocks, so the index "
            "sources after it cannot restrict their top-k to them."
        )
    return blocks


@dataclass
class V41CandidateBlocks:
    """The candidate blocks one V4.1 forward pass publishes, and its stamp.

    Produced and consumed by the indexer op -- the source layer (the config's
    `candidate_source_layer_id`) picks the blocks, the index sources after it
    mask their scores with them -- and held on the source layer's attention
    module, which is where the op can find it by prefix.

    The masks are bool and shaped like the logits that produced them: one per
    prefill chunk, in chunk order, and one for decode. `forward_context` is the
    forward pass they belong to, held rather than referred to by id: a source
    that does not run its indexer for a batch (the dense-MHA short-extend case,
    which deliberately leaves the top-k buffers alone) leaves it on the previous
    pass, and a reader that needs blocks then fails loudly rather than masking
    its scores with another batch's. Holding the object is what makes that
    reliable -- an id is the object's address, and the previous pass's context
    is usually dead by the time the reader asks, so the new one can be handed
    the same address and compare equal.
    """

    prefill: list[torch.Tensor] = field(default_factory=list)
    #: The `(token_start, token_end)` each mask above was built for, in the same
    #: order. Two layers can only share a chunk's blocks by having split the
    #: batch identically, and the token bounds are how a reader checks that
    #: without comparing the masks themselves.
    prefill_bounds: list[tuple[int, int]] = field(default_factory=list)
    decode: torch.Tensor | None = None
    forward_context: ForwardContext | None = None

    def reset(self) -> None:
        """Start a forward pass: drop the last one's masks and hold this one."""
        self.prefill.clear()
        self.prefill_bounds.clear()
        self.decode = None
        self.forward_context = get_forward_context()

    def publish_prefill(
        self, token_start: int, token_end: int, mask: torch.Tensor
    ) -> None:
        self.prefill.append(mask)
        self.prefill_bounds.append((token_start, token_end))

    def publish_decode(self, mask: torch.Tensor) -> None:
        self.decode = mask

    def _current_pass(self) -> None:
        if self.forward_context is not get_forward_context():
            raise RuntimeError(
                "DeepSeek-V4.1: candidate blocks are stale. The layer that "
                "selects them did not run its indexer for this batch (dense MHA "
                "short extends skip it), so the blocks this layer would mask "
                "with are from an earlier forward pass."
            )

    def take_prefill(
        self,
        index: int,
        token_start: int,
        token_end: int,
        shape: torch.Size,
    ) -> torch.Tensor:
        self._current_pass()
        if index >= len(self.prefill):
            raise RuntimeError(
                f"DeepSeek-V4.1: the candidate source published {len(self.prefill)} "
                f"prefill chunks, so there is no mask for chunk {index}."
            )
        if self.prefill_bounds[index] != (token_start, token_end):
            raise RuntimeError(
                "DeepSeek-V4.1: the candidate source chunked this batch "
                f"differently -- chunk {index} is tokens {token_start}:{token_end} "
                f"here and {self.prefill_bounds[index]} for the source, so its "
                "blocks are not this layer's to use."
            )
        mask = self.prefill[index]
        if mask.shape[0] != shape[0] or mask.shape[-1] < shape[-1]:
            raise RuntimeError(
                "DeepSeek-V4.1: candidate blocks are shaped "
                f"{tuple(mask.shape)}, which does not cover logits shaped "
                f"{tuple(shape)}."
            )
        return mask[:, : shape[-1]]

    def take_decode(self, shape: torch.Size) -> torch.Tensor:
        self._current_pass()
        mask = self.decode
        if mask is None:
            raise RuntimeError(
                "DeepSeek-V4.1: the candidate source ran no decode step, so "
                "there are no blocks for this layer's decode logits to use."
            )
        if mask.shape[0] != shape[0] or mask.shape[-1] < shape[-1]:
            raise RuntimeError(
                "DeepSeek-V4.1: candidate blocks are shaped "
                f"{tuple(mask.shape)}, which does not cover logits shaped "
                f"{tuple(shape)}."
            )
        return mask[:, : shape[-1]]


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    # A plain `str`, unlike the layer names above: torch cannot write a default
    # for the opaque `LayerName` into a schema (the parser has no `LayerName`
    # literal) and this one has to default to "off". The only thing given up is
    # that a compiled graph sees the prefix as a constant rather than a hoisted
    # input -- and V4.1's recipe runs eager (`--enforce-eager`), so there is no
    # graph for it to be a constant in.
    candidate_source_prefix: str = "",
    is_candidate_source: bool = False,
    candidate_topk_blocks: int = 0,
    candidate_block_size: int = 0,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = sparse_indexer_max_logits_bytes()
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_pcp,
            dense_mha_metadata_layer_name,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # V4.1's two-level candidate selection. Level one runs on the logits this op
    # materializes, so it is this op that publishes the blocks (the source layer)
    # and this op that masks the scores with them (every index source after it).
    # Both of the logits-free fast paths below have to stand down while it is
    # active: neither ever holds the scores the blocks are chosen from or applied
    # to.
    use_candidates = bool(candidate_source_prefix) and candidate_topk_blocks > 0
    candidate_blocks: V41CandidateBlocks | None = None
    if use_candidates:
        if dcp_world_size > 1:
            # A DCP rank owns a strided shard of the compressed positions, and
            # both the block a position belongs to and the block scores are
            # defined over all of them.
            raise NotImplementedError(
                "DeepSeek-V4.1 candidate block selection does not support "
                "decode context parallelism."
            )
        candidate_blocks = _v41_candidate_holder(candidate_source_prefix)
        if is_candidate_source:
            candidate_blocks.reset()

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    # Keep PCP padding so every rank contributes the same all-gather shape.
    num_tokens = slot_mapping.shape[0]
    if use_pcp:
        num_tokens //= get_pcp_group().world_size
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert k is not None
        k, slot_mapping_for_cache = maybe_gather_indexer_k(
            k,
            slot_mapping,
            num_decode_tokens,
            use_pcp,
        )
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping_for_cache,
            quant_block_size,
            scale_fmt,
        )

    # The indexer and main MLA may classify the same short extend differently
    # because they use independent decode thresholds. Only the main MLA route
    # can determine whether the top-k indices will be consumed.
    if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
        dense_mha_layer = _resolve_layer_name(dense_mha_metadata_layer_name)
        if dense_mha_layer:
            mla_metadata = attn_metadata.get(dense_mha_layer)
            prefill_metadata = getattr(mla_metadata, "prefill", None)
            if (
                getattr(prefill_metadata, "use_dense_mha", False)
                and getattr(mla_metadata, "num_decode_tokens", -1) == 0
                and not torch.cuda.is_current_stream_capturing()
            ):
                # Deliberately leave the buffer untouched. Dense MHA does not
                # consume top-k indices for this batch; clearing it would be
                # unnecessary work.
                return topk_indices_buffer

    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the
    # top-k kernels scatter valid indices into it. On the fused deepseek_v32
    # nvidia path, _fused_norm_rope_kernel already cleared the same
    # [:num_tokens, :topk] region earlier in this forward, so skip the redundant
    # fill.
    if not skip_topk_buffer_clear:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        # The chunk index is what pairs a consumer with the source's mask below:
        # both walk their own metadata's chunks, which are the same chunks
        # whenever the layers agree on the batch's geometry -- and the bounds
        # check in `take_prefill` is what makes a disagreement loud.
        for chunk_index, chunk in enumerate(prefill_metadata.chunks):
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if chunk.local_total_seq_lens == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
                if candidate_blocks is not None and is_candidate_source:
                    # A chunk with nothing to score still holds its place in the
                    # sequence of chunks both ends index by.
                    candidate_blocks.publish_prefill(
                        chunk.token_start,
                        chunk.token_end,
                        logits.new_zeros((logits.shape[0], 0), dtype=torch.bool),
                    )
            else:
                # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
                # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
                if use_fp4_cache:
                    q_slice_cast = q_slice.view(torch.int8)
                    k_quant_cast = k_quant.view(torch.int8)
                    k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                else:
                    q_slice_cast = q_slice
                    k_quant_cast = k_quant
                    k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                if (
                    candidate_blocks is None
                    and not current_platform.is_xpu()
                    and fp8_fp4_mqa_topk_indices(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        topk_indices,
                    )
                ):
                    continue
                if current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights[chunk.token_start : chunk.token_end],
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                    )
                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        clean_logits=False,
                    )
                num_rows = logits.shape[0]
                if candidate_blocks is not None:
                    if is_candidate_source:
                        candidate_blocks.publish_prefill(
                            chunk.token_start,
                            chunk.token_end,
                            _v41_prefill_candidate_mask(
                                logits,
                                cu_seqlen_ks,
                                cu_seqlen_ke,
                                candidate_topk_blocks,
                                candidate_block_size,
                            ),
                        )
                    else:
                        # Level two: this layer's own scores, but only inside the
                        # candidate source's blocks.
                        logits.masked_fill_(
                            ~candidate_blocks.take_prefill(
                                chunk_index,
                                chunk.token_start,
                                chunk.token_end,
                                logits.shape,
                            ),
                            float("-inf"),
                        )
                ops.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                row_starts=chunk.cu_seqlen_ks,
            )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if num_decode_tokens == 0:
            padded_q_quant_decode_tokens = q_quant[:1].reshape(1, 1, *q_quant.shape[1:])
            padded_q_scale = (
                q_scale[:1].reshape(1, 1, *q_scale.shape[1:])
                if q_scale is not None
                else None
            )
        elif decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        # ``.contiguous()`` was originally required because the
        # ``DeepseekV32IndexerMetadataBuilder`` allocated
        # ``decode_seq_lens_buffer`` as a 2D ``(max_num_seqs, next_n)``
        # tensor, and a ``[:num_decodes, :max_decode_len]`` slice was
        # non-contiguous when ``max_decode_len < next_n`` under V2 model
        # runner cudagraph capture. Reported by aabbccddwasd in PR #41834
        # comment 4450901180. Upstream PR #42135 (ee58665aa) since
        # unified the buffer to 1D ``(max_num_batched_tokens,)``, so the
        # slice is now always contiguous and this call is a no-op pointer
        # return. Kept as a defensive belt against future regressions in
        # the metadata builder's buffer shape.
        seq_lens = decode_metadata.seq_lens[:batch_size].contiguous()
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        logits_width = _decode_topk_logits_width(
            max_model_len, attn_metadata_narrowed.max_seq_len, topk_tokens
        )
        logits_bytes = num_padded_tokens * logits_width * torch.float32.itemsize
        used_direct_topk = False
        if (
            candidate_blocks is None
            and not current_platform.is_xpu()
            and decode_metadata.global_seq_lens is None
            and logits_bytes > sparse_indexer_max_logits_bytes()
        ):
            # The direct top-k kernel never materializes per-position scores,
            # but _merge_dcp_topk_global below needs the full local logits to
            # merge candidates across DCP ranks — fall back to the logits path
            # whenever DCP is active so the merge sees a real score matrix.
            used_direct_topk = fp8_fp4_paged_mqa_topk_indices(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                logits_width,
                topk_indices,
            )

        if not used_direct_topk:
            if current_platform.is_xpu():
                if padded_q_scale is not None:
                    raise RuntimeError(
                        "XPU fp8_paged_mqa_logits does not support FP4 Q"
                    )
                seq_lens_xpu = (
                    seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
                )
                logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                    padded_q_quant_cast,
                    kv_cache,
                    weights[:num_padded_tokens],
                    seq_lens_xpu,
                    decode_metadata.block_table,
                    decode_metadata.schedule_metadata,
                    max_model_len,
                )
            else:
                logits = fp8_fp4_paged_mqa_logits(
                    (padded_q_quant_cast, padded_q_scale),
                    kv_cache,
                    weights[:num_padded_tokens],
                    seq_lens,
                    decode_metadata.block_table,
                    decode_metadata.schedule_metadata,
                    max_model_len=logits_width,
                    clean_logits=False,
                    indices=decode_metadata.indices,
                )
            num_rows = logits.shape[0]

            if candidate_blocks is not None:
                if is_candidate_source:
                    candidate_blocks.publish_decode(
                        _v41_decode_candidate_mask(
                            logits, seq_lens, candidate_topk_blocks, candidate_block_size
                        )
                    )
                else:
                    logits.masked_fill_(
                        ~candidate_blocks.take_decode(logits.shape), float("-inf")
                    )

            use_cooperative_topk = (
                current_platform.is_cuda()
                and topk_tokens in (512, 1024, 2048)
                and num_rows <= 32
                and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
                and current_platform.has_device_capability(90)
                and not current_platform.is_device_capability_family(120)
            )
            use_persistent_topk = (
                current_platform.is_cuda()
                and topk_tokens in (512, 1024, 2048)
                and _device_has_persistent_topk_smem()
            )
            if _use_sm120_short_row_topk_decode(logits, topk_tokens):
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            elif use_cooperative_topk:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.cooperative_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    topk_tokens,
                    attn_metadata_narrowed.max_seq_len,
                )
            elif use_persistent_topk:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    topk_tokens,
                    logits.shape[1],
                )
            else:
                ops.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

        if decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    # A plain `str`, unlike the layer names above: torch cannot write a default
    # for the opaque `LayerName` into a schema (the parser has no `LayerName`
    # literal) and this one has to default to "off". The only thing given up is
    # that a compiled graph sees the prefix as a constant rather than a hoisted
    # input -- and V4.1's recipe runs eager (`--enforce-eager`), so there is no
    # graph for it to be a constant in.
    candidate_source_prefix: str = "",
    is_candidate_source: bool = False,
    candidate_topk_blocks: int = 0,
    candidate_block_size: int = 0,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        compress_ratio: int = 1,
        is_candidate_source: bool = False,
        candidate_source_prefix: str = "",
        candidate_topk_blocks: int = 0,
        candidate_block_size: int = 0,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.compress_ratio = compress_ratio
        # V4.1's two-level top-k. `candidate_source_prefix` is the *attention*
        # module of the layer that picks the candidate blocks -- this layer's own
        # when it is the source, the layer it reads its index keys from
        # otherwise -- and both ends have to agree on the block geometry for a
        # block index to mean the same thing on either side.
        self.is_candidate_source = is_candidate_source
        self.candidate_source_prefix = candidate_source_prefix
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        if candidate_source_prefix:
            assert candidate_topk_blocks > 0 and candidate_block_size > 0, (
                "V4.1 candidate block selection needs `candidate_topk_blocks` "
                "and `candidate_block_size` from the config."
            )
        self.dense_mha_metadata_layer_name = ""
        # DCP scalars are constant for the run; resolve them here (config is set
        # during model construction) and pass them into the custom op, rather
        # than threading them through per-step metadata.
        parallel_config = get_current_vllm_config().parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        if _sparse_indexer_requires_deep_gemm(use_fp4_cache) and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_pcp,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
            candidate_source_prefix=self.candidate_source_prefix,
            is_candidate_source=self.is_candidate_source,
            candidate_topk_blocks=self.candidate_topk_blocks,
            candidate_block_size=self.candidate_block_size,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if self.candidate_source_prefix:
            raise NotImplementedError(
                "DeepSeek-V4.1 candidate block selection is not implemented for "
                "the ROCm sparse indexer."
            )
        from vllm.platforms.rocm import on_gfx11

        if (
            rocm_aiter_ops.is_enabled()
            or rocm_aiter_ops.is_rdna_aiter_enabled()
            or on_gfx11()
        ):
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
                compress_ratio=self.compress_ratio,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
