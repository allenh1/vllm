# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.kernels.linear import (
    TritonFp8BlockScaledMMKernel,
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.sparse_attn_indexer import (
    SparseAttnIndexer,
    V41CandidateBlocks,
)
from vllm.models.common.ops import fused_q_kv_rmsnorm
from vllm.models.deepseek_v4.common.ops import (
    fused_indexer_q_rope_quant,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_pcp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.triton_utils import tl, triton
from vllm.utils.multi_stream_utils import (
    execute_in_parallel,
    maybe_execute_in_parallel,
)
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    dsa_indexer_uses_fp4,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.attention.ops.pcp import maybe_gather_indexer_k
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)


@triton.jit
def _fill_short_context_topk_indices(
    output,
    positions,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    # small triton kernel that selects every candidate, -1 otherwise
    row = tl.program_id(0)
    offsets = tl.arange(0, PADDED_TOP_K)
    num_compressed = (tl.load(positions + row) + 1) // COMPRESS_RATIO
    tl.store(
        output + row * TOP_K + offsets,
        tl.where(offsets < num_compressed, offsets, -1),
        mask=offsets < TOP_K,
    )


def _resolve_dsv4_kv_cache_dtype(
    use_fp8_ds_mla_layout: bool,
    kv_cache_dtype: str,
    cache_config: CacheConfig | None,
) -> tuple[str, torch.dtype]:
    """Map ``(layout, --kv-cache-dtype)`` to ``(cache_dtype_str, torch_dtype)``.

    Both layouts are paged; they differ in the per-token block format. The
    ``fp8_ds_mla`` format is UE8M0 block-scaled fp8 packed as ``uint8`` (the
    canonical ``fp8_ds_mla`` string is written back onto ``cache_config`` so the
    page-size specs pick the 576B per-token slot). Plain-row backends store each
    token's KV row in its element dtype: bf16 or per-tensor FP8 E4M3.
    """
    if use_fp8_ds_mla_layout:
        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.
        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, "
            f"got {kv_cache_dtype}"
        )
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8

    # Plain bf16 / per-tensor fp8 KV row (FlashInfer).
    if kv_cache_dtype.startswith("fp8"):
        return kv_cache_dtype, torch.float8_e4m3fn
    # auto / bfloat16 -> plain bf16 KV row.
    return kv_cache_dtype, torch.bfloat16


def resolve_layer_compress_ratio(config, layer_id: int) -> tuple[int, bool]:
    """Resolve (operational_compress_ratio, use_unscaled_rope) for a layer.

    NOTE(zyongye) Compress ratio can't be 0; historically every layer_id >=
    num_hidden_layers (the MTP draft layer) was mapped to 1 because "MTP layer
    is not included in the compress ratio list". Some checkpoints DO include
    their MTP draft layer in compress_ratios, with an entry of 0 meaning
    uncompressed KV and plain (unscaled) rope. The operational ratio stays
    clamped to >= 1 (KV-cache specs treat 1 as "no compression" and divide by
    it); a raw 0 only selects unscaled rope for that layer.
    """
    roles = v41_layer_roles(config, layer_id)
    if roles is not None:
        return roles.compress_ratio, roles.use_unscaled_rope
    if layer_id < config.num_hidden_layers:
        return max(1, config.compress_ratios[layer_id]), False
    if layer_id < len(config.compress_ratios):
        raw_compress_ratio = config.compress_ratios[layer_id]
        return max(1, raw_compress_ratio), raw_compress_ratio == 0
    return 1, False


# ---------------------------------------------------------------------------
# DeepSeek-V4.1 layer roles
# ---------------------------------------------------------------------------
#
# V4.1 reuses the V4 attention module but reads `compress_ratios` differently
# and shares one compressed cache across many layers instead of giving every
# layer its own copy. Three things change, all of them keyed off the raw entry:
#
#   raw 0  no compression: pure sliding window on plain (unscaled) rope. V4
#          spells the same thing, but in V4.1 it appears *inside* the backbone
#          (layers 0 and 1), not only on the draft layer.
#   raw 1  one latent per token: `norm(wkv(x))`, no gate, no fp32 (layers
#          20..39). V4 has no ratio-1 mode at all -- it reads a raw 1 as
#          "uncompressed" -- so the two readings cannot share a code path.
#   raw 2  two tokens softmax-pooled into one latent (layers 2..19).
#
# The cache is shared. `kv_source_layer_ids` names the only layers that build a
# compressor and own a compressed cache -- (2, 8, 14, 20) -- so every layer
# between two sources reads the cache its source published instead of one of
# its own. `index_source_layer_ids` does the same for the indexer; the entries
# that are not also KV sources (24, 28, 32, 36) carry no `wk`/`k_norm` in the
# checkpoint precisely because they score against the shared index keys.
#
# The indexer shares the same way, one level down. `index_source_layer_ids`
# names the eight layers that run an indexer; only the four that are also KV
# sources carry `wk`/`k_norm` and own an index-key cache, and the other four
# score against the keys the *most recent* owner published. `candidate_source_layer_id`
# (= 20) is the two-level selection: it is both a KV source and the ratio-1 index
# owner, so it is the one layer whose index keys every later source reads, and it
# is the layer that picks the candidate blocks every later source restricts its
# own top-k to.
#
# A V4 checkpoint has none of these config keys, so `is_deepseek_v41` is False
# for it and none of this code is reached.


class V41LayerRoles:
    """One V4.1 layer's role, resolved from the config. See the note above."""

    __slots__ = (
        "raw_compress_ratio",
        "compress_ratio",
        "compression_enabled",
        "owns_compressed_kv",
        "runs_indexer",
        "kv_owner_layer",
        "owns_index_keys",
        "index_owner_layer",
        "candidate_source_layer",
        "is_candidate_source",
        "uses_candidate_blocks",
        "use_unscaled_rope",
    )

    def __init__(
        self,
        raw_compress_ratio: int,
        compress_ratio: int,
        owns_compressed_kv: bool,
        runs_indexer: bool,
        kv_owner_layer: int | None,
        use_unscaled_rope: bool,
        owns_index_keys: bool = False,
        index_owner_layer: int | None = None,
        candidate_source_layer: int | None = None,
        is_candidate_source: bool = False,
        uses_candidate_blocks: bool = False,
    ):
        self.raw_compress_ratio = raw_compress_ratio
        # Never 0, even for a raw 0 layer: the KV-cache specs divide by it and
        # `compression_enabled` is what says whether the result is used.
        self.compress_ratio = compress_ratio
        self.compression_enabled = raw_compress_ratio > 0
        self.owns_compressed_kv = owns_compressed_kv
        self.runs_indexer = runs_indexer
        #: The KV-source layer whose compressed cache and index keys this layer
        #: reads, or None when it owns them.
        self.kv_owner_layer = kv_owner_layer
        #: This layer carries `wk`/`k_norm` and writes an index-key cache. True
        #: only for an index source that is also a KV source, which is what the
        #: checkpoint's tensor list shows: `indexer.wk`/`k_norm` exist on
        #: 2/8/14/20 and on none of 24/28/32/36.
        self.owns_index_keys = owns_index_keys
        #: The index source whose index-key cache this layer scores against, or
        #: None when it owns it. Only an index source has one: a layer that does
        #: not run an indexer reuses the top-k its source published and never
        #: looks at the keys.
        self.index_owner_layer = index_owner_layer
        #: The layer whose published candidate blocks this one will read, or
        #: None at or before the source. The consumer needs the *module* to find
        #: them, which is a prefix rewrite from here.
        self.candidate_source_layer = candidate_source_layer
        #: Level one of the two-level top-k: this layer runs
        #: `select_candidate_blocks` on its own scores and publishes the result.
        self.is_candidate_source = is_candidate_source
        #: Level two: this layer masks its own scores with the candidate blocks
        #: its source published before taking its top-k.
        self.uses_candidate_blocks = uses_candidate_blocks
        self.use_unscaled_rope = use_unscaled_rope

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"V41LayerRoles(raw={self.raw_compress_ratio}, "
            f"compression={self.compression_enabled}, "
            f"owns_kv={self.owns_compressed_kv}, indexer={self.runs_indexer}, "
            f"kv_owner={self.kv_owner_layer}, "
            f"owns_index={self.owns_index_keys}, "
            f"index_owner={self.index_owner_layer}, "
            f"candidate_source={self.candidate_source_layer}, "
            f"candidates={self.is_candidate_source}/{self.uses_candidate_blocks})"
        )


def is_deepseek_v41(config) -> bool:
    """True for a V4.1 config -- the only kind that names its source layers."""
    return getattr(config, "kv_source_layer_ids", None) is not None


#: `model.layers.7.attn` -> head `model.layers.`, index `7`, tail `.attn`.
_V41_LAYER_IN_PREFIX = re.compile(r"^(?P<head>.*?layers\.)(?P<index>\d+)(?P<tail>\..*)$")


def _v41_owner_prefix(prefix: str, owner_layer_id: int) -> str:
    """The attention prefix of `owner_layer_id`, given this layer's prefix.

    Cross-layer KV sharing is keyed on the module name, so a V4.1 layer that
    reads its source's compressed cache has to be able to name it. Both are the
    same `...layers.<id>.attn` path with a different index, so this is a rewrite
    of that one component rather than a second naming scheme.
    """
    match = _V41_LAYER_IN_PREFIX.match(prefix)
    if match is None:
        raise ValueError(
            f"DeepSeek-V4.1: cannot point {prefix!r} at layer {owner_layer_id}; "
            "its module path has no `layers.<n>` component to rewrite, so the "
            "shared compressed-KV cache cannot be named."
        )
    return f"{match.group('head')}{owner_layer_id}{match.group('tail')}"


def v41_layer_roles(config, layer_id: int) -> V41LayerRoles | None:
    """Resolve `layer_id`'s V4.1 role, or None when this is not a V4.1 config.

    `layer_id` past the backbone is a DSpark draft layer: those are
    uncompressed (raw 0) and share nothing, matching `compress_ratios[40:]`.
    """
    if not is_deepseek_v41(config):
        return None

    ratios = getattr(config, "compress_ratios", None) or ()
    raw = int(ratios[layer_id]) if layer_id < len(ratios) else 0

    kv_sources = tuple(getattr(config, "kv_source_layer_ids", None) or ())
    index_sources = tuple(getattr(config, "index_source_layer_ids", None) or ())

    owns_compressed_kv = raw > 0 and layer_id in kv_sources
    # The most recent source at or before this layer. Only a compressing layer
    # reads a shared cache, so a raw 0 layer (including every draft layer) has
    # no owner however many sources precede it -- it must not be put in a
    # compressed cache group it has no spec for.
    owner = max((s for s in kv_sources if s <= layer_id), default=None)

    # Index keys are derived from the compressor's latent, so -- exactly like
    # the compressed cache -- only a KV source can produce them. The index
    # sources that are not KV sources (24/28/32/36) carry no `wk`/`k_norm` in
    # the checkpoint for that reason and score against the newest owner's keys.
    runs_indexer = raw > 0 and layer_id in index_sources
    owns_index_keys = runs_indexer and layer_id in kv_sources
    index_owner = max((s for s in index_sources if s <= layer_id and s in kv_sources), default=None)

    candidate_source = getattr(config, "candidate_source_layer_id", None)
    candidate_enabled = candidate_source is not None and candidate_source >= 0

    return V41LayerRoles(
        raw_compress_ratio=raw,
        compress_ratio=max(1, raw),
        owns_compressed_kv=owns_compressed_kv,
        runs_indexer=runs_indexer,
        kv_owner_layer=None if (owns_compressed_kv or raw == 0) else owner,
        # A raw 0 layer is trained on plain rope; so is the draft layer.
        use_unscaled_rope=raw == 0,
        owns_index_keys=owns_index_keys,
        index_owner_layer=(
            None if (owns_index_keys or not runs_indexer) else index_owner
        ),
        # Named rather than left to the caller to read off the config: the
        # layers that consume the candidate set need the source's *module* to
        # find the blocks it published, and that is a prefix rewrite from here.
        candidate_source_layer=candidate_source if candidate_enabled else None,
        # The vendor's rule is `layer_id == candidate_source_layer` / `0 <=
        # candidate_source_layer < layer_id`, narrowed here to the layers that
        # run an indexer. Only those have scores of their own to select from or
        # to mask: a layer without one reuses the top-k its source published and
        # never sees the candidate set, and V4.1's three draft layers are raw 0.
        is_candidate_source=(
            runs_indexer and candidate_enabled and layer_id == candidate_source
        ),
        # The source itself is not a consumer -- it scores against every
        # reachable position, which is what makes its own top-k the one it
        # would have taken with no candidate step at all.
        uses_candidate_blocks=(
            runs_indexer and candidate_enabled and layer_id > candidate_source
        ),
    )


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of V4.1's two-level top-k: the blocks a query may attend to.

    A port of the vendor's `select_candidate_blocks`, semantics unchanged.
    `logits` is `[..., n_positions]` with the positions a query cannot reach
    already at -inf, which is what makes a block score of -inf mean "not
    reachable yet" rather than "scored badly"; `compress_lens` is how many of
    those positions this query can reach -- a plain int during decode, or a
    tensor broadcasting against `logits`' leading dims when every row is a
    different query (prefill, and speculative decode, where each draft position
    has its own context length).

    Two details carry the design and are easy to lose in a rewrite:

    * ``-width % block_size`` pads the *logits* out to a whole number of blocks
      with -inf, so a trailing partial block scores as the amax of its real
      entries only; a shorter width must not be scored as if the missing
      entries were 0.
    * The query's newest block is pinned to +inf before the block top-k. It is
      the one block that is only partly filled -- it holds the most recent
      tokens, which the sliding window has not yet covered, and without the pin
      a full older block outscoring it on amax would drop them. It is pinned
      unconditionally, including when the query can reach nothing at all
      (`compress_lens == 0`), where `last` is negative and pins nothing.

    The result is a bool mask shaped like `logits`, so the layers that consume
    it mask their scores and never think about blocks again. Blocks that came
    back -inf -- unreachable leftovers when fewer than `topk_blocks` blocks are
    reachable -- are dropped rather than kept, which is why the mask is built
    from `values > -inf` and not from the indices alone.

    The mask is per *block*, not per position: a selected block's positions come
    back set even where the query cannot reach them, which only ever happens
    inside the newest block. That is the vendor's behaviour and it is safe for
    the same reason there -- the caller owes this function logits that are -inf
    past the reach, so an unreachable position inside a selected block can never
    win the top-k that follows.

    `compress_lens` is an int, a 0-dim tensor for a whole-call value, or a
    tensor shaped like `logits`' leading dims (one entry per row). The vendor's
    prefill hands it a trailing 1 already on it; this wants the shape without
    that dim, because the trailing axis here is the block axis it broadcasts
    against.
    """
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    if isinstance(last, torch.Tensor):
        # One entry per row, so it has to meet `scores`' trailing block axis
        # with a singleton there. Without it, `arange(num_blocks) == last` is
        # read as a same-rank comparison and raises whenever the batch size
        # happens to differ from the block count.
        last = last.unsqueeze(-1)
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


# ---------------------------------------------------------------------------
# V4.1 index keys
# ---------------------------------------------------------------------------
#
# V4.1 deleted the indexer's own compressor. In V4 the indexer carried a full
# `DeepseekCompressor` of its own (its own `wkv`/`wgate` projection over the
# hidden state, its own state cache, its own pool+norm) and stored its keys
# through it. V4.1's checkpoint has no such tensors: the index sources that own
# keys (2/8/14/20) carry `indexer.wk` (512 -> 128) and `indexer.k_norm` (RMSNorm
# over 128) instead, and the vendor applies them to the *attention* compressor's
# RoPE-free latent:
#
#     k = k_norm(wk(latent))            # latent = the compressor's norm(pooled)
#     rope(k[..., -64:], at group position)
#     quantize and write to the index key cache
#
# The latent is how the vendor's `Compressor.forward` *returns*; there it is one
# row per compressed position (None while the current group is still filling
# up). The fork's compressor returns nothing and keeps the pooled, normed value
# inside its store kernel, so `_v41_latent_from_state_cache` below re-derives it
# from the state cache the compressor just wrote. That is a second copy of the
# compressor's pooling rule and it is the one place in this port that is not a
# straight port of a vendor function; it exists so V4.1 does not have to wait on
# `compressor.py` returning its latent, and it is written to be the kernel's
# twin rather than a re-interpretation of it -- same gather positions, same
# softmax, same RMSNorm, same place where bf16 rounding happens. If
# `DeepseekCompressor.forward` ever hands the latent over, this function is what
# it replaces.


def _v41_latent_from_state_cache(
    state_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compress_ratio: int,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """The attention compressor's RoPE-free latent, one row per token.

    Row `i` is the latent of the group token `i` *ends*, built exactly the way
    `_fused_kv_compress_norm_rope_insert_sparse_attn` builds it: gather the
    group's `compress_ratio` state rows, softmax the score half, weight the kv
    half by it, RMSNorm with the compressor's own weight. A token that does not
    end a group gets the same pooling over a clamped window instead of nothing;
    those rows are never written to the index key cache (their compressed slot
    is -1) and exist only so the tensor stays token-aligned.

    Positions before the start of the sequence are masked out of the softmax and
    contribute zero kv, which is what the kernel does with `mask_pos`.

    Only ratios 1 and 2 reach here -- V4.1's compressed layers -- neither of
    which overlaps two compression blocks, so the kernel's `head_offset` term
    (the C4 boundary's second read) has no counterpart.
    """
    assert compress_ratio in (1, 2), (
        f"V4.1 index keys are derived for compress_ratio 1 and 2, got "
        f"{compress_ratio}"
    )
    num_blocks, block_size, state_dim = state_cache.shape
    state_width = state_dim // 2
    head_dim = norm_weight.shape[0]
    assert state_width == head_dim, (
        f"compressor state row is {state_dim} wide ({state_width} per half) but "
        f"its norm weight covers {head_dim}; the latent would be mis-sliced."
    )

    # The group this token ends, in absolute positions. The kernel's
    # `start = position - (1 + overlap) * ratio + 1` with no overlap.
    group = torch.arange(compress_ratio, device=positions.device)
    group_positions = positions.unsqueeze(-1) - compress_ratio + 1 + group
    valid = group_positions >= 0
    safe_positions = group_positions.clamp(min=0)
    # Same slot arithmetic as the kernel: absolute position into the block table.
    slots = (
        block_table[token_to_req_indices.long().unsqueeze(-1), safe_positions // block_size]
        * block_size
        + safe_positions % block_size
    )
    rows = state_cache.reshape(-1, state_dim)[slots]  # [T, ratio, 2 * head_dim]

    keep = valid.unsqueeze(-1)
    score = rows[..., state_width:].float().masked_fill(~keep, float("-inf"))
    kv = rows[..., :state_width].float().masked_fill(~keep, 0.0)
    pooled = (kv * score.softmax(dim=-2)).sum(dim=-2)

    variance = pooled.pow(2).mean(dim=-1, keepdim=True)
    return pooled * torch.rsqrt(variance + norm_eps) * norm_weight


def _v41_index_rope(
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    group_positions: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """GPT-J RoPE on the last `rope_dim` dims of `k`, at `group_positions`.

    The same convention as the fused Q kernel (`is_neox=False`: pairs are
    adjacent, cos/sin are per pair) and as the compressor's store kernel, so the
    index keys land in the same frame as the index queries that score them.

    `group_positions` is the *compressed* position, `ratio * group_index`, which
    is the vendor's `freqs_cis[start_pos + 1 - ratio]` for the group that just
    completed and its prefill spelling for a whole chunk.
    """
    assert k.shape[-1] >= rope_dim
    half = rope_dim // 2
    cos_sin = cos_sin_cache[group_positions]
    cos = cos_sin[..., :half].float()
    sin = cos_sin[..., half:].float()
    pairs = k[..., -rope_dim:].unflatten(-1, (half, 2)).float()
    even, odd = pairs[..., 0], pairs[..., 1]
    roped = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    out = k.clone()
    out[..., -rope_dim:] = roped.flatten(-2).to(k.dtype)
    return out


class DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC):
    """DeepseekV4 MLA attention layer.

    The platform-specific sparse-MLA forward (``forward_mqa`` /
    ``get_padded_num_q_heads`` / ``_o_proj`` / ``backend_cls``) is provided by a
    subclass — ``DeepseekV4FlashMLAAttention`` /
    ``DeepseekV4FlashInferSM120Attention`` /
    ``DeepseekV4FlashInferMLAAttention`` (CUDA) or
    ``DeepseekV4ROCMAiterMLAAttention`` (ROCm) — selected by the platform-specific
    deepseek_v4 model module. The base is never instantiated directly.
    """

    # Provided by the platform subclass.
    backend_cls: ClassVar[type[AttentionBackend]]
    # Backend for the SWA cache layer; None uses the default SWA backend.
    swa_backend_cls: ClassVar[type[AttentionBackend] | None] = None
    # KV-cache per-token block format (both layouts are paged). True (default)
    # = fp8_ds_mla (UE8M0 block-scaled fp8 packed as uint8); False = plain
    # bf16 / per-tensor fp8 KV row. Backends can override the instance hook when
    # a single attention class dispatches across arch-specific layouts.
    use_fp8_ds_mla_layout: ClassVar[bool] = True
    # Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
    # workspace allocated in _forward_prefill and is also read by the dummy-run
    # path to pre-reserve that workspace.
    PREFILL_CHUNK_SIZE: ClassVar[int] = 4
    _q_padded_scratch_by_key: ClassVar[
        dict[tuple[str, int, int, torch.dtype, int, int], torch.Tensor]
    ] = {}

    @classmethod
    @abstractmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        """Q head count the q/output buffers are allocated at.

        The layer allocates the q/output buffers at
        ``[N, get_padded_num_q_heads(n_local_heads), head_dim]``. Must satisfy
        ``result >= num_heads``. Backends with no padding constraint return
        ``num_heads``.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Platform-specific sparse MLA forward; writes attention into ``output``."""
        raise NotImplementedError

    @abstractmethod
    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Inverse-RoPE + wo_a + wo_b output projection (platform-specific)."""
        raise NotImplementedError

    def _uses_fp8_ds_mla_layout(self) -> bool:
        """Return whether this instance stores fp8 KV in fp8_ds_mla layout."""
        return self.use_fp8_ds_mla_layout

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        eager_scratch_pool: "DeepseekV4EagerScratchPool | None" = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)

        self.prefix = prefix  # Alias for compatibility with compressor
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        self.compress_ratio, use_unscaled_rope = resolve_layer_compress_ratio(
            config, layer_id
        )
        # DeepSeek-V4.1 only: which layers own the compressed cache and which
        # read someone else's. None on V4, where every layer owns its own.
        self.v41_roles = v41_layer_roles(config, layer_id)
        self.layer_id = layer_id
        self.prefix = prefix
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5

        # Padded Q head count is dictated by the platform subclass.
        self.padded_heads = self.get_padded_num_q_heads(self.n_local_heads)
        # Sink padded to the same head count, initialized to -inf (no sink
        # effect). Weight loading fills the first n_local_heads slots.
        self.attn_sink = nn.Parameter(
            torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )

        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups

        # wo_a is consumed RAW by deep_gemm_fp8_o_proj -> deepseek_v4_fp8_einsum,
        # which expects the out-major fp8 weight as [out_pp, in_pp] (per-TP-shard
        # [2048, 4096]) plus its [N, K] block scale. The block-kernel selection
        # (post-rebase) routes this layer to a repacking kernel on SM120
        # (Humming / DeepGemm / Cutlass), whose process_weights_after_loading
        # mutates the weight into a K-major / packed / 3D layout the einsum cannot
        # consume. Force the pad-only Triton block kernel so the weight stays
        # out-major fp8 [out_pp, in_pp].
        qm = getattr(self.wo_a, "quant_method", None)
        if getattr(qm, "block_quant", False) and hasattr(qm, "fp8_linear"):
            qm.fp8_linear = init_fp8_linear_kernel(
                activation_quant_key=qm.activation_quant_key,
                weight_quant_key=qm.weight_quant_key,
                input_dtype=qm.input_dtype,
                out_dtype=qm.out_dtype,
                weight_shape=tuple(self.wo_a.weight.shape),
                force_kernel=TritonFp8BlockScaledMMKernel,
                module_name="deepseek_v4_wo_a",
            )
            qm.use_marlin = False
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        # Initialize rotary embedding before the indexer/compressor consume it.
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
            use_unscaled_rope=use_unscaled_rope,
            use_compress_rope_theta=(
                None
                if self.v41_roles is None
                else self.v41_roles.compression_enabled
            ),
        )
        self.indexer_rotary_emb = self.rotary_emb
        self.topk_indices_buffer = topk_indices_buffer
        self.eager_scratch_pool = eager_scratch_pool

        # Create the compressor for layers with compress_ratio > 1, before the
        # indexer: a V4.1 index-key owner derives its keys from the latent this
        # compressor publishes, so it is handed the module. V4.1 compresses at
        # ratio 1 too, but only its KV sources carry the weights -- the layers
        # in between read the cache their source wrote.
        self.compressor = None
        build_compressor = (
            self.compress_ratio > 1
            if self.v41_roles is None
            else self.v41_roles.owns_compressed_kv
        )
        if build_compressor:
            self.compressor = DeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
                eager_scratch_pool=eager_scratch_pool,
            )

        self.indexer = None
        # Where this layer publishes the candidate blocks it selects, for the
        # index sources after it to read. It lives on the attention module
        # because the indexer op is handed a module *prefix* rather than a
        # module, and this is the mapping (`static_forward_context`) both ends
        # already share their caches through -- `_v41_candidate_holder` is the
        # op's side of it. Only the candidate source has one; see
        # `V41CandidateBlocks` for why the handover is stamped by forward pass.
        self.v41_candidate_blocks = (
            V41CandidateBlocks()
            if self.v41_roles is not None and self.v41_roles.is_candidate_source
            else None
        )
        # V4 gates the indexer on the ratio (only C4A has one); V4.1 names the
        # layers that run one, which is a different set entirely -- the
        # index-only sources 24/28/32/36 have no compressor of their own but do
        # have an indexer, and the ratio-1 source 20 has one too.
        build_indexer = (
            self.compress_ratio == 4
            if self.v41_roles is None
            else self.v41_roles.runs_indexer
        )
        if build_indexer:
            # This layer's index-key cache is somebody else's when it is not an
            # owner: the same prefix rewrite the compressed cache uses, one
            # level down (an index source that does not own `wk`/`k_norm`).
            index_cache_owner_prefix = ""
            if (
                self.v41_roles is not None
                and not self.v41_roles.owns_index_keys
                and self.v41_roles.index_owner_layer is not None
            ):
                index_cache_owner_prefix = _v41_owner_prefix(
                    prefix, self.v41_roles.index_owner_layer
                )
            # The layer that publishes the candidate blocks every later index
            # source restricts its own top-k to: its own prefix when it is the
            # source, the source's when it consumes. An index source *before*
            # the candidate source (2/8/14) names it too -- `candidate_source_
            # layer` is set for every V4.1 layer -- but must not read it: the
            # vendor's rule is `candidate_source_layer < layer_id`, so those
            # layers score against every reachable position like the source
            # does, and taking a mask published by a layer that runs later in
            # the forward would be reading last step's.
            candidate_source_prefix = ""
            if self.v41_roles is not None and (
                self.v41_roles.is_candidate_source
                or self.v41_roles.uses_candidate_blocks
            ):
                candidate_source_prefix = (
                    prefix
                    if self.v41_roles.is_candidate_source
                    else _v41_owner_prefix(
                        prefix, self.v41_roles.candidate_source_layer
                    )
                )
            # aux_stream_list[2] is free here (outer GEMMs joined) for the inner
            # overlap of wq_b+fused_indexer_q_rope_quant vs compressor. None on
            # ROCm, where aux_stream_list is None.
            indexer_aux_stream = (
                aux_stream_list[2] if aux_stream_list is not None else None
            )
            self.indexer = DeepseekV4Indexer(
                vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=self.compress_ratio,
                prefix=f"{prefix}.indexer",
                aux_stream=indexer_aux_stream,
                eager_scratch_pool=eager_scratch_pool,
                v41_roles=self.v41_roles,
                attention_head_dim=self.head_dim,
                latent_compressor=self.compressor,
                index_cache_owner_prefix=index_cache_owner_prefix,
                candidate_source_prefix=candidate_source_prefix,
                candidate_topk_blocks=int(
                    getattr(config, "candidate_topk_blocks", 0) or 0
                ),
                candidate_block_size=int(
                    getattr(config, "candidate_block_size", 0) or 0
                ),
            )

        self._prepare_and_attn_fn = self._prepare_and_attn
        if not vllm_config.use_v2_model_runner:
            # MRV1's piecewise capture only tolerates the wide eager region: with
            # the narrow one the attention input preparation stays in the captured
            # graph and MRV1 produces garbage (#51430).
            self._prepare_and_attn_fn = self._prepare_and_attn_eager

        # Will be None on ROCm for now.
        self.aux_stream_list = aux_stream_list
        # [0]: GEMM start / post-GEMM event0. [1..3]: GEMM done events;
        # [1] doubles as post-GEMM event1. Reuse is safe: GEMM fully joins
        # before post-GEMM starts.
        self.ln_events = [torch.cuda.Event() for _ in range(4)]

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        # ---- Attention / KV-cache setup ----
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self._q_padded_scratch_num_ubatches = (
            2 if vllm_config.parallel_config.enable_dbo else 1
        )
        self._q_padded_scratch_dtype = vllm_config.model_config.dtype

        # Resolve the kv-cache dtype from this backend's block format. The same
        # resolution drives the SWA cache tensor dtype below.
        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            self._uses_fp8_ds_mla_layout(), cache_config.cache_dtype, cache_config
        )

        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=self.kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
            backend_cls=self.swa_backend_cls,
        )

        # Register with compilation context for metadata lookup.
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])

        # A V4.1 layer that does not own the compressed cache reads its source's
        # -- the same paged tensor and the same block table -- by name. That is
        # the mechanism the compressor already uses to find its own cache
        # (`_static_forward_context[k_cache_prefix]` + `attn_metadata[...]`),
        # and it is the one that works here: vLLM's own cross-layer sharing
        # (`kv_sharing_target_layer_name`) is discovered through
        # `get_layers_from_vllm_config(config, Attention)`, and this class is
        # an AttentionLayerBase but not an Attention, so declaring the
        # attribute would silently leave the layer with no cache at all and an
        # empty tensor in `forward_mqa`.
        #
        # `swa_cache` is deliberately not shared: V4.1's sliding window is
        # per-layer, only the compressed positions are common.
        self.compressed_cache_prefix = None
        if self.v41_roles is not None and self.v41_roles.kv_owner_layer is not None:
            self.compressed_cache_prefix = _v41_owner_prefix(
                prefix, self.v41_roles.kv_owner_layer
            )
        # The name the compressed cache's attention metadata is published under:
        # this layer's own, or its source's.
        self.compressed_metadata_prefix = self.compressed_cache_prefix or prefix
        self._static_forward_context = compilation_config.static_forward_context

    @staticmethod
    def _q_padded_scratch_device_index(device: torch.device) -> int:
        if device.index is not None:
            return int(device.index)
        if device.type == "cuda":
            return int(torch.cuda.current_device())
        return -1

    @classmethod
    def _reserve_q_padded_scratch_buffer(
        cls,
        num_tokens: int,
        padded_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        ubatch_id: int,
    ) -> torch.Tensor:
        key = (
            device.type,
            cls._q_padded_scratch_device_index(device),
            ubatch_id,
            dtype,
            padded_heads,
            head_dim,
        )
        scratch = cls._q_padded_scratch_by_key.get(key)
        if scratch is not None and scratch.shape[0] >= num_tokens:
            return scratch

        old_scratch = cls._q_padded_scratch_by_key.pop(key, None)
        if old_scratch is not None:
            del old_scratch
            torch.accelerator.empty_cache()

        scratch = torch.empty(
            (num_tokens, padded_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        cls._q_padded_scratch_by_key[key] = scratch
        return scratch

    def _get_q_padded_scratch(self, q: torch.Tensor) -> torch.Tensor:
        # dtype is part of the buffer cache key, and reserve_profile_scratch
        # reserves under model_config.dtype while this reads under the runtime
        # q.dtype. If they ever diverge the profile run warms a buffer nobody
        # reads and the real one is allocated lazily afterwards -- which is
        # exactly the post-profiling OOM the reservation exists to prevent
        # (jasl/vllm#26). Keep the assumption loud rather than implicit.
        assert q.dtype == self._q_padded_scratch_dtype, (
            f"q dtype {q.dtype} differs from the reserved scratch dtype "
            f"{self._q_padded_scratch_dtype}; the profile-run reservation would "
            "not be reused."
        )
        num_tokens = q.shape[0]
        reserved_tokens = max(num_tokens, int(self.max_num_batched_tokens))
        scratch = self._reserve_q_padded_scratch_buffer(
            reserved_tokens,
            int(self.padded_heads),
            int(self.head_dim),
            q.dtype,
            q.device,
            dbo_current_ubatch_id(),
        )
        return scratch[:num_tokens]

    def _qnorm_rope_can_write_inplace(self) -> bool:
        return self.n_local_heads == self.padded_heads

    def reserve_profile_scratch(self) -> None:
        if self.kv_cache_torch_dtype != torch.uint8:
            return
        if self._qnorm_rope_can_write_inplace():
            return
        device = self.q_norm.weight.device
        if device.type not in ("cuda", "xpu"):
            return
        for ubatch_id in range(self._q_padded_scratch_num_ubatches):
            self._reserve_q_padded_scratch_buffer(
                max(1, int(self.max_num_batched_tokens)),
                int(self.padded_heads),
                int(self.head_dim),
                self._q_padded_scratch_dtype,
                device,
                ubatch_id,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Keep the attention input preparation in the captured graph. Only the
        # sparse indexer and MLA attention run in the eager break below.
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self._run_parallel_input_projections(hidden_states)
        )
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )

        self._prepare_and_attn_fn(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
        )
        o = o_padded[:, : self.n_local_heads, :]

        # Inverse-RoPE + wo_a + wo_b output projection (platform-specific).
        return self._o_proj(o, positions)

    @eager_break_during_capture
    def _prepare_and_attn_eager(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        o_padded: torch.Tensor,
    ) -> None:
        """Wide eager region: the whole of ``_prepare_and_attn`` runs eagerly.

        The nested ``_sparse_indexer_and_attn`` break runs inline, since
        ``add_eager`` clears ``_capturing`` before invoking this.
        """
        self._prepare_and_attn(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
        )

    def _prepare_and_attn(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        o_padded: torch.Tensor,
    ) -> None:
        """Attention input preparation followed by the sparse indexer and MLA.

        Only the latter runs in the eager break.
        """
        attn_metadata = get_forward_context().attn_metadata
        indexer = self.indexer
        compressor = self.compressor
        aux_streams = self.aux_stream_list

        def project_query_and_cache_kv() -> torch.Tensor:
            q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
            return self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        index_q: torch.Tensor | None = None
        index_q_scale: torch.Tensor | None = None
        index_weights_out: torch.Tensor | None = None

        # Keep Q projection and KV insertion on the default stream. The indexer
        # and MLA compressor use aux streams 0 and 1; aux 2 is internal to the
        # indexer. ROCm runs the same work sequentially without aux streams.
        if indexer is not None and (
            self.v41_roles is not None and not self.v41_roles.owns_index_keys
        ):
            # A V4.1 index-only source: it scores against the index keys a KV
            # source published and compresses nothing of its own, so there is no
            # compressor here to overlap with.
            q = project_query_and_cache_kv()
            indexer_inputs = indexer(
                hidden_states,
                qr,
                None,
                indexer_weights,
                positions,
                self.indexer_rotary_emb,
            )
        elif indexer is not None and self.v41_roles is not None:
            # A V4.1 index-key owner: the indexer derives its keys from the
            # latent this compressor writes into the state cache, so the
            # compressor has to be joined before the indexer runs -- the two
            # cannot overlap the way V4's can (its indexer has a compressor of
            # its own and derives nothing from this one).
            assert compressor is not None
            aux_stream = aux_streams[0] if aux_streams is not None else None
            q, _ = maybe_execute_in_parallel(
                project_query_and_cache_kv,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
            indexer_inputs = indexer(
                hidden_states,
                qr,
                None,
                indexer_weights,
                positions,
                self.indexer_rotary_emb,
            )
        elif indexer is not None:
            assert compressor is not None
            q, (indexer_inputs, _) = execute_in_parallel(
                project_query_and_cache_kv,
                [
                    lambda: indexer(
                        hidden_states,
                        qr,
                        indexer_kv_score,
                        indexer_weights,
                        positions,
                        self.indexer_rotary_emb,
                    ),
                    lambda: compressor(kv_score, positions, self.rotary_emb),
                ],
                self.ln_events[0],
                [self.ln_events[1], self.ln_events[2]],
                [aux_streams[0], aux_streams[1]] if aux_streams is not None else None,
                enable=aux_streams is not None,
            )
        elif compressor is not None:
            aux_stream = aux_streams[0] if aux_streams is not None else None
            q, _ = maybe_execute_in_parallel(
                project_query_and_cache_kv,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
        else:
            q = project_query_and_cache_kv()

        if indexer is not None:
            index_q, index_q_scale, index_weights_out = indexer_inputs

        self._sparse_indexer_and_attn(
            hidden_states,
            index_q,
            index_q_scale,
            index_weights_out,
            q,
            kv,
            positions,
            o_padded,
        )

    def _fused_wqa_wkv_gemm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Override point: the ROCm layer preshuffles this weight in place, so
        # it cannot go through fused_wqa_wkv directly.
        # MergedColumnParallelLinear returns (output, bias); bias is None.
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        return qr_kv

    def _run_parallel_input_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            assert len(aux_streams) >= 3
            aux_streams = aux_streams[:3]

        # fused_wqa_wkv (heaviest) on default; the three lighter input GEMMs
        # on aux streams 0..2 when their owning module exists. ln_events[0]
        # is the fan-out start event; ln_events[1..3] are per-aux done events.
        # On ROCm, aux_streams is None and execute_in_parallel runs serially.
        aux_fns: list[Callable[[], Any] | None] = [None, None, None]

        if self.compressor is not None:
            # Local ref so the closure keeps a non-None type for mypy.
            compressor = self.compressor

            def compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[0] = compressor_kv_score

        if self.indexer is not None:
            indexer = self.indexer

            def indexer_weights_proj() -> torch.Tensor:
                # ReplicatedLinear returns (output, bias); bias is None.
                weights, _ = indexer.weights_proj(hidden_states)
                return weights

            aux_fns[1] = indexer_weights_proj
            if indexer.compressor is not None:
                # V4: the indexer's own compressor projects the hidden state to
                # its latent here. V4.1 has no such projection -- the keys come
                # off the *attention* compressor's latent instead -- so there is
                # nothing for this stream to do.
                def indexer_compressor_kv_score() -> torch.Tensor:
                    return torch.mm(
                        hidden_states,
                        indexer.compressor.fused_wkv_wgate.weight.T,
                        out_dtype=torch.float32,
                    )

                aux_fns[2] = indexer_compressor_kv_score

        qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
            lambda: self._fused_wqa_wkv_gemm(hidden_states),
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:4],
            aux_streams,
            enable=hidden_states.shape[0]
            <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        )

        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    @eager_break_during_capture
    def _sparse_indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor | None,
        index_q_scale: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        if self.indexer is not None and index_q is not None:
            assert index_weights is not None
            q_quant = (index_q, index_q_scale) if index_q_scale is not None else index_q
            self.indexer.indexer_op(
                hidden_states,
                q_quant,
                None,
                index_weights,
            )

        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        self.forward_mqa(q, kv, positions, out)

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        if not isinstance(attn_metadata, dict):
            # Profile run: kernel doesn't fire; produce a padded tensor so
            # downstream FlashMLA gets the right shape.
            if self.kv_cache_torch_dtype == torch.uint8:
                if self._qnorm_rope_can_write_inplace():
                    return q
                return self._get_q_padded_scratch(q)
            if self.n_local_heads < self.padded_heads:
                return F.pad(
                    q,
                    (0, 0, 0, self.padded_heads - self.n_local_heads),
                    value=0.0,
                )
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        # The fused insert ops require int64 position_ids; the runner's positions
        # buffer is already int64, so no cast is needed.
        assert positions.dtype == torch.int64
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        cache_dtype = swa_kv_cache.dtype

        # kv is unchanged; attention reads kv solely via swa_kv_cache.
        if cache_dtype == torch.uint8:
            # fp8_ds_mla UE8M0 paged path. Horizontally fused:
            #   Q side:  per-head RMSNorm (no weight) + GPT-J RoPE, zero-filling
            #            the padding head slots; the kernel allocates and returns
            #            the padded q tensor.
            #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert.
            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
            if self._qnorm_rope_can_write_inplace():
                q_out = q
            elif self.eager_scratch_pool is not None:
                q_out = self.eager_scratch_pool.q_out(q.shape[0])
                torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out(
                    q,
                    kv,
                    q_out,
                    swa_kv_cache_2d,
                    swa_metadata.slot_mapping,
                    positions,
                    cos_sin_cache,
                    self.padded_heads,
                    self.eps,
                    swa_metadata.block_size,
                )
                return q_out
            else:
                q_out = self._get_q_padded_scratch(q)
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q,
                kv,
                q_out,
                swa_kv_cache_2d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.eps,
                swa_metadata.block_size,
            )
            return q_out

        # Plain-row path: the [num_blocks, block_size, 512] cache stores the KV
        # row in its element dtype (no Q padding). bf16 rewrites q in place;
        # per-tensor fp8 writes a separately-allocated fp8 q and quantizes the
        # KV row.
        block_size = swa_metadata.block_size
        assert swa_kv_cache.shape[1:] == (block_size, self.head_dim)
        swa_kv_cache_3d = swa_kv_cache
        if cache_dtype == torch.bfloat16:
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
                q,
                kv,
                swa_kv_cache_3d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.eps,
                block_size,
            )
            return q

        # per-tensor fp8 (torch.float8_e4m3fn)
        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
            q,
            kv,
            q_fp8,
            swa_kv_cache_3d,
            swa_metadata.slot_mapping,
            positions,
            cos_sin_cache,
            self._flashinfer_fp8_kv_scale,
            self._flashinfer_fp8_q_scale_inv,
            self.eps,
            block_size,
        )
        return q_fp8

    def _global_topk_output_buffers(
        self, topk_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.compress_ratio != 4 or self.eager_scratch_pool is None:
            return None
        return self.eager_scratch_pool.global_topk_outputs(topk_indices)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def compressed_kv_cache(self) -> torch.Tensor:
        """The paged compressed-KV cache this layer reads.

        Its own for every V4 layer and for V4.1's KV sources; its source's for
        the V4.1 layers in between. Looked up by prefix rather than captured at
        construction because the worker binds the tensors afterwards.
        """
        if self.compressed_cache_prefix is None:
            return self.kv_cache
        return self._static_forward_context[self.compressed_cache_prefix].kv_cache

    @property
    def compresses(self) -> bool:
        """Whether this layer attends over compressed positions at all.

        Not `compress_ratio > 1`: V4.1 reads a raw 1 as one latent per token,
        i.e. compressed, where V4 reads it as uncompressed. The operational
        ratio is clamped to >= 1 on both, so it cannot answer this on its own.
        """
        if self.v41_roles is None:
            return self.compress_ratio > 1
        return self.v41_roles.compression_enabled

    @property
    def uses_indexer_topk(self) -> bool:
        """Whether the compressed positions are the indexer's top-k.

        True for V4's ratio-4 layers and for every compressed V4.1 layer --
        ratios 1 and 2 are both indexer-driven there. V4's ratio-128 layers are
        the exception: their sparse set is decided when the metadata is built,
        so they read the C128A fields instead of `topk_indices_buffer`.

        The buffer itself is one tensor on the model, shared by every layer
        (`DeepseekV4Model.topk_indices_buffer`), which is what lets V4.1's
        layers read a top-k that an index-only source layer published.
        """
        return self.compresses and (
            self.v41_roles is not None or self.compress_ratio == 4
        )

    def tile_sched_layer_type(self) -> str:
        """Which FlashMLA tile-scheduler plan this layer's decode needs.

        Named type, not ratio: the plan carries the layer type's topk /
        extra_topk / page-block size, and V4.1's two compressed ratios have
        distinct geometry from each other and from V4's c4a.
        """
        from vllm.v1.attention.backends.mla.sparse_swa import (
            _LAYER_TYPE_C1A,
            _LAYER_TYPE_C2A,
            _LAYER_TYPE_C4A,
            _LAYER_TYPE_C128A,
            _LAYER_TYPE_SWAONLY,
        )

        if self.v41_roles is not None:
            if not self.v41_roles.compression_enabled:
                return _LAYER_TYPE_SWAONLY
            if self.v41_roles.raw_compress_ratio == 2:
                return _LAYER_TYPE_C2A
            return _LAYER_TYPE_C1A
        if self.compress_ratio <= 1:
            return _LAYER_TYPE_SWAONLY
        if self.compress_ratio == 4:
            return _LAYER_TYPE_C4A
        if self.compress_ratio == 128:
            return _LAYER_TYPE_C128A
        raise ValueError(
            f"Unsupported compress_ratio={self.compress_ratio}; "
            "expected 1, 4, or 128."
        )

    def compressed_cache_and_metadata(self, attn_metadata) -> tuple[Any, Any]:
        """The paged compressed cache this layer attends over, and its metadata.

        `(None, None)` for a layer that does not compress at all -- the
        SWA-only case. `attn_metadata` is the per-step dict keyed by module
        prefix, and only the owner of a compressed cache is in a KV-cache group
        and so has an entry of its own; the V4.1 layers reading it find the
        owner's.
        """
        if not self.compresses:
            return None, None
        return (
            self.compressed_kv_cache(),
            attn_metadata[self.compressed_metadata_prefix],
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # V4.1 compresses at ratio 1 as well, so the operational ratio is not
        # what decides this -- whether the layer compresses at all is.
        if not self.compresses:
            # SWA part only. Allocated separately as DeepseekV4SWACache.
            return None
        if self.compressed_cache_prefix is not None:
            # Reads its source's paged cache. Declaring a spec here would
            # allocate a second copy that nothing ever writes -- and being in
            # no KV-cache group is fine, because the group it needs to be in is
            # already the owner's, and it finds it by prefix.
            return None
        # fp8_ds_mla is a UE8M0 block-scaled uint8 layout and needs 576B
        # alignment; plain bf16 / per-tensor fp8 rows use natural element-size
        # pages.
        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            tokens_per_state=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token;
            # head_size stays semantic (512).
            state_content_bytes=584 if uses_fp8_ds_mla_layout else None,
        )


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # tokens_per_state=1 for V3.2, >1 for DeepseekV4; same cache layout.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            tokens_per_state=self.compress_ratio,
            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4IndexerBackend


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
        eager_scratch_pool: "DeepseekV4EagerScratchPool | None" = None,
        v41_roles: V41LayerRoles | None = None,
        attention_head_dim: int = 0,
        latent_compressor: "DeepseekCompressor | None" = None,
        index_cache_owner_prefix: str = "",
        candidate_source_prefix: str = "",
        candidate_topk_blocks: int = 0,
        candidate_block_size: int = 0,
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.v41_roles = v41_roles
        self.candidate_source_prefix = candidate_source_prefix
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        # V4's indexer always owns its keys: its own compressor derives them and
        # writes the cache. V4.1 splits the index sources -- the four that are
        # also KV sources (2/8/14/20) carry `wk`/`k_norm` and own an index-key
        # cache, the other four (24/28/32/36) score against the newest owner's
        # keys and carry neither.
        self.owns_k = True if v41_roles is None else v41_roles.owns_index_keys
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        self.compress_ratio = compress_ratio
        self.use_fp4_kv = dsa_indexer_uses_fp4(vllm_config)
        logger.info_once(
            "Using %s indexer cache for Lightning Indexer.",
            "MXFP4" if self.use_fp4_kv else "FP8",
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        # V4.1's index keys, in place of V4's indexer-local compressor: `wk`
        # projects the *attention* compressor's latent (512 wide) down to the
        # index head dim and `k_norm` normalizes it, which is the vendor's
        # `k = k_norm(wk(latent))`. Only a key owner has these tensors.
        self.attention_head_dim = attention_head_dim
        self.wk = None
        self.k_norm = None
        if self.owns_k and v41_roles is not None:
            assert attention_head_dim > 0, (
                "V4.1 index keys are projected from the attention head dim; "
                "the attention layer has to pass it in."
            )
            self.wk = ReplicatedLinear(
                attention_head_dim,
                self.head_dim,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.wk",
            )
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        # The attention compressor, when this layer owns one. Its state cache is
        # where the latent comes from and its `norm` is the weight that normed
        # it; V4's indexer has a compressor of its own below instead.
        self.latent_compressor = latent_compressor
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer
        self.eager_scratch_pool = eager_scratch_pool

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        if self.use_fp4_kv:
            # MXFP4 stores two values per byte plus one UE8M0 byte per 32 values.
            # head_dim bytes = 64 packed values + 4 UE8M0 scales = 68.
            k_cache_head_dim = self.head_dim // 2 + self.head_dim // MXFP4_BLOCK_SIZE
        else:
            # NOTE(yifan): FP8 indexer cache uses the same layout as V3.2:
            # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
            k_cache_head_dim = (
                self.head_dim + self.head_dim // self.quant_block_size * 4
            )
        self.use_pcp = vllm_config.parallel_config.prefill_context_parallel_size > 1
        self.k_cache: DeepseekV4IndexerCache
        if v41_roles is not None and not self.owns_k:
            # Scores against the keys a KV source published, so it reads that
            # layer's cache rather than getting one of its own -- the V4.1
            # layers in between a source and its consumers do the same for the
            # compressed KV cache (`compressed_kv_cache`). A cache module of its
            # own would declare a spec and get a paged tensor nothing ever
            # writes, so the module itself is the thing to share, not a name.
            registry = vllm_config.compilation_config.static_forward_context
            owner_key = f"{index_cache_owner_prefix}.indexer.k_cache"
            if owner_key not in registry:
                raise ValueError(
                    f"DeepSeek-V4.1: {prefix} scores against the index keys of "
                    f"{index_cache_owner_prefix or 'an unnamed layer'}, but that "
                    f"layer's index cache ({owner_key!r}) has not been built. "
                    "Layers are constructed in order, so this means the owner "
                    "is not an index-key owner at all."
                )
            self.k_cache = cast("DeepseekV4IndexerCache", registry[owner_key])
        else:
            self.k_cache = DeepseekV4IndexerCache(
                head_dim=k_cache_head_dim,
                dtype=torch.uint8,
                prefix=f"{prefix}.k_cache",
                cache_config=cache_config,
                compress_ratio=self.compress_ratio,
            )
        # V4's indexer compressor (its own wkv/wgate over the hidden state, its
        # own state cache) does not exist in V4.1: the checkpoint carries no
        # such tensors, and the keys are derived from the attention compressor's
        # latent instead. Building it would both fail to load and write the
        # wrong keys.
        self.compressor = None
        if v41_roles is None:
            self.compressor = DeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.k_cache.prefix,
                use_fp4_cache=self.use_fp4_kv,
                eager_scratch_pool=eager_scratch_pool,
            )

        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            # V4's nested compressor writes the key cache; V4.1 derives the keys
            # in `_v41_write_index_keys` below, which also has to work for the
            # short-context shortcut where the op is skipped but the keys are
            # still owed to every later step.
            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
            compress_ratio=self.compress_ratio,
            is_candidate_source=(
                False if v41_roles is None else v41_roles.is_candidate_source
            ),
            candidate_source_prefix=candidate_source_prefix,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
        )

        # None on ROCm — maybe_execute_in_parallel falls back to sequential.
        self.aux_stream = aux_stream
        self.ln_events: list[torch.cuda.Event] = [
            torch.cuda.Event(),
            torch.cuda.Event(),
        ]

    def _v41_write_index_keys(
        self,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        attn_metadata: dict[str, Any],
    ) -> torch.Tensor:
        """Derive and store this layer's index keys (V4.1 key owners).

        `latent` -> `wk` -> `k_norm` -> RoPE at the group's position -> quantized
        insert into the index key cache, which is the vendor's `Indexer.forward`
        key path. Returns the keys, for tests and callers that want them.

        This is the one place where V4.1's keys are produced, and it runs on
        every call that touches tokens -- including the short-context shortcut,
        where the top-k is trivial but the cache still has to grow, or every
        later decode step would score against keys that are not there.

        The attention compressor must have run already: the latent is gathered
        out of the state cache it writes (`_v41_latent_from_state_cache`), which
        is why `_prepare_and_attn` joins it before the indexer for these layers
        instead of overlapping the two.
        """
        compressor = self.latent_compressor
        assert compressor is not None, (
            f"{self.prefix} owns index keys but has no attention compressor to "
            "derive them from; a V4.1 index-key owner is a KV source."
        )
        assert self.wk is not None and self.k_norm is not None
        state_metadata = cast(Any, attn_metadata[compressor.state_cache.prefix])
        index_metadata = cast(Any, attn_metadata[self.k_cache.prefix])
        slot_mapping = index_metadata.slot_mapping
        # Spec decode may pad the batch past the tokens the metadata covers (the
        # sparse indexer op truncates for the same reason).
        num_tokens = slot_mapping.shape[0]
        if self.use_pcp:
            num_tokens //= get_pcp_group().world_size
        assert num_tokens <= positions.shape[0], (
            f"index key metadata covers {num_tokens} tokens but only "
            f"{positions.shape[0]} positions were passed"
        )
        positions = positions[:num_tokens]

        latent = _v41_latent_from_state_cache(
            compressor.state_cache.kv_cache,
            state_metadata.block_table,
            state_metadata.token_to_req_indices,
            positions,
            self.compress_ratio,
            compressor.norm.weight,
            compressor.rms_norm_eps,
        )
        k, _ = self.wk(latent.to(self.wk.weight.dtype))
        k = self.k_norm(k)
        # A latent stands for the first token of its group, so group `g` is
        # roped at position `g * ratio` -- the vendor's
        # `freqs_cis[start_pos + 1 - ratio]` for a completed group.
        group_positions = (positions // self.compress_ratio) * self.compress_ratio
        k = _v41_index_rope(
            k, rotary_emb.cos_sin_cache, group_positions, self.rope_dim
        )
        k, cache_slot_mapping = maybe_gather_indexer_k(
            k, slot_mapping, index_metadata.num_decode_tokens, self.use_pcp
        )
        # -1 slots (a token that does not end a group, at ratio 2) are skipped
        # by the kernel, which is what keeps the derivation's per-token shape
        # composable with a per-compressed-position cache.
        ops.indexer_k_quant_and_cache(
            k,
            self.k_cache.kv_cache,
            cache_slot_mapping,
            self.quant_block_size,
            self.scale_fmt,
        )
        return k

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor | None,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        compressor = self.compressor

        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            indexer_metadata = cast(Any, attn_metadata[self.k_cache.prefix])
            if (
                indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens
                and not torch.cuda.is_current_stream_capturing()
            ):
                # candidates num smaller than topk, every candidate is selected
                # but we still need to build k cache
                if compressor is not None:
                    compressor(compressed_kv_score, positions, rotary_emb)
                elif self.owns_k:
                    self._v41_write_index_keys(positions, rotary_emb, attn_metadata)
                assert self.topk_indices_buffer is not None
                num_tokens = (
                    indexer_metadata.num_decode_tokens
                    + indexer_metadata.num_prefill_tokens
                )
                if num_tokens > 0:
                    _fill_short_context_topk_indices[(num_tokens,)](
                        self.topk_indices_buffer,
                        positions,
                        TOP_K=self.topk_tokens,
                        COMPRESS_RATIO=self.compress_ratio,
                        PADDED_TOP_K=triton.next_power_of_2(self.topk_tokens),
                        num_warps=8,
                    )
                return None, None, None

        def wq_b_and_q_quant():
            # ReplicatedLinear returns (output, bias); bias is None.
            q, _ = self.wq_b(qr)
            q = q.view(-1, self.n_head, self.head_dim)
            return fused_indexer_q_rope_quant(
                positions,
                q,
                rotary_emb.cos_sin_cache,
                indexer_weights,
                self.softmax_scale,
                self.n_head**-0.5,
                use_fp4=self.use_fp4_kv,
            )

        if compressor is not None:
            # compressor returns None and writes K to the indexer KV cache; the
            # join orders that write before indexer_op (skip_k_cache_insert=True).
            (q_quant, weights), _ = maybe_execute_in_parallel(
                wq_b_and_q_quant,
                lambda: compressor(compressed_kv_score, positions, rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
        else:
            # V4.1: the keys are derived from the attention compressor's state
            # cache, which the caller has already joined -- it cannot overlap
            # the indexer the way V4's own compressor does, but the query side
            # has nothing to wait for either. A dummy run has no metadata and
            # nothing to write (the compressor returns early for the same
            # reason).
            if self.owns_k and isinstance(attn_metadata, dict):
                self._v41_write_index_keys(positions, rotary_emb, attn_metadata)
            q_quant, weights = wq_b_and_q_quant()
        if isinstance(q_quant, tuple):
            q, q_scale = q_quant
        else:
            q, q_scale = q_quant, None
        return q, q_scale, weights
