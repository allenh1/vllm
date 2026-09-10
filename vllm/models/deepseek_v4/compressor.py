# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
    compress_norm_rope_store_two_stage_triton,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    save_partial_states,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


def _prefer_two_stage_compressor() -> bool:
    # Platforms that favor the triton variant of two-stage compressor split.
    # Currently only tested on ROCm
    return current_platform.is_rocm()


#: The compress ratios the head=512 CuTe DSL store kernel implements. It has
#: exactly two specialisations -- C4-with-overlap and C128 -- and rejects
#: everything else (`nvidia/ops/sparse_attn_compress_cutedsl.py`), so V4.1's
#: ratios 1 and 2 have no CuTe DSL kernel to route to.
_CUTEDSL_COMPRESS_RATIOS: tuple[int, ...] = (4, 128)


def _compressor_store_uses_cutedsl(
    head_dim: int, compress_ratio: int, is_cuda: bool
) -> bool:
    """Whether the fused compress → norm → RoPE → store step runs on CuTe DSL.

    Only CUDA's head=512 path has CuTe DSL kernels, and those cover exactly the
    ratios V4 uses. Everything else -- the indexer's head=128, non-CUDA
    platforms, and V4.1's ratios 1 and 2 -- takes the generic triton launcher
    `compress_norm_rope_store_triton`, whose gather is a plain
    `tl.arange(0, COMPRESS_RATIO)` softmax with no per-ratio code in it.
    """
    return is_cuda and head_dim == 512 and compress_ratio in _CUTEDSL_COMPRESS_RATIOS


def _checkpoint_has_ape(config) -> bool:
    """Whether this checkpoint carries a compressor `ape` tensor.

    V4 does: `save_partial_states` adds `ape[position % compress_ratio]` to the
    gate score before the softmax. V4.1 does not -- the vendor's `Compressor`
    has no positional term and the released index has no such tensor -- and the
    two are told apart by the config, so this defers to the same V4.1 predicate
    `attention.py` uses rather than re-deriving it.
    """
    from vllm.models.deepseek_v4.attention import is_deepseek_v41

    return not is_deepseek_v41(config)


def _get_c128_boundary(metadata: CommonAttentionMetadata) -> bool | None:
    starts = metadata._num_computed_tokens_cpu
    if starts is None:
        return None

    starts_list = starts.tolist()
    query_start_loc = metadata.query_start_loc_cpu.tolist()
    return any(
        start % 128 + query_start_loc[i + 1] - query_start_loc[i] >= 128
        for i, start in enumerate(starts_list)
    )


class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]
    num_decode_tokens: int | None = None
    c128_boundary: bool | None = None


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.block_size = mla_spec.block_size

        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )
        num_decode_tokens = None
        if _prefer_two_stage_compressor():
            _, _, num_decode_tokens, _ = split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=1
            )
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
            num_decode_tokens=num_decode_tokens,
            c128_boundary=(
                _get_c128_boundary(common_attn_metadata)
                if self.block_size == 8
                else None
            ),
        )


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [1, 2, 4, 128]
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # V4.1's ratios 1 and 2 have no overlap either, so their state row is
        # the same 512*2*4 = 4096 B as C128's and they keep C128's block size:
        # block_size * state_row = 32768 B for every ratio.
        # The sliding window is the compressor's lookback -- the `coff*ratio`
        # rows a boundary gathers -- and not a policy knob: the state cache
        # manager retains the block holding the oldest of those rows exactly.
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio in (1, 2, 128):
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # fp8_ds_mla is the UE8M0 paged layout and needs 576B alignment. Plain
        # full-cache rows share state pages with contiguous KV pages, so padding
        # would break page matching.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


class DeepseekCompressor(nn.Module):
    """DeepSeek V4 KV/score compressor.

    Owns the linear / norm / state-cache / ape state and the shared forward
    prologue (kv/score split, save_partial_states launch). The
    compress → norm → RoPE → store step is dispatched to a triton kernel
    (``compress_norm_rope_store_triton``) by default, except for the head_dim=512
    path on CUDA, which uses the cutedsl kernel
    (``compress_norm_rope_store_cutedsl``) for better performance.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
        eager_scratch_pool: "DeepseekV4EagerScratchPool | None" = None,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache
        self.eager_scratch_pool = eager_scratch_pool

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        # Only C4's boundary window straddles two compression blocks and so
        # gathers 2 rows; every other ratio compresses a single row per block.
        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap
        # Ratio 1 has nothing to weigh: the gate score is a single row, whose
        # softmax is exactly 1, so the pooled latent passes through unweighted
        # and the vendor ships no `wgate` for it.
        self.has_gate = compress_ratio > 1

        # The head=512 C128 deep gather uses the two-stage compressor, which
        # needs an fp32 scratch [max_batched, 512] for the intermediate
        # compressed_kv. C128 is the only ratio it has been exercised at, and
        # V4.1's ratios 1 and 2 are served by the generic launcher anyway.
        # Currently only tested on ROCm
        self._use_two_stage_fused_compressor = (
            _prefer_two_stage_compressor()
            and head_dim == 512
            and not self.overlap
            and compress_ratio == 128
        )
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self._compress_scratch: torch.Tensor | None = None
        if self._use_two_stage_fused_compressor:
            self._compress_scratch = torch.empty(
                self.max_num_batched_tokens,
                self.head_dim,
                dtype=torch.float32,
                device=self.device,
            )

        state_dtype = torch.float32
        # V4.1 dropped the absolute positional embedding: its `Compressor` has no
        # positional term and the released index carries no such tensor. The term is
        # added to the gate score *before* the softmax, so leaving the parameter out
        # is exact, not an approximation of zero -- but the parameter must be absent
        # rather than left uninitialised, which the loader would faithfully leave as
        # garbage. Keep a zero row for `ape`-taking kernels (`save_partial_states`).
        self.ape: nn.Parameter | None = (
            nn.Parameter(
                torch.empty(
                    (compress_ratio, self.coff * self.head_dim),
                    dtype=state_dtype,
                    device=self.device,
                ),
                requires_grad=False,
            )
            if _checkpoint_has_ape(config)
            else None
        )
        self._zero_ape = (
            None
            if self.ape is not None
            else torch.zeros(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            )
        )

        # A single output shard for ratio 1: there is no gate to fuse, so the
        # mapper's `compressor.wgate -> shard 1` entry simply never fires, and
        # `compressor.wkv` loads into shard 0 as it does for every other ratio.
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim] * (2 if self.has_gate else 1),
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        # Every ratio gets a state cache, ratio 1 included. This is not the
        # vendor's `kv_state`/`score_state` (which ratio 1 has no use for and
        # does not allocate): in this fork the projection's output only ever
        # reaches the store kernel *through* the state -- `save_partial_states`
        # stages kv/score there, and the compress → norm → RoPE → store kernels
        # gather the group's rows back out of it, one row per token for ratio 1
        # -- so a ratio-1 layer without one would have nothing to read, no
        # `CompressorMetadata` (the metadata builder is bound to this layer's
        # spec) and no bound cache tensor. Its state row is the same 512*2*4 B
        # as C128's, so it shares C128's block size and the 32768 B page the
        # sharing invariant needs.
        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
        elif self.head_dim == 128:
            if use_fp4_cache:
                self._quant_block = MXFP4_BLOCK_SIZE
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        if self.has_gate:
            kv, score = kv_score.split(
                [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
            )
        else:
            # No gate: the projection *is* the latent. Feeding it as the score too
            # costs nothing -- a one-row softmax is exactly 1, so the kernel's
            # weighted sum returns the latent unchanged -- and keeps the kernel
            # signature free of a special case.
            kv = score = kv_score

        # Get the metadata and handle dummy profiling run.
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2
        pdl_kwargs = (
            {}
            if current_platform.is_rocm() or current_platform.is_xpu()
            else {"launch_pdl": False}
        )

        # Store the KV and score (with fused APE addition) in the state.
        # NOTE: PDL is disabled — both this kernel and the compress kernels
        # below depend on preceding kernel outputs (kv/score from the cublas
        # GEMM; state_cache from this kernel) but neither emits/waits on PDL
        # grid dependency primitives, so launch_pdl=True caused a
        # read-after-write race and non-deterministic output.
        save_partial_states(
            kv=kv,
            score=score,
            # `save_partial_states` takes the positional term as a mandatory operand
            # and adds it to the score; a checkpoint without one adds its zeros.
            ape=self.ape if self.ape is not None else self._zero_ape,
            positions=positions,
            state_cache=state_cache,
            slot_mapping=slot_mapping,
            block_size=block_size,
            state_width=state_width,
            compress_ratio=self.compress_ratio,
            pdl_kwargs=pdl_kwargs,
        )

        # full graph cannot branch on per-step CPU metadata after capture
        if (
            current_platform.is_cuda()
            and self.head_dim == 512
            and self.compress_ratio == 128
            and forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
            and state_metadata.c128_boundary is False
        ):
            return

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        # Plain-row V4 reads a contiguous bf16 / per-tensor fp8 cache row; the
        # fp8_ds_mla path uses the UE8M0 paged uint8 layout.
        store_full_kv = self.head_dim == 512 and kv_cache.dtype != torch.uint8
        store_full_fp8 = kv_cache.dtype == torch.float8_e4m3fn
        fp8_scale = (
            getattr(k_cache_layer, "_flashinfer_fp8_kv_scale", None)
            if store_full_fp8
            else None
        )

        # cutedsl (head=512) accepts the full-cache flags; triton (indexer/AMD)
        # does not, so the two callables have different signatures.
        compress_norm_rope_store_fn: Any
        if _compressor_store_uses_cutedsl(
            self.head_dim, self.compress_ratio, current_platform.is_cuda()
        ):
            from .nvidia.ops.sparse_attn_compress_cutedsl import (
                compress_norm_rope_store_cutedsl,
            )

            # Both the fp8_ds_mla layout and the plain full-cache layout go
            # through cutedsl here. The full-cache flags are consumed only here.
            compress_norm_rope_store_fn = compress_norm_rope_store_cutedsl
            extra_kwargs: dict[str, Any] = dict(
                store_full_kv=store_full_kv,
                store_full_fp8=store_full_fp8,
                fp8_scale=fp8_scale,
            )
            if not self.overlap and self.eager_scratch_pool is not None:
                extra_kwargs["compress_scratch"] = (
                    self.eager_scratch_pool.compressor_scratch(num_actual)
                )
        elif self._use_two_stage_fused_compressor:
            # head=512 C128 (no overlap): two-pass split compressor on the
            # prefill suffix, single-pass on the decode prefix.
            assert state_metadata.num_decode_tokens is not None
            compress_norm_rope_store_fn = compress_norm_rope_store_two_stage_triton
            extra_kwargs = {
                "num_decode_tokens": state_metadata.num_decode_tokens,
                "compress_scratch": self._compress_scratch,
            }
        else:
            # Indexer path (head_dim == 128), non-CUDA GPUs (AMD, XPU, etc.)
            # and V4.1's ratios 1 and 2, whose head=512 gather takes the
            # generic single-pass launcher.
            if store_full_kv and self.compress_ratio not in _CUTEDSL_COMPRESS_RATIOS:
                raise ValueError(
                    f"compress_ratio={self.compress_ratio} has no kernel that "
                    "writes the plain head=512 cache layout: only the CuTe DSL "
                    "store kernels do, and they cover ratios "
                    f"{_CUTEDSL_COMPRESS_RATIOS}. Run with the fp8_ds_mla cache "
                    "layout instead."
                )
            compress_norm_rope_store_fn = compress_norm_rope_store_triton
            extra_kwargs = {}

        compress_norm_rope_store_fn(
            state_cache=state_cache,
            num_actual=num_actual,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            pdl_kwargs=pdl_kwargs,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            use_fp4_cache=self.use_fp4_cache,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
            token_stride=self._token_stride,
            scale_dim=self._scale_dim,
            **extra_kwargs,
        )
