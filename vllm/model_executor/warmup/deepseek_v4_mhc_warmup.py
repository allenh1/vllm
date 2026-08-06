# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek-V4 / DSpark mHC TileLang kernels before serving requests.

The mHC pre/fused TileLang kernels (``mhc_pre_big_fuse_with_norm_tilelang``,
``mhc_pre_big_fuse_broadcast_with_norm_tilelang``, ``mhc_fused_tilelang``,
``mhc_post_tilelang``, ``mhc_post_hc_head_tilelang`` and friends) are the DSv4
hot path: both the target model and the DSpark draft model run them for every
decoder layer on every token, and they are the only TileLang kernels that still
JIT mid-inference on the SM120 serve.

* ``num_tokens`` is a *dynamic* TileLang grid dimension, so the kernel source is
  shared across token counts; the only token-dependent term in the compile key
  is the split-k factor ``n_splits``, re-derived per launch from ``num_tokens``
  via ``compute_num_split`` (``tilelang.py``: ``n_splits =
  min(n_sms // grid_size, cdiv(k, 64) // 4)`` with ``grid_size =
  cdiv(num_tokens, 64)``). Every token count whose grid lands in a new band
  therefore triggers a fresh kernel compile inside the first prefill/decode
  step that reaches it.
* The prior DeepSeek-V4 mHC warmup (removed by commit 92feb0efdc) never ran on
  NVIDIA/SM120 at all: it dispatched through AMD-only ``hc_pre``/``hc_post``/
  ``hc_head_op`` layer methods that do not exist on the SM120 model, warmed the
  wrong (no-norm / non-broadcast) variants, and only covered the handful of
  ``n_splits`` values its fixed token ladder mapped to.

This module drives the *real* NVIDIA wrapper entry points with the model's own
``fn``/``scale``/``base``/eps parameters across a minimal ``num_tokens`` ladder
that exhaustively covers every distinct ``n_splits`` band the scheduler can
emit for the live device, so no chunk size or decode batch JITs later.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_fused_post_pre_tilelang_reuse_residual,
    mhc_post_hc_head_tilelang,
    mhc_post_mean_hc_head_tilelang,
    mhc_pre_broadcast_tilelang,
    mhc_pre_tilelang,
)
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)

# ``_mhc_fused_post_pre_tilelang_impl`` routes ``num_tokens <= 16`` onto the
# small-FMA kernel ``mhc_fused_tilelang`` (``tile_n``/``n_splits`` chosen by
# size bucket: (2, 8) below 8 tokens, (3, 4) for 8..16) and everything above it
# onto ``mhc_post_tilelang`` + the big-fuse pre kernel. These representatives
# cover both small-path buckets and the first big-path band (grid == 1).
_SMALL_FMA_WARMUP_TOKENS = (1, 8)
_BIG_PATH_MIN_TOKENS = 17
# Block-m used by the SM120 tf32 HC prenorm GEMM grid sizing.
_WARMUP_GRID_BLOCK_M = 64


def _n_splits_token_ladder(
    max_tokens: int,
    *,
    k: int,
    n_sms: int | None = None,
) -> tuple[int, ...]:
    """Return one ``num_tokens`` per distinct ``n_splits`` band up to max_tokens.

    ``n_splits`` is a step function of the launch grid ``grid_size =
    cdiv(num_tokens, 64)`` (see ``compute_num_split`` in
    ``tilelang_kernels.py``). Iterate the grids in ascending order and retain
    the smallest token count that maps to each distinct ``n_splits`` value, so
    the ladder size equals the number of bands (≈ 20 for a typical SM120
    ``n_sms`` and the DSv4 ``k = hc_mult * hidden_size``), not the number of
    token counts.
    """
    if max_tokens < _BIG_PATH_MIN_TOKENS:
        return ()
    if n_sms is None:
        n_sms = torch.cuda.get_device_properties(0).multi_processor_count
    split_cap = max(1, cdiv(k, _WARMUP_GRID_BLOCK_M) // 4)
    representatives: dict[int, int] = {}
    for grid in range(1, cdiv(max_tokens, _WARMUP_GRID_BLOCK_M) + 1):
        n_splits = max(1, min(n_sms // grid, split_cap))
        if n_splits in representatives:
            continue
        # Smallest token count with this grid that still runs the big-fuse
        # path (grid 1 maps to token 17, every other grid to >= 65 > 16).
        representatives[n_splits] = max(
            _BIG_PATH_MIN_TOKENS, (grid - 1) * _WARMUP_GRID_BLOCK_M + 1
        )
    return tuple(sorted(representatives.values()))


def _warmup_broadcast_pre(
    layer: torch.nn.Module,
    token_sizes: tuple[int, ...],
) -> None:
    """Drive the 2D first-layer mHC pre (broadcast-with-norm variant)."""
    fn_broadcast = getattr(layer, "hc_attn_fn_broadcast", None)
    if fn_broadcast is None:
        return
    hidden_size = int(layer.hidden_size)
    device = layer.hc_attn_fn.device
    for num_tokens in token_sizes:
        x = torch.zeros(
            (num_tokens, hidden_size), dtype=torch.bfloat16, device=device
        )
        mhc_pre_broadcast_tilelang(
            x,
            layer.hc_attn_fn.data,
            layer.hc_attn_scale.data,
            layer.hc_attn_base.data,
            float(layer.rms_norm_eps),
            float(layer.hc_eps),
            float(layer.hc_eps),
            float(getattr(layer, "hc_post_alpha", 2.0)),
            int(layer.hc_sinkhorn_iters),
            norm_weight=layer.attn_norm.weight.data,
            norm_eps=float(layer.attn_norm.variance_epsilon),
            fn_broadcast=fn_broadcast.data,
        )


def _warmup_fused_layer(
    layer: torch.nn.Module,
    token_sizes: tuple[int, ...],
) -> None:
    """Drive the per-layer mHC post + fused pre (big-fuse with norm) path.

    The fused wrapper internally re-derives ``n_splits`` from ``num_tokens``
    and runs ``mhc_post_tilelang`` + the tf32 HC prenorm GEMM + the big-fuse
    pre kernel, so one ladder covers all three kernel families for both the
    attention and FFN ``fn`` streams exactly as the model calls them.
    """
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    rms_eps = float(layer.rms_norm_eps)
    hc_eps = float(layer.hc_eps)
    post_alpha = float(getattr(layer, "hc_post_alpha", 2.0))
    sinkhorn_repeat = int(layer.hc_sinkhorn_iters)

    for fn, scale, base, norm in (
        (layer.hc_attn_fn.data, layer.hc_attn_scale.data,
         layer.hc_attn_base.data, layer.attn_norm),
        (layer.hc_ffn_fn.data, layer.hc_ffn_scale.data,
         layer.hc_ffn_base.data, layer.ffn_norm),
    ):
        norm_weight = norm.weight.data
        norm_eps = float(norm.variance_epsilon)
        for num_tokens in token_sizes:
            x = torch.zeros(
                (num_tokens, hidden_size), dtype=torch.bfloat16, device=device
            )
            residual = torch.zeros(
                (num_tokens, hc_mult, hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
            post_mix = torch.zeros(
                (num_tokens, hc_mult, 1), dtype=torch.float32, device=device
            )
            res_mix = torch.zeros(
                (num_tokens, hc_mult, hc_mult),
                dtype=torch.float32,
                device=device,
            )
            mhc_fused_post_pre_tilelang_reuse_residual(
                x,
                residual,
                post_mix,
                res_mix,
                fn,
                scale,
                base,
                rms_eps,
                hc_eps,
                hc_eps,
                post_alpha,
                sinkhorn_repeat,
                n_splits=1,
                tile_n=1,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )


def _warmup_head(
    model: torch.nn.Module,
    num_tokens: int,
) -> None:
    """Drive the fused post+hc_head kernels with the model's head params.

    ``mhc_post_hc_head_tilelang`` / ``mhc_post_mean_hc_head_tilelang`` have
    flat compile keys (hidden_size, hc_mult, eps values), so one call covers
    every batch.
    """
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(model.hc_mult)
    device = model.hc_head_fn.device
    x = torch.zeros(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=device
    )
    residual = torch.zeros(
        (num_tokens, hc_mult, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    post_mix = torch.zeros(
        (num_tokens, hc_mult, 1), dtype=torch.float32, device=device
    )
    res_mix = torch.zeros(
        (num_tokens, hc_mult, hc_mult),
        dtype=torch.float32,
        device=device,
    )
    kwargs = dict(
        x=x,
        residual=residual,
        post_layer_mix=post_mix,
        comb_res_mix=res_mix,
        fn=model.hc_head_fn.data,
        hc_scale=model.hc_head_scale.data,
        hc_base=model.hc_head_base.data,
        rms_eps=float(model.rms_norm_eps),
        hc_eps=float(model.hc_eps),
    )
    mhc_post_hc_head_tilelang(**kwargs)
    mhc_post_mean_hc_head_tilelang(**kwargs)


def _iter_target_layers(model: torch.nn.Module):
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4DecoderLayer

    return (
        module
        for module in model.modules()
        if isinstance(module, DeepseekV4DecoderLayer)
        and hasattr(module, "hc_attn_fn")
    )


def _iter_draft_layers(draft_model: torch.nn.Module):
    from vllm.models.deepseek_v4.nvidia.dspark import DeepSeekV4DSparkLayer

    return (
        module
        for module in draft_model.modules()
        if isinstance(module, DeepSeekV4DSparkLayer)
        and hasattr(module, "hc_attn_fn")
    )


@torch.inference_mode()
def deepseek_v4_mhc_warmup(
    model: torch.nn.Module,
    *,
    draft_model: torch.nn.Module | None = None,
    max_tokens: int,
) -> None:
    """Force-compile every mHC TileLang kernel shape the scheduler can emit.

    Args:
        model: The DeepSeek-V4 target model (decoder layers must expose the
            ``hc_attn_fn``/``hc_ffn_fn`` + norm parameters consumed by the
            NVIDIA ``mhc_*_tilelang`` wrappers).
        draft_model: Optional DSpark/MTP draft model whose layers use the same
            wrappers; driven here too so draft decode cannot JIT.
        max_tokens: Upper bound on per-step token counts (usually
            ``scheduler_config.max_num_batched_tokens``). ``num_tokens`` is a
            dynamic grid dimension, so this only bounds the ``n_splits`` sweep.

    Returns immediately (no-op) for models without the mHC layer parameters and
    never raises: a failure leaves the kernels uncompiled and the first request
    JITs them in-inference, which the caller logs at warning level.
    """
    try:
        layers = list(_iter_target_layers(model))
        if not layers:
            return
        first = layers[0]
        if first.hc_attn_fn.device.type != "cuda":
            return
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            return

        # Head parameters live on the nested DeepseekV4Model (the top-level
        # DeepseekV4ForCausalLM wrapper hides them under ``.model``).
        from vllm.models.deepseek_v4.nvidia.model import DeepseekV4Model

        head_model = model
        if getattr(model, "hc_head_fn", None) is None:
            for module in model.modules():
                if isinstance(module, DeepseekV4Model) and getattr(
                    module, "hc_head_fn", None
                ) is not None:
                    head_model = module
                    break

        hidden_size = int(first.hidden_size)
        hc_mult = int(first.hc_mult)
        fused_ladder = _n_splits_token_ladder(max_tokens, k=hc_mult * hidden_size)
        broadcast_ladder = _n_splits_token_ladder(max_tokens, k=hidden_size)
        small_ladder = tuple(
            t for t in _SMALL_FMA_WARMUP_TOKENS if t <= max_tokens
        )
        if not fused_ladder and not broadcast_ladder and not small_ladder:
            return

        logger.info(
            "Warming up DeepSeek-V4 mHC TileLang kernels (n_splits bands "
            "fused=%s broadcast=%s decode=%s).",
            list(fused_ladder),
            list(broadcast_ladder),
            list(small_ladder),
        )

        # The wrappers' compile keys depend only on the layer geometry (hidden
        # size / eps / n_splits band), not on per-layer weight data, so the
        # first target and first draft layer cover every layer's kernels; the
        # TileLang JIT cache dedupes the rest.
        _warmup_broadcast_pre(first, broadcast_ladder)
        _warmup_fused_layer(first, fused_ladder + small_ladder)
        if draft_model is not None:
            for layer in _iter_draft_layers(draft_model):
                _warmup_fused_layer(layer, fused_ladder + small_ladder)
                break

        # Standalone 3D mhc_pre path (PP rank > 0 first layer): same big-fuse
        # with-norm kernel, one representative token suffices.
        if fused_ladder:
            residual = torch.zeros(
                (min(fused_ladder), hc_mult, hidden_size),
                dtype=torch.bfloat16,
                device=first.hc_attn_fn.device,
            )
            mhc_pre_tilelang(
                residual,
                first.hc_attn_fn.data,
                first.hc_attn_scale.data,
                first.hc_attn_base.data,
                float(first.rms_norm_eps),
                float(first.hc_eps),
                float(first.hc_eps),
                float(getattr(first, "hc_post_alpha", 2.0)),
                int(first.hc_sinkhorn_iters),
                norm_weight=first.attn_norm.weight.data,
                norm_eps=float(first.attn_norm.variance_epsilon),
            )

        rep_token = min(fused_ladder, default=_BIG_PATH_MIN_TOKENS)
        if getattr(head_model, "hc_head_fn", None) is not None:
            _warmup_head(head_model, rep_token)
        if draft_model is not None and getattr(
            draft_model, "hc_head_fn", None
        ) is not None:
            _warmup_head(draft_model, rep_token)

        torch.accelerator.synchronize()
    except Exception as exc:  # noqa: BLE001 - warmup must never break startup
        logger.warning(
            "DeepSeek-V4 mHC TileLang warmup skipped after error "
            "(the first prefill/decode may JIT the mHC kernels in-inference): "
            "%s",
            exc,
        )
