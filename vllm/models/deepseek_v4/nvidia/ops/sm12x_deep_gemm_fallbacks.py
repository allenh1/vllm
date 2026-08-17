# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x fallback implementations for DeepGEMM-only interfaces."""

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_SM120_MQA_LOGITS_MAX_SCORE_BYTES = 64 * 1024 * 1024
_SM120_MQA_TRITON_TOPK_MAX_LOGITS_BYTES = 512 * 1024 * 1024
_SM120_MQA_TRITON_CHUNKED_TOPK_CHUNK_SIZE = 32768
_SM120_PAGED_MQA_TOPK_CHUNK_SIZE = 8192


def _top_k_per_row_prefill_op():
    try:
        from vllm import _custom_ops as _custom_ops  # noqa: F401

        return torch.ops._C.top_k_per_row_prefill
    except (AttributeError, ImportError, RuntimeError):
        return None


def _fp8_mqa_logits_head_chunk_size(
    seq_len: int,
    seq_len_kv: int,
    num_heads: int,
) -> int:
    # The SM120 torch path is used on long prefill paths where materializing
    # [head_chunk, M, N] scores can otherwise allocate multiple GiB. Keep the
    # transient score tensor bounded, while still using larger head chunks for
    # short prompts where they are faster.
    score_elems_per_head = max(1, seq_len * seq_len_kv)
    max_heads = _SM120_MQA_LOGITS_MAX_SCORE_BYTES // (score_elems_per_head * 4)
    return max(1, min(8, num_heads, max_heads))


def _fp8_mqa_logits_k_chunk_size(
    seq_len: int,
    seq_len_kv: int,
    head_chunk_size: int,
) -> int:
    score_elems_per_key = max(1, seq_len * head_chunk_size)
    max_keys = _SM120_MQA_LOGITS_MAX_SCORE_BYTES // (score_elems_per_key * 4)
    return max(1, min(seq_len_kv, max_keys))


def _fp8_mqa_logits_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 MQA logits torch path only supports FP8 Q")

    k_values, k_scales = kv
    k_f32 = k_values.to(torch.float32)
    k_f32.mul_(k_scales.reshape(-1, 1).to(torch.float32))
    k_t = k_f32.transpose(0, 1).contiguous()

    seq_len, num_heads, _ = q_values.shape
    seq_len_kv = k_f32.shape[0]
    logits = torch.zeros(
        (seq_len, seq_len_kv), device=q_values.device, dtype=torch.float32
    )
    head_chunk_size = _fp8_mqa_logits_head_chunk_size(seq_len, seq_len_kv, num_heads)

    for head_start in range(0, num_heads, head_chunk_size):
        head_end = min(head_start + head_chunk_size, num_heads)
        q_chunk = q_values[:, head_start:head_end, :].to(torch.float32)
        q_chunk = q_chunk.transpose(0, 1).contiguous()
        head_weights = weights[:, head_start:head_end].transpose(0, 1).unsqueeze(-1)
        k_chunk_size = _fp8_mqa_logits_k_chunk_size(
            seq_len, seq_len_kv, head_end - head_start
        )
        for k_start in range(0, seq_len_kv, k_chunk_size):
            k_end = min(k_start + k_chunk_size, seq_len_kv)
            scores = torch.matmul(q_chunk, k_t[:, k_start:k_end])
            scores.relu_()
            scores.mul_(head_weights)
            logits[:, k_start:k_end].add_(
                scores[0] if scores.shape[0] == 1 else scores.sum(dim=0)
            )

    if clean_logits:
        offsets = torch.arange(seq_len_kv, device=q_values.device)
        valid = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
            offsets[None, :] < cu_seqlen_ke[:, None]
        )
        logits = logits.masked_fill(~valid, float("-inf"))

    return logits


def _fp8_mqa_logits_topk_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_tokens: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 MQA top-k torch path only supports FP8 Q")

    k_values, k_scales = kv
    k_f32 = k_values.to(torch.float32)
    k_f32.mul_(k_scales.reshape(-1, 1).to(torch.float32))
    k_t = k_f32.transpose(0, 1).contiguous()

    seq_len, num_heads, _ = q_values.shape
    seq_len_kv = k_f32.shape[0]
    if out is None:
        out = torch.empty(
            (seq_len, topk_tokens), device=q_values.device, dtype=torch.int32
        )
    else:
        assert out.shape == (seq_len, topk_tokens)
        assert out.dtype == torch.int32
    out.fill_(-1)

    best_values = torch.full(
        (seq_len, topk_tokens),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )
    head_chunk_size = _fp8_mqa_logits_head_chunk_size(seq_len, seq_len_kv, num_heads)
    k_chunk_size = _fp8_mqa_logits_k_chunk_size(seq_len, seq_len_kv, head_chunk_size)
    max_chunk_topk = min(topk_tokens, k_chunk_size)
    chunk_values_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_indices_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int64,
    )
    chunk_indices_i32 = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    candidate_values = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    candidate_indices = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    next_best_values = torch.empty_like(best_values)
    selected = torch.empty(
        (seq_len, topk_tokens),
        device=q_values.device,
        dtype=torch.int64,
    )

    for k_start in range(0, seq_len_kv, k_chunk_size):
        k_end = min(k_start + k_chunk_size, seq_len_kv)
        chunk_logits = torch.zeros(
            (seq_len, k_end - k_start),
            device=q_values.device,
            dtype=torch.float32,
        )
        for head_start in range(0, num_heads, head_chunk_size):
            head_end = min(head_start + head_chunk_size, num_heads)
            q_chunk = q_values[:, head_start:head_end, :].to(torch.float32)
            q_chunk = q_chunk.transpose(0, 1).contiguous()
            head_weights = weights[:, head_start:head_end].transpose(0, 1).unsqueeze(-1)
            scores = torch.matmul(q_chunk, k_t[:, k_start:k_end])
            scores.relu_()
            scores.mul_(head_weights)
            chunk_logits.add_(scores[0] if scores.shape[0] == 1 else scores.sum(dim=0))

        offsets = torch.arange(k_start, k_end, device=q_values.device)
        valid = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
            offsets[None, :] < cu_seqlen_ke[:, None]
        )
        chunk_logits.masked_fill_(~valid, float("-inf"))

        chunk_topk = min(topk_tokens, k_end - k_start)
        chunk_values = chunk_values_buf[:, :chunk_topk]
        chunk_indices = chunk_indices_buf[:, :chunk_topk]
        torch.topk(chunk_logits, chunk_topk, dim=1, out=(chunk_values, chunk_indices))
        chunk_indices_out = chunk_indices_i32[:, :chunk_topk]
        chunk_indices_out.copy_(chunk_indices)
        chunk_indices_out.add_(k_start)

        candidate_cols = topk_tokens + chunk_topk
        candidate_values_view = candidate_values[:, :candidate_cols]
        candidate_indices_view = candidate_indices[:, :candidate_cols]
        candidate_values_view[:, :topk_tokens].copy_(best_values)
        candidate_values_view[:, topk_tokens:candidate_cols].copy_(chunk_values)
        candidate_indices_view[:, :topk_tokens].copy_(out)
        candidate_indices_view[:, topk_tokens:candidate_cols].copy_(chunk_indices_out)
        torch.topk(
            candidate_values_view,
            topk_tokens,
            dim=1,
            out=(next_best_values, selected),
        )
        torch.gather(candidate_indices_view, 1, selected, out=out)
        best_values, next_best_values = next_best_values, best_values
        out.masked_fill_(~torch.isfinite(best_values), -1)

    return out


def _fp8_mqa_logits_topk_triton(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    q_values, q_scale = q
    k_values, _ = kv
    if not (q_scale is None and q_values.dim() == 3 and k_values.dim() == 2):
        return False

    logits_bytes = q_values.shape[0] * k_values.shape[0] * torch.float32.itemsize
    if logits_bytes > _SM120_MQA_TRITON_TOPK_MAX_LOGITS_BYTES:
        return False

    from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
        fp8_mqa_logits_triton,
    )

    logits = fp8_mqa_logits_triton(q_values, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
    topk_tokens = out.shape[1]
    select_k = min(topk_tokens, logits.shape[1])
    out.fill_(-1)
    if select_k == 0:
        return True

    selected = out[:, :select_k]
    topk_op = _top_k_per_row_prefill_op()
    if topk_op is not None:
        # top_k_per_row_prefill writes its output as a contiguous [M, select_k]
        # buffer (it is given the logits strides, not the output strides). When
        # select_k < out.shape[1] -- i.e. the compressed-KV count is below the
        # topk width, which happens for short prompts and the early queries of
        # long prompts -- out[:, :select_k] is non-contiguous (row stride =
        # out.shape[1]), so writing it as contiguous silently corrupts later
        # rows and drops their top-k (all -1). Hand the op a fresh contiguous
        # buffer and copy back. The buffer is left uninitialized (new_empty)
        # rather than copied (.contiguous()) because the op overwrites every
        # element, so copying the placeholder -1s in would be wasted work.
        work = (
            selected if selected.is_contiguous() else selected.new_empty(selected.shape)
        )
        topk_op(
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            work,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            select_k,
        )
        work.add_(cu_seqlen_ks[:, None])
        valid = (work >= cu_seqlen_ks[:, None]) & (work < cu_seqlen_ke[:, None])
        work.masked_fill_(~valid, -1)
        if work is not selected:
            selected.copy_(work)
    else:
        values, indices = torch.topk(logits, select_k, dim=1)
        selected.copy_(indices.to(torch.int32))
        selected.masked_fill_(~torch.isfinite(values), -1)
    return True


def _fp8_mqa_logits_topk_triton_chunked(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    q_values, q_scale = q
    k_values, k_scales = kv
    if not (q_scale is None and q_values.dim() == 3 and k_values.dim() == 2):
        return False

    from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
        fp8_mqa_logits_triton,
    )

    seq_len = q_values.shape[0]
    seq_len_kv = k_values.shape[0]
    topk_tokens = out.shape[1]
    out.fill_(-1)
    if seq_len == 0 or seq_len_kv == 0 or topk_tokens == 0:
        return True

    chunk_size = max(1, _SM120_MQA_TRITON_CHUNKED_TOPK_CHUNK_SIZE)
    best_values = torch.full(
        (seq_len, topk_tokens),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )
    max_chunk_topk = min(topk_tokens, chunk_size)
    chunk_values_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_indices_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int64,
    )
    chunk_indices_i32 = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    candidate_values = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    candidate_indices = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    next_best_values = torch.empty_like(best_values)
    selected = torch.empty(
        (seq_len, topk_tokens),
        device=q_values.device,
        dtype=torch.int64,
    )

    for k_start in range(0, seq_len_kv, chunk_size):
        k_end = min(k_start + chunk_size, seq_len_kv)
        local_width = k_end - k_start
        local_ks = torch.clamp(cu_seqlen_ks - k_start, min=0, max=local_width)
        local_ke = torch.clamp(cu_seqlen_ke - k_start, min=0, max=local_width)
        chunk_logits = fp8_mqa_logits_triton(
            q_values,
            (k_values[k_start:k_end], k_scales[k_start:k_end]),
            weights,
            local_ks,
            local_ke,
        )
        chunk_topk = min(topk_tokens, local_width)
        chunk_values = chunk_values_buf[:, :chunk_topk]
        chunk_indices = chunk_indices_buf[:, :chunk_topk]
        torch.topk(chunk_logits, chunk_topk, dim=1, out=(chunk_values, chunk_indices))
        chunk_indices_out = chunk_indices_i32[:, :chunk_topk]
        chunk_indices_out.copy_(chunk_indices)
        chunk_indices_out.add_(k_start)

        candidate_cols = topk_tokens + chunk_topk
        candidate_values_view = candidate_values[:, :candidate_cols]
        candidate_indices_view = candidate_indices[:, :candidate_cols]
        candidate_values_view[:, :topk_tokens].copy_(best_values)
        candidate_values_view[:, topk_tokens:candidate_cols].copy_(chunk_values)
        candidate_indices_view[:, :topk_tokens].copy_(out)
        candidate_indices_view[:, topk_tokens:candidate_cols].copy_(chunk_indices_out)
        torch.topk(
            candidate_values_view,
            topk_tokens,
            dim=1,
            out=(next_best_values, selected),
        )
        torch.gather(candidate_indices_view, 1, selected, out=out)
        best_values, next_best_values = next_best_values, best_values
        out.masked_fill_(~torch.isfinite(best_values), -1)

    return True


def fp8_fp4_mqa_topk_indices(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
) -> bool:
    """Write SM120 FP8 MQA top-k indices without materializing full logits."""
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and q[1] is None
    ):
        return False
    if _fp8_mqa_logits_topk_triton(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
    ):
        return True
    if _fp8_mqa_logits_topk_triton_chunked(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
    ):
        return True
    _fp8_mqa_logits_topk_torch(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices.shape[1],
        out=topk_indices,
    )
    return True


def _fp8_mqa_logits_sm12x(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    q_values, q_scale = q
    if clean_logits and q_scale is None and q_values.dim() == 3 and kv[0].dim() == 2:
        from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
            fp8_mqa_logits_triton,
        )

        return fp8_mqa_logits_triton(q_values, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
    return _fp8_mqa_logits_torch(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
    )


def _fp8_paged_mqa_logits_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 paged MQA torch path only supports FP8 Q")

    batch_size, next_n, num_heads, head_dim = q_values.shape
    head_dim_with_scale = kv_cache.shape[-1]
    assert head_dim_with_scale > head_dim
    assert weights.shape == (batch_size * next_n, num_heads)
    assert context_lens.shape == (batch_size, next_n)

    from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
        _view_packed_fp8_paged_mqa_kv_cache,
    )

    kv_values, kv_scales = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_kv, _, _ = kv_values.shape
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )

    q_f32 = q_values.float()
    score_bytes = _SM120_MQA_LOGITS_MAX_SCORE_BYTES
    max_tokens_per_chunk = max(1, score_bytes // max(1, num_heads * 4))
    token_offsets_cache: dict[int, torch.Tensor] = {}

    for batch_idx in range(batch_size):
        for next_idx in range(next_n):
            row = batch_idx * next_n + next_idx
            context_len = int(context_lens[batch_idx, next_idx].item())
            if context_len <= 0:
                continue

            q_row = q_f32[batch_idx, next_idx]
            row_weights = weights[row]
            for token_start in range(0, context_len, max_tokens_per_chunk):
                token_end = min(context_len, token_start + max_tokens_per_chunk)
                chunk_len = token_end - token_start
                token_offsets = token_offsets_cache.get(chunk_len)
                if token_offsets is None or token_offsets.device != q_values.device:
                    token_offsets = torch.arange(
                        chunk_len, device=q_values.device, dtype=torch.long
                    )
                    token_offsets_cache[chunk_len] = token_offsets
                token_ids = token_start + token_offsets
                logical_blocks = token_ids // block_kv
                token_in_block = token_ids - logical_blocks * block_kv
                physical_blocks = block_tables[batch_idx, logical_blocks]
                kv_chunk = kv_values[physical_blocks, token_in_block, 0].float()
                scale_chunk = kv_scales[physical_blocks, token_in_block, 0].squeeze(-1)
                kv_chunk.mul_(scale_chunk[:, None])
                scores = torch.matmul(q_row, kv_chunk.T)
                scores.relu_()
                scores.mul_(row_weights[:, None])
                logits[row, token_start:token_end] = scores.sum(dim=0)

    return logits


def _fp8_paged_mqa_logits_sm12x(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    q_values, q_scale = q
    if (
        q_scale is None
        and q_values.dim() == 4
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[-1] == q_values.shape[-1] + 4
    ):
        from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
            fp8_paged_mqa_logits_triton,
        )

        return fp8_paged_mqa_logits_triton(
            q_values, kv_cache, weights, context_lens, block_tables, max_model_len
        )
    logger.warning_once(
        "SM12x paged-MQA falling back to the torch reference path "
        "(q_scale=%s, q.dim=%s, kv_cache.dtype=%s, kv_cache.shape[-1]=%s, "
        "q_values.shape[-1]=%s). This path is intended for correctness checks "
        "and is not graph-compatible; expect a large per-step latency.",
        "set" if q_scale is not None else "None",
        q_values.dim(),
        kv_cache.dtype,
        kv_cache.shape[-1] if kv_cache.dim() else None,
        q_values.shape[-1],
    )
    return _fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )


def fp8_fp4_paged_mqa_topk_indices(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    topk_indices: torch.Tensor,
) -> bool:
    """Write SM120 FP8 paged MQA top-k indices without full logits."""
    q_values, q_scale = q
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and q_scale is None
        and q_values.dim() == 4
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[-1] == q_values.shape[-1] + 4
    ):
        return False

    num_rows = q_values.shape[0] * q_values.shape[1]
    topk_tokens = topk_indices.shape[1]
    assert topk_indices.shape == (num_rows, topk_tokens)
    assert topk_indices.dtype == torch.int32
    topk_indices.fill_(-1)
    if num_rows == 0 or topk_tokens == 0 or max_model_len == 0:
        return True

    best_values = torch.full(
        (num_rows, topk_tokens),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_size = max(1, _SM120_PAGED_MQA_TOPK_CHUNK_SIZE)
    max_chunk_topk = min(topk_tokens, chunk_size)
    chunk_values_buf = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_indices_buf = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int64,
    )
    chunk_indices_i32 = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    candidate_values = torch.empty(
        (num_rows, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    candidate_indices = torch.empty(
        (num_rows, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    next_best_values = torch.empty_like(best_values)
    selected = torch.empty(
        (num_rows, topk_tokens),
        device=q_values.device,
        dtype=torch.int64,
    )

    from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
        fp8_paged_mqa_logits_triton,
    )

    for token_start in range(0, max_model_len, chunk_size):
        token_count = min(chunk_size, max_model_len - token_start)
        chunk_logits = fp8_paged_mqa_logits_triton(
            q_values,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
            token_start=token_start,
            token_count=token_count,
        )
        chunk_topk = min(topk_tokens, token_count)
        chunk_values = chunk_values_buf[:, :chunk_topk]
        chunk_indices = chunk_indices_buf[:, :chunk_topk]
        torch.topk(chunk_logits, chunk_topk, dim=1, out=(chunk_values, chunk_indices))
        chunk_indices_out = chunk_indices_i32[:, :chunk_topk]
        chunk_indices_out.copy_(chunk_indices)
        chunk_indices_out.add_(token_start)

        candidate_cols = topk_tokens + chunk_topk
        candidate_values_view = candidate_values[:, :candidate_cols]
        candidate_indices_view = candidate_indices[:, :candidate_cols]
        candidate_values_view[:, :topk_tokens].copy_(best_values)
        candidate_values_view[:, topk_tokens:candidate_cols].copy_(chunk_values)
        candidate_indices_view[:, :topk_tokens].copy_(topk_indices)
        candidate_indices_view[:, topk_tokens:candidate_cols].copy_(chunk_indices_out)
        torch.topk(
            candidate_values_view,
            topk_tokens,
            dim=1,
            out=(next_best_values, selected),
        )
        torch.gather(candidate_indices_view, 1, selected, out=topk_indices)
        best_values, next_best_values = next_best_values, best_values
        topk_indices.masked_fill_(~torch.isfinite(best_values), -1)

    return True


def _tf32_hc_prenorm_gemm_torch(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    """Portable SM12x HyperConnection prenorm GEMM fallback.

    DeepGEMM's split ABI only requires that downstream consumers recover the
    full result by summing over the split dimension. Keep the implementation
    simple by writing the full product to split zero and clearing the rest.
    """
    del num_split
    product = x.float() @ fn.float().T
    norm = x.float().square().sum(dim=-1)

    if out.dim() == 3:
        out.zero_()
        sqrsum.zero_()
        out[0].copy_(product)
        sqrsum[0].copy_(norm)
    else:
        out.copy_(product)
        sqrsum.copy_(norm)
    return out


def _tf32_hc_prenorm_gemm_sm12x(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    if out.dim() == 3 and sqrsum.dim() == 2:
        from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
            tf32_hc_prenorm_gemm_triton,
        )

        tf32_hc_prenorm_gemm_triton(x, fn, out, sqrsum, num_split)
        return out

    return _tf32_hc_prenorm_gemm_torch(x, fn, out, sqrsum, num_split)




# ---------------------------------------------------------------------------
# SM12x MegaMoE orchestration
# ---------------------------------------------------------------------------
# The fused tcgen05 MegaMoE kernel is SM100-only; GB10 (SM121) has no tcgen05
# or tensor memory. This fallback reproduces the MegaMoE dataflow on SM12x:
#
#   topk dispatch (one packed NCCL all-to-all across the EP group; the
#   symmetric-buffer transport requires intra-node NVLink and cannot work on
#   1-GPU-per-node clusters) -> sorted fused local compute (the native SM120
#   `sm120_fp8_fp4_mega_moe` kernel: L1 (FP8xFP8, pre-scaled weights) ->
#   swiglu + clamp -> per-64 FP8/UE8M0 activation quantization -> L2 with
#   fp32 scale folding -> reverse scatter by pair id) -> reverse all-to-all
#   -> topk-weighted scatter combine.
#
# All tensor shapes are static (capped by max_num_tokens * top_k) so the
# path stays cudagraph-capturable; rows beyond the real counts are
# neutralized (zero inputs -> zero partials -> zero contribution).

from vllm.triton_utils import tl, triton

_E2M1_DECODE_F32 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

_MEGA_MOE_ALIGN = 128  # DeepGEMM MGroupedContiguous M alignment
_MEGA_MOE_BLOCK_ROWS = 64


@triton.jit
def _mega_fold_x_sf_kernel(
    x_ptr, sf_ptr, out_ptr,
    H: tl.constexpr, CHUNK: tl.constexpr,
):
    """Fold the per-128 UE8M0 scales into FP8 x bytes.

    x: [rows, H] fp8 viewed as uint8; sf: [rows, H/128] int32 UE8M0 words.
    The fold is an exact e4m3 exponent shift (2^k), clamped to zero on
    underflow and to the e4m3 max (448) on overflow.
    """
    row = tl.program_id(0)
    for c in tl.static_range(H // CHUNK):
        offs = c * CHUNK + tl.arange(0, CHUNK)
        g = offs // 128
        b = (offs % 128) // 32
        sf = tl.load(sf_ptr + row * (H // 128) + g)
        k = ((sf >> (b * 8)) & 0xFF).to(tl.int32) - 127
        xb = tl.load(x_ptr + row * H + offs).to(tl.int32)
        field = (xb >> 3) & 0xF
        sign = xb & 0x80
        is_zero = (xb & 0x7F) == 0
        nf = field + k
        out_b = tl.where(
            is_zero, sign,
            tl.where(
                nf <= 0, sign,
                tl.where(
                    nf >= 15, sign | 0x7E,
                    (xb & 0x87) | ((nf & 0xF) << 3),
                ),
            ),
        )
        tl.store(out_ptr + row * H + offs, out_b.to(tl.uint8))


def _mega_moe_deinterleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    """Inverse of deep_gemm's `_interleave_weights`:
    [g0..7, u0..7, g8..15, u8..15, ...] -> [gate | up] along the N dim."""
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)
    g, n, *rest = t.shape
    half = n // 2
    paired = t.reshape(g, half // gran, 2, gran, *rest)
    gate = paired[:, :, 0].reshape(g, half, *rest)
    up = paired[:, :, 1].reshape(g, half, *rest)
    result = torch.cat([gate, up], dim=1).contiguous()
    return result.squeeze(0) if squeeze_group_dim else result


def _mega_moe_untranspose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    """Inverse of deep_gemm's `_transpose_sf_for_utccp` (reshape to the
    (32, 4) block form and transpose back)."""
    assert sf.dtype == torch.int and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)
    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 32, 4, packed_sf_k)
              .transpose(2, 3)
              .reshape(num_groups, mn, packed_sf_k))
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze_group_dim else result


def _mega_moe_decode_fp4(packed: torch.Tensor) -> torch.Tensor:
    """[..., K/2] int8 packed FP4 (lo nibble = even k) -> [..., K] fp32."""
    u = packed.to(torch.uint8).to(torch.int64)
    codes = torch.stack([u & 0xF, (u >> 4) & 0xF], dim=-1)
    codes = codes.reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return _E2M1_DECODE_F32.to(codes.device)[codes]


def _mega_moe_unpack_ue8m0(sf_packed: torch.Tensor) -> torch.Tensor:
    """[..., K/128] int32 UE8M0 words -> [..., K/32] fp32 powers of two."""
    bytes_ = sf_packed.contiguous().view(torch.uint8)
    return (bytes_.to(torch.int32) << 23).view(torch.float32).reshape(
        *sf_packed.shape[:-1], sf_packed.shape[-1] * 4
    )


_mega_moe_fused_weight_cache: dict[
    tuple[int, tuple], tuple[torch.Tensor, torch.Tensor]
] = {}


def _mega_moe_fused_weights(
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize the packed FP4 MegaMoE weights and fold the UE8M0 scales
    into FP8 for the SM120 fused kernel.

    Every E2M1 value times a power of two fits E4M3's mantissa exactly, so
    the fold is lossless (up to the |448| fp8 range clamp). The L1 result is
    returned in the fused kernel's [gate | up] N layout. Cached by weight
    data pointers (weights are fixed after load).
    """
    key = (
        l1_weights[0].data_ptr(), tuple(l1_weights[0].shape),
        l2_weights[0].data_ptr(), tuple(l2_weights[0].shape),
    )
    cached_ = _mega_moe_fused_weight_cache.get(key)
    if cached_ is not None:
        return cached_
    w13_packed = _mega_moe_deinterleave_weights(l1_weights[0].contiguous())
    # the L1 sf was transformed as utccp(interleave(raw)): invert utccp first
    # (the interleave and utccp reshapes do not commute)
    w13_sf = _mega_moe_deinterleave_weights(
        _mega_moe_untranspose_sf_for_utccp(l1_weights[1].contiguous())
    )
    w13_32 = _mega_moe_decode_fp4(w13_packed)
    w13_sf32 = _mega_moe_unpack_ue8m0(w13_sf)
    w13_fp8 = (w13_32 * w13_sf32.repeat_interleave(32, dim=-1)).to(torch.float8_e4m3fn)
    w2_packed = l2_weights[0].contiguous()
    w2_sf = _mega_moe_untranspose_sf_for_utccp(l2_weights[1].contiguous())
    w2_32 = _mega_moe_decode_fp4(w2_packed)
    w2_sf32 = _mega_moe_unpack_ue8m0(w2_sf)
    w2_fp8 = (w2_32 * w2_sf32.repeat_interleave(32, dim=-1)).to(torch.float8_e4m3fn)
    result = (w13_fp8, w2_fp8)
    _mega_moe_fused_weight_cache[key] = result
    return result


def _mega_moe_layout_and_gather(
    expert_local: torch.Tensor,  # [n] int64 local expert ids (unsorted)
    x_rows: torch.Tensor,        # [n, H] fp8
    x_sf_rows: torch.Tensor,     # [n, H/128] int32
    row_ids: torch.Tensor,       # [n] int32 ids to permute alongside rows
    num_local_experts: int,
    scratch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort rows by expert and build a padded grouped layout.

    Returns (starts, counts): the per-expert padded (aligned) cumulative row
    offsets and the real token counts. The scratch layout/a/a_sf/home tensors
    are filled in place: `home` carries each layout row's pair id (-1 for
    padding), which is the d2 slot the fused kernel scatters into.
    """
    n = expert_local.shape[0]
    order = torch.argsort(expert_local, stable=True)
    s_exp = expert_local[order]
    counts = torch.bincount(s_exp, minlength=num_local_experts)
    aligned = ((counts + _MEGA_MOE_ALIGN - 1) // _MEGA_MOE_ALIGN) * _MEGA_MOE_ALIGN
    starts = torch.cumsum(aligned, 0) - aligned
    real_starts = torch.cumsum(counts, 0) - counts
    cap = scratch["layout"].shape[0]
    torch._assert(starts[-1] + aligned[-1] <= cap, "MegaMoE layout cap exceeded")

    layout = scratch["layout"]
    layout.fill_(-1)
    r = torch.arange(cap, device=expert_local.device)
    group = torch.searchsorted(starts, r, right=True) - 1
    group_safe = group.clamp(min=0)
    valid = (group >= 0) & (r - starts[group_safe] < counts[group_safe])
    row_in_sorted = r - starts[group_safe]
    src_sorted = torch.where(
        valid,
        real_starts[group_safe] + row_in_sorted,
        torch.zeros_like(r),
    )
    layout.copy_(torch.where(valid, group, -1).to(torch.int32))
    src_rows = order[src_sorted]
    scratch["a"].copy_(x_rows[src_rows])
    scratch["a_sf"].copy_(x_sf_rows[src_rows])
    # the fused kernel scatters d2 by slot: layout row r writes d2[home[r]]
    home = scratch["home"]
    home.fill_(-1)
    home[valid] = row_ids[src_rows[valid]]
    return starts, counts


def _mega_moe_local_compute(
    x_rows: torch.Tensor,        # [n, H] fp8 (unsorted)
    x_sf_rows: torch.Tensor,     # [n, H/128] int32
    expert_local: torch.Tensor,  # [n] int64 local expert ids
    row_ids: torch.Tensor,       # [n] int32 ids to permute alongside rows
    l1_weights: tuple[torch.Tensor, torch.Tensor],  # (int8 [E,2I,H/2], int32 sf)
    l2_weights: tuple[torch.Tensor, torch.Tensor],  # (int8 [E,H,I/2], int32 sf)
    activation_clamp: float | None,
    scratch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused L1 -> swiglu -> quantize -> L2 for a local batch of (row, expert)
    pairs, via the native SM120 mega kernel. Returns (partial, row_ids_out)
    in the input row order."""
    from vllm.utils import deep_gemm

    H = x_rows.shape[1]
    E = l1_weights[0].shape[0]

    starts, counts = _mega_moe_layout_and_gather(
        expert_local, x_rows, x_sf_rows, row_ids, E, scratch,
    )
    cap = scratch["layout"].shape[0]

    # Fold the per-128 x scales into the fp8 bytes (the fused kernel takes
    # pre-scaled operands; SM121 has no block-scaled MMA).
    chunk = min(2048, triton.next_power_of_2(H))
    _mega_fold_x_sf_kernel[(cap,)](
        scratch["a"].view(torch.uint8), scratch["a_sf"],
        scratch["a_folded"].view(torch.uint8),
        H=H, CHUNK=chunk, num_warps=4,
    )
    w13_fp8, w2_fp8 = _mega_moe_fused_weights(l1_weights, l2_weights)
    deep_gemm.sm120_fp8_fp4_mega_moe(
        scratch["a_folded"], w13_fp8, w2_fp8,
        starts.to(torch.int32), counts.to(torch.int32),
        scratch["home"], scratch["d2"],
        acts=scratch["acts"], acts_sf=scratch["acts_sf"],
        activation_clamp=(
            activation_clamp if activation_clamp is not None else float("inf")
        ),
    )
    # d2 is slot-indexed: the j-th input row's partial lives at d2[row_ids[j]]
    d2o = scratch["d2"][row_ids]
    return d2o, row_ids


_mega_moe_scratch_cache: dict[tuple, dict] = {}


def _mega_moe_get_scratch(device, H: int, I: int, E: int, cap: int) -> dict:
    key = (device, H, I, E, cap)
    scratch = _mega_moe_scratch_cache.get(key)
    if scratch is None:
        # the fused kernel's acts scratch has one slot per SM
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        scratch = {
            "layout": torch.full((cap,), -1, dtype=torch.int32, device=device),
            "a": torch.zeros((cap, H), dtype=torch.float8_e4m3fn, device=device),
            "a_sf": torch.full(
                (cap, H // 128), 0x7F7F7F7F, dtype=torch.int32, device=device
            ),
            "a_folded": torch.zeros((cap, H), dtype=torch.float8_e4m3fn, device=device),
            "home": torch.full((cap,), -1, dtype=torch.int32, device=device),
            "d2": torch.zeros((cap, H), dtype=torch.bfloat16, device=device),
            "acts": torch.zeros(
                (num_sms * 64 * I,), dtype=torch.float8_e4m3fn, device=device
            ),
            "acts_sf": torch.zeros(
                (num_sms * 64 * (I // 128),), dtype=torch.int32, device=device
            ),
            "acc": torch.zeros((cap, H), dtype=torch.float32, device=device),
        }
        _mega_moe_scratch_cache[key] = scratch
    return scratch


def make_sm12x_mega_moe_buffer(
    num_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
):
    """Create the MegaMoE staging buffer for SM12x (single-rank layout)."""
    import types

    from vllm.utils import deep_gemm

    sizer = deep_gemm.get_mega_moe_symm_buffer_sizer()
    if sizer is None:
        raise RuntimeError("DeepGEMM MegaMoE buffer sizer is unavailable")
    num_bytes, slice_input_buffers = sizer(
        1, num_experts, num_max_tokens, num_topk, hidden, intermediate,
        "fp8xfp4", "swiglu", 0,
    )
    buf = torch.empty(num_bytes, dtype=torch.int8, device="cuda")
    buf.zero_()
    x, x_sf, topk_idx, topk_weights, *_ = slice_input_buffers(buf)
    # Keep a reference to the base allocation: the C++ slicer may return
    # views that do not hold the storage alive, and the allocator would
    # recycle the buffer under our feet.
    return types.SimpleNamespace(
        base_buf=buf,
        x=x,
        x_sf=x_sf,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_max_tokens_per_rank=num_max_tokens,
        num_experts=num_experts,
        num_topk=num_topk,
    )


def _mega_moe_ordered(valid: torch.Tensor) -> torch.Tensor:
    """Stable argsort of a bool mask putting valid rows first (static shapes)."""
    return torch.argsort(~valid, stable=True)


def fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    symm_buffer,
    activation_clamp: float | None = None,
    fast_math: bool = True,
) -> None:
    """SM12x orchestrated MegaMoE: EP token dispatch + grouped-gemm compute."""
    import torch.distributed as dist

    from vllm.distributed.parallel_state import get_ep_group
    from vllm.utils import deep_gemm

    num_tokens = y.shape[0]
    if num_tokens == 0:
        return
    H = y.shape[1]
    I = l1_weights[0].shape[1] // 2
    E_local = l1_weights[0].shape[0]
    top_k = symm_buffer.topk_idx.shape[1]
    max_tokens = symm_buffer.x.shape[0]

    x = symm_buffer.x[:num_tokens]
    x_sf = symm_buffer.x_sf[:num_tokens]
    topk_idx = symm_buffer.topk_idx[:num_tokens]
    topk_weights = symm_buffer.topk_weights[:num_tokens]

    group = get_ep_group().device_group
    ep_size = group.size()
    cap = max_tokens * top_k
    layout_cap = cap + _MEGA_MOE_ALIGN * E_local
    scratch = _mega_moe_get_scratch(y.device, H, I, E_local, layout_cap)

    if ep_size == 1:
        expert_local = topk_idx.reshape(-1).to(torch.int64)
        pair = torch.arange(num_tokens * top_k, device=y.device, dtype=torch.int32)
        token_of = pair // top_k
        partial, _ = _mega_moe_local_compute(
            x[token_of], x_sf[token_of], expert_local, pair,
            l1_weights, l2_weights, activation_clamp, scratch,
        )
        token = torch.arange(num_tokens, device=y.device).repeat_interleave(top_k)
        slot = torch.arange(top_k, device=y.device).repeat(num_tokens)
        w = topk_weights[token, slot]
        acc = scratch["acc"]
        acc.zero_()
        acc.index_add_(0, token, partial.float() * w.unsqueeze(-1))
        y.copy_(acc[:num_tokens].to(y.dtype))
        return

    # ---- forward dispatch: one packed all-to-all (static shapes) ----
    # payload bytes per row: x fp8 [H] + sf int32 [H/128] + expert int32 [1]
    # + pair id int32 [1] (the original (token, slot) pair index, needed by
    # the home combine to weight and scatter the returned partials).
    sf_off = H
    expert_off = sf_off + (H // 128) * 4
    pair_off = expert_off + 4
    pack_bytes = ((pair_off + 4 + 15) // 16) * 16
    n_pairs = num_tokens * top_k
    flat_ids = topk_idx.reshape(-1).to(torch.int64)
    target = flat_ids // E_local
    order = torch.argsort(target, stable=True)
    s_target = target[order]
    counts = torch.bincount(s_target, minlength=ep_size)
    starts_r = torch.cumsum(counts, 0) - counts

    send = torch.zeros((ep_size * cap, pack_bytes), dtype=torch.uint8, device=y.device)
    recv = torch.zeros((ep_size * cap, pack_bytes), dtype=torch.uint8, device=y.device)
    # neutral rows beyond the real pairs
    x_aug = torch.cat([x, torch.zeros(1, H, device=y.device, dtype=torch.float8_e4m3fn)])
    sf_aug = torch.cat([x_sf, torch.full((1, H // 128), 0x7F7F7F7F, device=y.device, dtype=torch.int32)])
    token_of = torch.arange(n_pairs, device=y.device) // top_k
    s_token = token_of[order]
    pos = s_target * cap + (torch.arange(n_pairs, device=y.device) - starts_r[s_target])
    send_x = send[:, :H].view(torch.float8_e4m3fn)
    send_sf = send[:, sf_off:expert_off].view(torch.int32)
    send_expert = send[:, expert_off:pair_off].view(torch.int32).reshape(-1)
    send_pair = send[:, pair_off: pair_off + 4].view(torch.int32).reshape(-1)
    send_x[pos] = x_aug[s_token]
    send_sf[pos] = sf_aug[s_token]
    send_expert[pos] = (flat_ids[order] - s_target * E_local).to(torch.int32)
    # order[p] is the original pair index of the p-th dispatched row
    send_pair[pos] = order.to(torch.int32)
    dist.all_to_all_single(recv, send, group=group)

    # ---- counts matrix exchange (static all_gather) ----
    counts_gather = [torch.empty_like(counts) for _ in range(ep_size)]
    dist.all_gather(counts_gather, counts, group=group)
    counts_matrix = torch.stack(counts_gather)  # [W, W]: rows s->r
    recv_counts = counts_matrix[:, group.rank()]  # [W]: from each source



    # ---- unpack recv rows: valid rows first (static shapes) ----
    # a rank can receive at most all n_pairs <= cap rows in total
    seg = torch.arange(ep_size * cap, device=y.device) // cap
    j = torch.arange(ep_size * cap, device=y.device) % cap
    valid = j < recv_counts[seg]
    rorder = _mega_moe_ordered(valid)
    rows = recv[rorder][:cap]
    # neutralize invalid rows -> zero partials. x and expert come from the
    # zeroed recv buffer (0 bytes = fp8 0.0 / expert 0) and pair ids are
    # masked below; only the sf needs an explicit fix-up because UE8M0 byte 0
    # is invalid on SM120 hardware.
    rx = rows[:, :H].view(torch.float8_e4m3fn).contiguous()
    rsf = rows[:, sf_off:expert_off].view(torch.int32).contiguous()
    rexp = rows[:, expert_off:pair_off].view(torch.int32).reshape(-1)
    rpair = rows[:, pair_off: pair_off + 4].view(torch.int32).reshape(-1)
    invalid_mask = ~valid[rorder][:cap]
    rsf[invalid_mask] = 0x7F7F7F7F
    rpair = torch.where(invalid_mask, torch.zeros_like(rpair), rpair)
    rexp64 = rexp.to(torch.int64).clamp(min=0, max=E_local - 1)
    # the original pair ids permute alongside the rows through local compute
    row_ids = rpair

    partial, partial_ids = _mega_moe_local_compute(
        rx, rsf, rexp64, row_ids, l1_weights, l2_weights,
        activation_clamp, scratch,
    )
    del row_ids

    # ---- reverse all-to-all: partials back to their source segment ----
    # scatter partials back into recv slot order (across ALL source
    # segments), then pack per-source segments
    inv_rorder = torch.empty_like(rorder)
    inv_rorder[rorder] = torch.arange(ep_size * cap, device=y.device)
    # partial rows follow the valid-first ordering of the recv buffer; pad
    # the tail (positions beyond the valid rows) with zeros and gather every
    # row back to its original (source, slot) position.
    partial_full = torch.zeros((ep_size * cap, H), dtype=torch.bfloat16, device=y.device)
    ids_full = torch.zeros((ep_size * cap,), dtype=torch.int32, device=y.device)
    partial_full[:cap] = partial
    ids_full[:cap] = partial_ids
    partial_slot = partial_full[inv_rorder]
    ids_slot = ids_full[inv_rorder]
    rev_bytes = H * 2 + 4
    rev_send = torch.zeros((ep_size * cap, rev_bytes), dtype=torch.uint8, device=y.device)
    rev_recv = torch.zeros((ep_size * cap, rev_bytes), dtype=torch.uint8, device=y.device)
    rev_send[:, : H * 2].view(torch.bfloat16)[:] = partial_slot
    rev_send[:, H * 2: H * 2 + 4].view(torch.int32)[:, 0] = ids_slot
    dist.all_to_all_single(rev_recv, rev_send, group=group)

    # ---- home combine ----
    back_counts = counts_matrix[group.rank()]  # [W]: what I sent to each rank
    valid_back = j < back_counts[seg]
    border = _mega_moe_ordered(valid_back)
    brows = rev_recv[border]
    bpartial = brows[:, : H * 2].view(torch.bfloat16).contiguous()
    bpair = brows[:, H * 2: H * 2 + 4].view(torch.int32).reshape(-1)
    bpair = torch.where(valid_back[border], bpair, torch.zeros_like(bpair))
    btoken = (bpair // top_k).clamp(max=num_tokens - 1)
    bslot = (bpair % top_k).clamp(max=top_k - 1)
    bw = topk_weights[btoken, bslot]
    valid_contrib = valid_back[border]
    bpartial = torch.where(
        valid_contrib.unsqueeze(-1), bpartial,
        torch.zeros_like(bpartial),
    )
    acc = scratch["acc"]
    acc.zero_()
    acc.index_add_(0, btoken, bpartial.float() * bw.unsqueeze(-1))
    y.copy_(acc[:num_tokens].to(y.dtype))
