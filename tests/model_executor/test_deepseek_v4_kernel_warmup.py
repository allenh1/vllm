# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.warmup import kernel_warmup


def _mtp_runner(query_len: int = 3):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp"),
        num_spec_tokens=query_len - 1,
        uniform_decode_query_len=query_len,
    )


def test_deepseek_v4_mtp_uniform_decode_warmup_covers_c256():
    requests = kernel_warmup._deepseek_v4_mtp_uniform_decode_warmup_requests(
        _mtp_runner(),
        max_tokens=4096,
        max_reqs=256,
    )

    assert requests == (1, 2, 4, 8, 16, 24, 32, 256)


def test_deepseek_v4_mtp_uniform_decode_warmup_still_respects_limits():
    assert kernel_warmup._deepseek_v4_mtp_uniform_decode_warmup_requests(
        _mtp_runner(),
        max_tokens=4096,
        max_reqs=24,
    ) == (1, 2, 4, 8, 16, 24)
    assert kernel_warmup._deepseek_v4_mtp_uniform_decode_warmup_requests(
        _mtp_runner(),
        max_tokens=96,
        max_reqs=256,
    ) == (1, 2, 4, 8, 16, 24, 32)


def _dspark_runner(num_spec_tokens: int = 5):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="dspark"),
        num_spec_tokens=num_spec_tokens,
        uniform_decode_query_len=1 + num_spec_tokens,
    )


def test_deepseek_v4_dspark_spec_decode_gate():
    assert kernel_warmup._is_deepseek_v4_dspark_spec_decode(_dspark_runner())
    assert not kernel_warmup._is_deepseek_v4_dspark_spec_decode(
        _dspark_runner(num_spec_tokens=0)
    )
    assert not kernel_warmup._is_deepseek_v4_dspark_spec_decode(_mtp_runner())


def test_dspark_spec_decode_warmup_request_classes():
    # The representatives must touch the ==1, plain, and %16 divisibility
    # classes Triton uses to specialize scalar int args.
    assert 1 in kernel_warmup._DSPARK_SPEC_DECODE_WARMUP_REQS
    assert any(n % 16 == 0 for n in kernel_warmup._DSPARK_SPEC_DECODE_WARMUP_REQS)
    # A plain (non-1, non-16-multiple) representative.
    assert any(n != 1 and n % 16 != 0 for n in
               kernel_warmup._DSPARK_SPEC_DECODE_WARMUP_REQS)


def _mhc_ladder_symbols():
    # Import lazily so the pure-ladder unit tests do not require the TileLang
    # package (imported at module import time by deepseek_v4_mhc_warmup).
    from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
        _BIG_PATH_MIN_TOKENS,
        _n_splits_token_ladder,
    )

    return _BIG_PATH_MIN_TOKENS, _n_splits_token_ladder


def test_mhc_n_splits_ladder_covers_every_band():
    big_path_min, ladder_fn = _mhc_ladder_symbols()
    # DSv4-Flash geometry: hc_mult=4, hidden=2048 -> k = 8192 -> split cap 32.
    k = 4 * 2048
    n_sms = 148
    ladder = ladder_fn(4096, k=k, n_sms=n_sms)
    # Every ladder token is on the big-fuse path and within bounds.
    assert ladder == tuple(sorted(set(ladder)))
    assert all(token >= big_path_min for token in ladder)
    assert all(token <= 4096 for token in ladder)
    # Reconstruct every grid Triton can emit for tokens <= 4096 and check each
    # band's n_splits appears in the ladder's coverage.
    covered_splits = {
        max(1, min(n_sms // (token // 64 + (1 if token % 64 else 0)), 32))
        for token in ladder
    }
    for grid in range(1, (4096 + 63) // 64 + 1):
        n_splits = max(1, min(n_sms // grid, 32))
        assert n_splits in covered_splits, f"band {n_splits} (grid {grid}) missed"
    # The smallest ladder token maps to the first big-path band (grid 1).
    assert min(ladder) == big_path_min


@pytest.mark.parametrize("max_tokens", (0, 16))
def test_mhc_n_splits_ladder_below_big_path_min_is_empty(max_tokens):
    _, ladder_fn = _mhc_ladder_symbols()
    assert ladder_fn(max_tokens, k=8192, n_sms=148) == ()
