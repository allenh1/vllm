# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DeepSeek V4.1's indexer roles and two-level candidate selection.

V4.1's indexer is the V4 one plus two things V4 has no notion of. The first is
that the index keys come from the layer's *compressor* -- `k_norm(wk(latent))`
-- so only a layer that compresses its own KV can produce them, and the index
sources that are not KV sources (24/28/32/36) score against the newest owner's
keys instead. The second is a two-level top-k: layer 20 selects candidate
*blocks* from its own scores, and every later index source masks its own scores
with that selection before taking its top-k.

`select_candidate_blocks` is a pure function on small tensors, so it is tested
here against the released vendor implementation's behaviour, including the two
details that decide correctness: the newest partly-filled block is pinned in,
and blocks that came back -inf are dropped rather than kept.

Around it are the two frames the indexer op calls it in -- a packed prefill
chunk, where a row's blocks start at its own `ks`, and decode, where the logits
past a row's reach are uncleaned -- and the handover that carries the source's
answer to the layers that mask their scores with it. Both are pure tensor and
bookkeeping logic, so both are tested here too; what is not tested is any of it
on a GPU, which is the whole of the rest of the port's open risk.
"""

import types

import pytest
import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import (
    V41CandidateBlocks,
    _v41_candidate_holder,
    _v41_decode_candidate_mask,
    _v41_prefill_candidate_mask,
)
from vllm.models.deepseek_v4.attention import (
    select_candidate_blocks,
    v41_layer_roles,
)

#: The real DeepSeek-V4.1-Flash text config, minus the fields nothing here reads.
V41_COMPRESS_RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]

NEG_INF = float("-inf")


def _v41_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=list(V41_COMPRESS_RATIOS),
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
    )


def _v4_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(num_hidden_layers=4, compress_ratios=[1, 4, 4, 1])


def _masked(scores, compress_lens):
    """One row of index scores with everything past `compress_lens` at -inf.

    That clamp is not part of `select_candidate_blocks` -- it is what the caller
    owes it, and what makes "this block scores -inf" mean "unreachable" rather
    than "scored badly".
    """
    scores = torch.tensor([float(s) for s in scores], dtype=torch.float32)
    scores[torch.arange(scores.numel()) >= compress_lens] = NEG_INF
    return scores


def _selected_positions(mask) -> set[int]:
    return set(torch.nonzero(mask, as_tuple=True)[0].tolist())


# ---------------------------------------------------------------------------
# select_candidate_blocks
# ---------------------------------------------------------------------------


def test_the_newest_partly_filled_block_is_pinned_in():
    """A partly-filled newest block beats a full older one on merit alone.

    Positions 0..3 hold the whole compressed prefix and positions 4..6 are the
    newest, partly-filled block. Every one of 4..6 scores *below* the whole of
    block 0, so an unpinned block top-k would drop them -- and they are the most
    recent tokens the sliding window does not cover, which is the reason the pin
    exists.
    """
    logits = _masked([9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0], compress_lens=7)
    mask = select_candidate_blocks(logits, 7, topk_blocks=1, block_size=4)
    # block 0 (positions 0..3) is pinned nowhere, but it is the only block the
    # top-k would have picked; block 1 must have been forced in instead.
    assert _selected_positions(mask) == {4, 5, 6}

    # And with room for both, the pinned one is still there.
    mask = select_candidate_blocks(logits, 7, topk_blocks=2, block_size=4)
    assert _selected_positions(mask) == {0, 1, 2, 3, 4, 5, 6}


def test_the_pin_does_not_apply_when_the_newest_block_is_full():
    """A full newest block is pinned too, but the pin changes nothing.

    `last = (compress_lens - 1) // block_size` names the block holding the
    newest position whether or not it is full, so this is the case that proves
    the pin is about the newest position and not about partialness.
    """
    logits = _masked([1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0], compress_lens=8)
    mask = select_candidate_blocks(logits, 8, topk_blocks=1, block_size=4)
    assert _selected_positions(mask) == {4, 5, 6, 7}


def test_unreachable_blocks_are_dropped_not_kept():
    """Fewer reachable blocks than `topk_blocks`: the leftovers are -inf.

    Two blocks are reachable out of five and `topk_blocks` is 5, so three of the
    five top-k picks are the padded/unreachable blocks -- which score -inf and
    must not come back as a mask of "everything is a candidate". If they did,
    the two-level top-k would silently turn into a one-level one.
    """
    width = 40  # 8 blocks of 5
    logits = _masked([5.0] * width, compress_lens=10)
    mask = select_candidate_blocks(logits, 10, topk_blocks=8, block_size=5)
    assert _selected_positions(mask) == set(range(10))


def test_the_mask_is_per_block_and_not_clamped_to_the_reach():
    """A selected block comes back whole, reachable or not.

    Reach 6 of 12 with 4-wide blocks: block 1 holds the newest reachable
    position and is pinned, and positions 6 and 7 come back set with it even
    though the query cannot reach them. That is deliberate -- it is what
    `repeat_interleave` gives, and it is harmless because those positions are
    -inf in the logits (the caller's clamp) and so cannot win the top-k that
    follows. Clamping the mask here instead would be a second place that has to
    know the reach, and a second place to get it wrong.

    The same mask shows the other half of the rule: block 2 is entirely past
    the reach, scores -inf, and is not kept even though `topk_blocks` has room
    for it. The mask is built from the top-k's values, not its indices.
    """
    logits = _masked([1.0] * 12, compress_lens=6)
    mask = select_candidate_blocks(logits, 6, topk_blocks=4, block_size=4)
    assert _selected_positions(mask) == set(range(8))
    assert mask.shape == logits.shape

    # And the pin is not a tie-break: with one slot it takes it, and block 0 --
    # which scores the same -- loses.
    mask = select_candidate_blocks(logits, 6, topk_blocks=1, block_size=4)
    assert _selected_positions(mask) == {4, 5, 6, 7}


def test_a_query_that_reaches_nothing_selects_nothing():
    """`compress_lens == 0` pins nothing: `last` is -1 and matches no block.

    This is the first decode step of a ratio-8 layer, and the vendor's own
    arithmetic answers it -- no block of it is "newest" because no position of
    it is reachable.
    """
    logits = _masked([1.0] * 12, compress_lens=0)
    mask = select_candidate_blocks(logits, 0, topk_blocks=4, block_size=4)
    assert _selected_positions(mask) == set()


def test_the_width_pad_is_neg_inf_and_not_zero():
    """A trailing partial block scores as its real entries alone.

    9 columns with 4-wide blocks is 3 blocks, the last holding one real column
    and three padded ones -- and that one real column is past the reach, so
    block 2's true score is -inf. With a zero pad it would score 0.0, beat
    block 0's -1.0, and take the second slot from a block that really is
    reachable.
    """
    logits = _masked([-1.0] * 4 + [-2.0] * 4 + [99.0], compress_lens=8)
    mask = select_candidate_blocks(logits, 8, topk_blocks=2, block_size=4)
    # Block 1 is pinned (it holds the newest position) and block 0 is the best
    # of the rest; the 99.0 at position 8 is unreachable and must not count.
    assert _selected_positions(mask) == set(range(8))


def test_a_width_that_is_not_a_multiple_of_the_block_size():
    """The mask comes back truncated to the width it was given."""
    logits = _masked([1.0] * 7, compress_lens=7)
    mask = select_candidate_blocks(logits, 7, topk_blocks=4, block_size=4)
    assert mask.shape == (7,)
    assert _selected_positions(mask) == {0, 1, 2, 3, 4, 5, 6}


def test_a_high_scoring_position_that_is_unreachable_does_not_count():
    """The block scores come from what the query can reach, not from behind it."""
    # Positions 0..3 score high but are unreachable; positions 4..5 are all the
    # query can reach, and they are its newest block.
    logits = torch.tensor(
        [9.0, 9.0, 9.0, 9.0, 1.0, 1.0, NEG_INF, NEG_INF], dtype=torch.float32
    )
    mask = select_candidate_blocks(logits, 6, topk_blocks=1, block_size=4)
    assert _selected_positions(mask) == {4, 5, 6, 7}


def test_a_pinned_newest_block_displaces_the_best_older_one():
    """What the pin costs, stated: when only one slot is left it takes it.

    This is the mechanism doing its job -- the newest block is the one the
    sliding window has not covered yet -- and it is worth pinning down because
    it means the candidate set is not simply "the top blocks by score".
    """
    # Increasing block scores, so the pin is the only reason block 2 is kept.
    logits = _masked([1.0] * 4 + [2.0] * 4 + [3.0] * 4, compress_lens=12)
    unpinned_by_score = {0, 1, 2, 3, 4, 5, 6, 7}  # blocks 3 and 2, say
    mask = select_candidate_blocks(logits, 12, topk_blocks=2, block_size=4)
    # block 2 is the newest and is pinned; block 1 has the best score left.
    assert _selected_positions(mask) == {4, 5, 6, 7, 8, 9, 10, 11}
    assert _selected_positions(mask) != unpinned_by_score


def test_decode_takes_an_int_and_prefill_a_tensor():
    """The two call shapes the vendor distinguishes, on the same numbers.

    Decode has one query per row and a single reach for the whole batch, so
    `compress_lens` is an int that broadcasts; prefill has a different reach per
    row, so it is a tensor. Both must produce the same mask when the numbers
    agree.
    """
    scores = [9.0] * 4 + [5.0] * 4 + [1.0] * 4

    def clamped(reach):
        # Per-row clamping, which is what the contract asks of the caller: the
        # positions past *this row's* reach have to be -inf for the block scores
        # to mean "reachable" rather than "scored badly".
        return _masked(scores, compress_lens=reach)

    row = clamped(8)
    from_int = select_candidate_blocks(row, 8, topk_blocks=2, block_size=4)
    # reach 8 -> block 1 holds the newest position and is pinned; block 0 is
    # the best of what is left.
    assert _selected_positions(from_int) == set(range(8))

    from_scalar = select_candidate_blocks(row, torch.tensor(8), topk_blocks=2, block_size=4)
    assert torch.equal(from_int, from_scalar)

    logits = torch.stack([clamped(8)] * 3)
    # A tensor reach with one entry per row, which is what prefill passes.
    batched = select_candidate_blocks(
        logits, torch.tensor([8, 8, 8]), topk_blocks=2, block_size=4
    )
    assert batched.shape == logits.shape
    assert torch.equal(batched, from_int.expand(3, -1))

    # ...and a per-row reach, which is where an int could not express the
    # answer: row 0 reaches block 0 alone, and row 2 reaches all three, where
    # the pin costs it block 1.
    per_row = select_candidate_blocks(
        torch.stack([clamped(4), clamped(8), clamped(12)]),
        torch.tensor([4, 8, 12]),
        topk_blocks=2,
        block_size=4,
    )
    assert _selected_positions(per_row[0]) == {0, 1, 2, 3}
    assert torch.equal(per_row[1], from_int)
    assert _selected_positions(per_row[2]) == {0, 1, 2, 3, 8, 9, 10, 11}


def _reference_select(logits, compress_lens, topk_blocks, block_size):
    """`select_candidate_blocks` written the long way, per row, with no tensors.

    Independent of the ported implementation on purpose: it walks positions,
    groups them into blocks and ranks the blocks by hand. It is only meaningful
    when the input's reachable scores are distinct, so that no tie-break can
    make the two disagree for a reason that is not a bug.
    """
    out = []
    for row, reach in zip(logits, compress_lens, strict=True):
        width = len(row)
        reach = int(reach)
        num_blocks = -(-width // block_size)  # the pad rounds up, not down
        block_scores = {}
        for pos in range(min(reach, width)):
            block = pos // block_size
            block_scores[block] = max(block_scores.get(block, NEG_INF), row[pos])
        # The pin names the block holding the newest reachable position, and it
        # has to name a block the width actually has: a reach past the width
        # (the vendor's `unsqueeze(-1)` at the real geometry never does this)
        # pins nothing rather than conjuring a block out of the pad.
        newest = (reach - 1) // block_size if reach > 0 else -1
        if 0 <= newest < num_blocks:
            block_scores[newest] = float("inf")
        ranked = sorted(block_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        keep = {
            block
            for block, score in ranked[: min(topk_blocks, len(ranked))]
            if score > NEG_INF
        }
        out.append([pos // block_size in keep for pos in range(width)])
    return torch.tensor(out, dtype=torch.bool)


@pytest.mark.parametrize(
    "width,block_size,reach,topk_blocks",
    [
        (8, 4, 8, 2),
        (8, 4, 7, 2),
        (8, 4, 5, 4),
        (7, 4, 7, 8),
        (12, 5, 9, 3),
        (12, 5, 3, 1),
        (16, 8, 16, 2),
        (16, 8, 15, 1),
        (16, 8, 0, 2),
        (3, 8, 3, 4),
    ],
)
def test_against_a_position_at_a_time_reference(width, block_size, reach, topk_blocks):
    """The vectorised port against an independent per-position formulation."""
    generator = torch.Generator().manual_seed(width * 1000 + block_size * 10 + reach)
    rows = []
    lens = []
    for row_reach in range(1, reach + 2):
        scores = torch.rand(width, generator=generator) * 100.0
        rows.append(_masked(scores.tolist(), row_reach))
        lens.append(row_reach)
    logits = torch.stack(rows)
    lens_tensor = torch.tensor(lens)

    got = select_candidate_blocks(logits, lens_tensor, topk_blocks, block_size)
    want = _reference_select(
        logits.tolist(), lens_tensor.tolist(), topk_blocks, block_size
    )
    assert torch.equal(got, want)


def test_v4_1s_own_numbers():
    """The released config's geometry, at a width where its own limit bites.

    2048 blocks of 8 is 16384 positions, so the 2100-block width here is more
    blocks than `candidate_topk_blocks` -- the `min(topk_blocks, num_blocks)`
    clamp really does drop 52 of them, and the scores are laid out so it is
    unambiguous which 52: the lowest-scoring ones. At any shorter context every
    block fits and the mask is the whole reach, which is bit-for-bit the plain
    (uncapped) top-k the vendor would have taken.
    """
    block_size = 8
    topk_blocks = 2048
    width = 2100 * block_size  # 16800 positions, 2100 blocks
    logits = torch.arange(width, dtype=torch.float32) // block_size
    mask = select_candidate_blocks(
        logits, width, topk_blocks=topk_blocks, block_size=block_size
    )
    assert mask.shape == (width,)
    first_kept = (2100 - topk_blocks) * block_size
    assert _selected_positions(mask) == set(range(first_kept, width))


# ---------------------------------------------------------------------------
# Level one in the op's two frames
# ---------------------------------------------------------------------------
#
# `select_candidate_blocks` scores one request per tensor with the blocks
# anchored at column 0, which is neither of the shapes the indexer op has. The
# decode logits are `[rows, width]` with columns past each row's context length
# left uncleaned by the paged-logits kernel, and a prefill chunk packs several
# requests into one tensor, where a row owns the columns `[ks, ke)` and its
# blocks start at its own `ks`. Both wrappers are pure tensor arithmetic, so
# both are tested here.


def test_prefill_rows_are_anchored_at_their_own_ks():
    """Two requests in one chunk, each with its own block grid.

    `cu_seqlen_ks` is where a row's request starts in the gathered key buffer
    and `cu_seqlen_ke` where it ends, so row 0 owns columns 0..5 and row 1 owns
    6..10. The +100s are what the logits kernel leaves outside a row's frame
    (`clean_logits=False`), and they must not reach the block scores: the row's
    scores are gathered from its own frame and everything past its reach is
    -inf before the blocks are ranked.
    """
    logits = torch.tensor(
        [
            [-5.0, -5.0, -5.0, -5.0, -10.0, -10.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, -5.0, -5.0, -5.0, -5.0, -10.0],
        ],
        dtype=torch.float32,
    )
    ks = torch.tensor([0, 6])
    ke = torch.tensor([6, 11])
    mask = _v41_prefill_candidate_mask(
        logits, ks, ke, topk_blocks=1, block_size=4
    )

    # One grid for the whole chunk: the widest reach rounded up to a block, on
    # top of the last row's base. It is wider than the logits, which is what
    # the consumer trims back (`take_prefill`).
    assert mask.shape == (2, 6 + 8)
    assert mask.shape[-1] > logits.shape[-1]
    # Each row's newest block and nothing else: row 0's block 1 covers 4..7,
    # row 1's block 1 covers 10..13 (its grid starts at 6, not at 0), and both
    # beat the older block that scores better.
    assert _selected_positions(mask[0]) == {4, 5, 6, 7}
    assert _selected_positions(mask[1]) == {10, 11, 12, 13}
    # A selected block comes back whole, reachable or not: row 0 can only reach
    # columns 0..5 and columns 6..7 are set anyway, exactly as the vendor's
    # per-block mask does it -- the top-k that follows is bounded by `ks`/`ke`
    # and cannot pick them.
    assert mask[0, 6:8].all()
    # Another request's columns are never selectable, however they score.
    assert not mask[0, 8:].any()
    assert not mask[1, :6].any()


def test_prefill_does_not_anchor_a_rows_grid_at_column_zero():
    """The one place the frame differs from the vendor's, on its own.

    The vendor scores a request whose tensor starts at its first compressed
    position, so its blocks are groups of `block_size` from column 0. Here a
    row's grid starts at its own `ks`, and this is the smallest case where the
    two disagree: the reach is 9 of a 12-wide frame with 8-wide blocks, so the
    pin lands on columns 11..18 as written and would land on 8..15 anchored at
    0 -- which is why the answer below is not the one borrowing the vendor's
    frame would have given.
    """
    logits = torch.full((1, 12), -1.0)
    mask = _v41_prefill_candidate_mask(
        logits, torch.tensor([3]), torch.tensor([12]), topk_blocks=1, block_size=8
    )
    assert mask.shape[-1] == 3 + 16
    assert _selected_positions(mask[0]) == set(range(11, 19))
    assert not mask[0, 8:11].any()


def test_prefill_accepts_the_metadata_int32_bounds():
    """`cu_seqlen_ks/ke` are int32 on the device, and they index a tensor.

    A signed 32-bit index is not a torch index -- `Tensor.gather` takes int64
    and nothing else -- so the cast to the index dtype happens before the
    gather. Without it this raises "Expected dtype int64 for index" the first
    time a prefill runs against the real metadata, which is a shape of bug no
    amount of small-int testing finds.
    """
    logits = torch.full((1, 9), -1.0)
    from_int32 = _v41_prefill_candidate_mask(
        logits,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([9], dtype=torch.int32),
        topk_blocks=2,
        block_size=4,
    )
    from_int64 = _v41_prefill_candidate_mask(
        logits, torch.tensor([0]), torch.tensor([9]), topk_blocks=2, block_size=4
    )
    assert torch.equal(from_int32, from_int64)


def test_prefill_drops_the_unreachable_leftovers_of_the_block_topk():
    """Blocks that own no reachable column are not kept, however many slots.

    The grid is one width for the whole chunk, so a short-reach row has blocks
    in it that score -inf; `topk_blocks` has room for them and they must still
    come back empty. Otherwise level two is handed "everything in this row's
    grid" rather than the blocks level one chose -- a silent widening, and one
    that only shows up in the layers after the source.
    """
    # Row 0 reaches the whole frame (4 blocks); row 1 reaches 5 columns (2
    # blocks) but its grid -- and so the mask -- runs four blocks wide.
    logits = torch.full((2, 16), -1.0)
    mask = _v41_prefill_candidate_mask(
        logits, torch.tensor([0, 0]), torch.tensor([16, 5]), topk_blocks=8, block_size=4
    )
    assert _selected_positions(mask[0]) == set(range(16))
    assert _selected_positions(mask[1]) == set(range(8))


def test_a_prefill_row_that_reaches_nothing_selects_nothing():
    """`ke == ks`: no compressed position yet, so no block is "newest".

    The pin names block `(0 - 1) // block_size`, which matches no block, and
    the row comes back empty -- the same arithmetic the vendor relies on.
    """
    logits = torch.full((1, 10), 5.0)
    mask = _v41_prefill_candidate_mask(
        logits, torch.tensor([4]), torch.tensor([4]), topk_blocks=2, block_size=4
    )
    assert not mask.any()


def test_prefill_rows_of_one_chunk_have_their_own_reaches():
    """Three rows of one chunk: ten blocks, two, and none.

    A single int cannot say this, which is why the prefill call passes a
    per-row reach -- and the +1000s past each row's reach are the uncleaned
    logits the block scores must not see.
    """
    scores = torch.arange(40, dtype=torch.float32) * 0.1
    logits = scores.expand(3, 40).clone()
    logits[1, 6:] = 1000.0
    logits[2, :] = 1000.0
    mask = _v41_prefill_candidate_mask(
        logits,
        torch.zeros(3, dtype=torch.int64),
        torch.tensor([40, 6, 0]),
        topk_blocks=2,
        block_size=4,
    )
    assert mask.shape == logits.shape
    # Row 0: ten blocks, the two best of which are the last two.
    assert _selected_positions(mask[0]) == set(range(32, 40))
    # Row 1: two blocks, block 1 pinned over block 0 -- and the 1000s are past
    # its reach, so block 1's real score is what it is.
    assert _selected_positions(mask[1]) == set(range(8))
    # Row 2: nothing reachable, nothing selected, no matter how it scores.
    assert not mask[2].any()


@pytest.mark.parametrize(
    "ks,reach,block_size",
    [(0, 5, 4), (3, 9, 8), (7, 1, 8), (6, 5, 4), (3, 0, 4), (5, 3, 3), (11, 2, 8)],
)
def test_the_prefill_mask_always_covers_the_chunk_it_came_from(ks, reach, block_size):
    """The reader's contract, over the geometries that produce a tail.

    A consumer reads the mask as `mask[:, :logits.shape[-1]]` and refuses it
    otherwise, so the mask has to be at least as wide as the chunk's logits,
    and its grid has to be a whole number of blocks -- the tail past the frame
    is the block grid's pad, and nothing else.
    """
    width = ks + reach  # the chunk's logits are exactly this row's frame
    logits = torch.arange(width, dtype=torch.float32).unsqueeze(0)
    mask = _v41_prefill_candidate_mask(
        logits, torch.tensor([ks]), torch.tensor([ks + reach]), 4, block_size
    )
    assert mask.shape[0] == 1
    assert mask.shape[-1] >= width
    assert (mask.shape[-1] - ks) % block_size == 0
    if reach:
        assert mask.any()
    else:
        assert not mask.any()


def test_decode_ignores_what_a_row_cannot_reach():
    """The paged logits kernel is asked not to clean the columns out of reach.

    They hold whatever was in the output buffer -- +100 below -- and they have
    to leave the block scores before the max: a block scored by garbage is a
    candidate set that has nothing to do with the model's scores. The reach is
    the metadata's *compressed* context length, one per row.
    """
    clean = torch.tensor(
        [[-5.0, -5.0, -5.0, -5.0, -10.0, -10.0, -10.0, -10.0]], dtype=torch.float32
    )
    dirty = clean.clone()
    dirty[0, 6:] = 100.0
    seq_lens = torch.tensor([6])
    got = _v41_decode_candidate_mask(dirty, seq_lens, topk_blocks=1, block_size=4)
    want = _v41_decode_candidate_mask(clean, seq_lens, topk_blocks=1, block_size=4)
    assert torch.equal(got, want)
    # Block 1 is the newest (columns 4..7) and the pin takes it over block 0,
    # which scores better.
    assert _selected_positions(got[0]) == {4, 5, 6, 7}


def test_decode_reads_one_context_length_per_row():
    """A long row, a short one, and a padded token in the same batch.

    A padded token has `seq_lens == 0`, which pins nothing and selects nothing;
    its top-k is the caller's `-1` fill either way.
    """
    logits = torch.tensor(
        [[-1.0] * 8, [-1.0] * 8, [100.0] * 8], dtype=torch.float32
    )
    mask = _v41_decode_candidate_mask(
        logits, torch.tensor([8, 3, 0]), topk_blocks=2, block_size=4
    )
    assert mask.shape == logits.shape
    assert _selected_positions(mask[0]) == set(range(8))
    # Reach 3 is one block, and the pin keeps it whole -- column 3 is set with
    # it though the row cannot reach it.
    assert _selected_positions(mask[1]) == set(range(4))
    assert not mask[2].any()


def test_decode_reads_the_two_dimensional_context_lengths_of_spec_decode():
    """`seq_lens` is `(B, next_n)` whenever the batch carries draft positions.

    The logits are `B * next_n` rows -- one query position each -- and the
    flattened context lengths line up with them in that order.
    """
    logits = torch.full((4, 8), -1.0)
    seq_lens = torch.tensor([[8, 8], [5, 3]])
    mask = _v41_decode_candidate_mask(logits, seq_lens, topk_blocks=1, block_size=4)
    assert mask.shape == logits.shape
    assert _selected_positions(mask[0]) == {4, 5, 6, 7}  # reach 8: block 1
    assert _selected_positions(mask[1]) == {4, 5, 6, 7}
    assert _selected_positions(mask[2]) == {4, 5, 6, 7}  # reach 5: block 1
    assert _selected_positions(mask[3]) == {0, 1, 2, 3}  # reach 3: block 0


def test_decode_requires_one_context_length_per_row():
    """A wrong-length `seq_lens` is a wrong mask, so it raises instead."""
    logits = torch.full((2, 8), -1.0)
    with pytest.raises(AssertionError, match="one context length per logits row"):
        _v41_decode_candidate_mask(logits, torch.tensor([8]), topk_blocks=1, block_size=4)


# ---------------------------------------------------------------------------
# The handover from the source layer to the layers after it
# ---------------------------------------------------------------------------


def _forward_context(layers=None) -> ForwardContext:
    """A `ForwardContext` with just what the candidate handover touches."""
    return ForwardContext(
        no_compile_layers=dict(layers or {}), attn_metadata=None, slot_mapping={}
    )


def test_a_published_chunk_comes_back_trimmed_to_the_reader():
    blocks = V41CandidateBlocks()
    with override_forward_context(_forward_context()):
        blocks.reset()
        mask = torch.tensor([[True, False], [True, True]])
        blocks.publish_prefill(0, 4, mask)
        taken = blocks.take_prefill(0, 0, 4, torch.Size((2, 1)))
        assert torch.equal(taken, mask[:, :1])
        # A chunk the source has not published is an error, not a mask.
        with pytest.raises(RuntimeError, match="no mask for chunk 1"):
            blocks.take_prefill(1, 4, 8, torch.Size((2, 2)))


def test_a_chunk_the_source_split_differently_is_not_this_layers_to_use():
    """Two layers only share a chunk's blocks if they chunked the batch alike.

    The token bounds are how that is checked without comparing masks: a layer
    whose chunk 0 is tokens 0:8 while the source's is 4:8 has a different
    (ks, ke) per row, and the source's blocks are in another frame.
    """
    blocks = V41CandidateBlocks()
    with override_forward_context(_forward_context()):
        blocks.reset()
        blocks.publish_prefill(4, 8, torch.zeros((2, 2), dtype=torch.bool))
        with pytest.raises(RuntimeError, match="chunked this batch"):
            blocks.take_prefill(0, 0, 8, torch.Size((2, 2)))


def test_blocks_that_do_not_cover_the_logits_are_refused():
    blocks = V41CandidateBlocks()
    with override_forward_context(_forward_context()):
        blocks.reset()
        blocks.publish_decode(torch.ones((2, 3), dtype=torch.bool))
        with pytest.raises(RuntimeError, match="does not cover logits"):
            blocks.take_decode(torch.Size((3, 3)))
        with pytest.raises(RuntimeError, match="does not cover logits"):
            blocks.take_decode(torch.Size((2, 4)))
        assert blocks.take_decode(torch.Size((2, 2))).shape == (2, 2)
        # A pass that published no decode at all is its own error, with its own
        # message -- this is the source running a prefill-only step.
        fresh = V41CandidateBlocks()
        fresh.reset()
        with pytest.raises(RuntimeError, match="ran no decode step"):
            fresh.take_decode(torch.Size((2, 3)))


def test_a_new_pass_starts_empty():
    blocks = V41CandidateBlocks()
    with override_forward_context(_forward_context()):
        blocks.reset()
        blocks.publish_prefill(0, 2, torch.ones((2, 3), dtype=torch.bool))
        blocks.publish_decode(torch.ones((2, 3), dtype=torch.bool))
        blocks.reset()
        with pytest.raises(RuntimeError, match="no mask for chunk 0"):
            blocks.take_prefill(0, 0, 2, torch.Size((2, 3)))
        with pytest.raises(RuntimeError, match="ran no decode step"):
            blocks.take_decode(torch.Size((2, 3)))


def test_blocks_from_another_forward_pass_are_refused():
    """A source that does not run its indexer leaves the previous pass's blocks.

    The dense-MHA short-extend case deliberately skips the indexer op, so the
    blocks stay where the last pass put them; a consumer that then asks for
    them has to fail loudly rather than mask its scores with another batch's
    candidate sets -- a wrong answer no one would see in the output.
    """
    blocks = V41CandidateBlocks()
    with override_forward_context(_forward_context()):
        blocks.reset()
        blocks.publish_decode(torch.ones((1, 4), dtype=torch.bool))
    with override_forward_context(_forward_context()):
        with pytest.raises(RuntimeError, match="stale"):
            blocks.take_decode(torch.Size((1, 4)))


def test_the_pass_is_held_and_not_remembered_by_an_id():
    """Why the check above is reliable: the context object is held, not its id.

    An id is an address, and the previous pass's `ForwardContext` is normally
    dead by the time a consumer asks -- so the new one can be handed the same
    address and compare equal, and the stale blocks would be used silently. A
    held reference cannot be recycled, so identity means what it says.
    """
    blocks = V41CandidateBlocks()
    first = _forward_context()
    with override_forward_context(first):
        blocks.reset()
    assert blocks.forward_context is first
    second = _forward_context()
    with override_forward_context(second):
        blocks.reset()
        assert blocks.forward_context is second


def test_the_source_module_is_found_by_its_layer_prefix():
    """The op is handed a layer *name*; the blocks live on that layer's module.

    `ForwardContext.no_compile_layers` is the model's `static_forward_context`,
    the same mapping the compressed caches are shared through, so the op finds
    the source without knowing the module tree.
    """
    blocks = V41CandidateBlocks()
    module = types.SimpleNamespace(v41_candidate_blocks=blocks)
    with override_forward_context(
        _forward_context({"model.layers.20.attn": module})
    ):
        assert _v41_candidate_holder("model.layers.20.attn") is blocks
        # A prefix that is not registered at all ...
        with pytest.raises(ValueError, match="not a candidate source"):
            _v41_candidate_holder("model.layers.24.attn")
        # ... and a module that is not a V4.1 attention layer, which has no
        # blocks of its own to publish.
        with override_forward_context(
            _forward_context({"model.layers.2.attn": object()})
        ):
            with pytest.raises(ValueError, match="not a candidate source"):
                _v41_candidate_holder("model.layers.2.attn")


# ---------------------------------------------------------------------------
# Layer roles: which layers run an indexer, own its keys, and select candidates
# ---------------------------------------------------------------------------


def _roles(config, layer_id):
    return v41_layer_roles(config, layer_id)


def test_only_layer_20_is_the_candidate_source():
    config = _v41_config()
    sources = [i for i in range(43) if _roles(config, i).is_candidate_source]
    assert sources == [20]


def test_the_layers_after_the_source_use_its_candidates():
    config = _v41_config()
    # The four index sources past layer 20, and nothing else. Layers 21..23 and
    # 25..27 run no indexer at all, and the draft layers 40..42 run none either
    # -- they only *look* like consumers because their raw ratio is 0 while
    # their layer id is past the source.
    assert [i for i in range(43) if _roles(config, i).uses_candidate_blocks] == [
        24, 28, 32, 36
    ]
    # The source itself is not a consumer: it scores against every reachable
    # position, which is what makes its own top-k the one it would have taken
    # anyway.
    assert not _roles(config, 20).uses_candidate_blocks
    assert not _roles(config, 19).uses_candidate_blocks
    # An index source *before* the source has nothing to consume yet.
    assert not _roles(config, 2).uses_candidate_blocks
    assert not _roles(config, 14).uses_candidate_blocks


def test_candidates_are_off_without_a_source_layer():
    """A V4.1 config may name no source at all; nothing then selects blocks."""
    config = _v41_config()
    del config.candidate_source_layer_id
    assert not any(_roles(config, i).uses_candidate_blocks for i in range(43))
    assert not any(_roles(config, i).is_candidate_source for i in range(43))

    config = _v41_config()
    config.candidate_source_layer_id = -1  # the vendor's "turned off" spelling
    assert not any(_roles(config, i).uses_candidate_blocks for i in range(43))
    assert not any(_roles(config, i).is_candidate_source for i in range(43))


def test_only_the_kv_sources_own_index_keys():
    """The four that carry `wk`/`k_norm` in the released checkpoint."""
    config = _v41_config()
    owners = [i for i in range(43) if _roles(config, i).owns_index_keys]
    assert owners == [2, 8, 14, 20]


def test_the_index_sources_that_do_not_own_keys_read_the_newest_owner():
    config = _v41_config()
    for layer_id, owner in ((24, 20), (28, 20), (32, 20), (36, 20)):
        roles = _roles(config, layer_id)
        assert roles.runs_indexer and not roles.owns_index_keys
        assert roles.index_owner_layer == owner
    # An owner reads nothing: it *is* the source.
    for layer_id in (2, 8, 14, 20):
        assert _roles(config, layer_id).index_owner_layer is None


def test_a_layer_that_runs_no_indexer_reads_no_index_keys():
    """Non-index layers reuse the published top-k and never touch the keys."""
    config = _v41_config()
    for layer_id in (0, 1, 3, 5, 21, 23, 39, 40, 42):
        roles = _roles(config, layer_id)
        assert not roles.runs_indexer
        assert roles.index_owner_layer is None
        assert not roles.owns_index_keys


def test_v4_has_no_v41_roles_at_all():
    """The gate the whole port hangs on: a V4 config resolves to None."""
    config = _v4_config()
    assert all(_roles(config, i) is None for i in range(4))
