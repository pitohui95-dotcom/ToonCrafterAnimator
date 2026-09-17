from __future__ import annotations

from tooncrafter_animator.core.pipeline import (
    FRAMES_PER_PASS,
    MAX_INTERMEDIATES,
    evenly_spaced_indices,
    plan_intermediates,
    select_from_clip,
)


def test_even_picks_are_unique_and_ordered() -> None:
    for length in range(1, 17):
        for n in range(0, length + 3):
            idx = evenly_spaced_indices(length, n)
            assert idx == sorted(idx)
            assert len(idx) == len(set(idx))
            assert len(idx) == min(n, length)
            assert all(0 <= i < length for i in idx)
            if n >= 2 and n <= length:
                assert idx[0] == 0
                assert idx[-1] == length - 1


def test_n_le_14_is_one_pass_no_duplicates() -> None:
    for n in range(1, 15):
        plan = plan_intermediates(n)
        assert plan.n_passes == 1
        assert plan.scout is False
        take = plan.passes[0].take_indices
        assert take == sorted(take)
        assert len(take) == len(set(take)) == n
        assert all(1 <= i <= 14 for i in take)
        select_from_clip(FRAMES_PER_PASS, take)


def test_n_gt_14_chains_real_passes() -> None:
    for n in range(15, 61):
        plan = plan_intermediates(n)
        assert plan.scout is True
        assert plan.n_passes == 1 + plan.k_segments
        assert plan.k_segments >= 2
        anchors = plan.anchor_indices
        assert anchors[0] == 0 and anchors[-1] == FRAMES_PER_PASS - 1
        assert len(anchors) == plan.k_segments + 1
        produced = 0
        seen_take = []
        for spec in plan.passes:
            take = spec.take_indices
            assert take == sorted(take)
            assert len(take) == len(set(take))
            assert all(1 <= i <= MAX_INTERMEDIATES for i in take)
            produced += len(take)
            if spec.include_end_anchor:
                produced += 1
            seen_take.extend(take)
        assert produced == n, f"n={n} produced={produced} plan={plan}"


def test_zero() -> None:
    plan = plan_intermediates(0)
    assert plan.n_passes == 0
    assert plan.passes == []
