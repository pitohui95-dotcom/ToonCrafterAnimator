from __future__ import annotations

import math
from dataclasses import replace

from tooncrafter_animator.core.models import FramePlan, PassSpec

FRAMES_PER_PASS = 16
MAX_INTERMEDIATES = 14  # indices 1..14 of a 16-frame clip


def evenly_spaced_indices(length: int, n: int) -> list[int]:
    """n unique indices in ``[0, length)``, ordered, covering the span.

    Used so a user-requested frame count never duplicates a generated frame
    (the regression the planner exists to prevent).
    """
    if n <= 0 or length <= 0:
        return []
    if n >= length:
        return list(range(length))
    if n == 1:
        return [length // 2]
    raw = [int(round(i * (length - 1) / (n - 1))) for i in range(n)]
    # Repair any rounding collision without reordering.
    used: set[int] = set()
    out: list[int] = []
    for value in raw:
        candidate = value
        if candidate in used:
            for delta in range(1, length):
                hi = value + delta
                lo = value - delta
                if 0 <= hi < length and hi not in used:
                    candidate = hi
                    break
                if 0 <= lo < length and lo not in used:
                    candidate = lo
                    break
        used.add(candidate)
        out.append(candidate)
    out.sort()
    # Final uniqueness (should already hold).
    unique: list[int] = []
    for value in out:
        if value not in unique:
            unique.append(value)
    return unique


def plan_intermediates(n: int) -> FramePlan:
    """Plan real ToonCrafter passes for ``n`` intermediate frames.

    One pass always yields 16 frames: start + 14 generated in-betweens + end.
    ``n <= 14`` selects a subset of those 14. ``n > 14`` uses a scout pass to
    pick anchors, then one real pass per segment (no duplication, no blend).
    """
    if n < 0:
        raise ValueError("Intermediate frame count cannot be negative.")
    if n == 0:
        return FramePlan(
            n_intermediates=0,
            n_passes=0,
            scout=False,
            k_segments=0,
            passes=[],
            description="No intermediate frames requested — nothing to generate.",
        )
    if n <= MAX_INTERMEDIATES:
        mid_local = evenly_spaced_indices(MAX_INTERMEDIATES, n)
        take = [i + 1 for i in mid_local]  # 1..14
        return FramePlan(
            n_intermediates=n,
            n_passes=1,
            scout=False,
            k_segments=1,
            passes=[PassSpec(index=0, take_indices=take, include_end_anchor=False, source="user")],
            description=(
                f"One ToonCrafter pass (16 generated frames). "
                f"Exporting {n} of the 14 in-betweens; every exported frame is model output."
            ),
        )

    k = math.ceil((n + 1) / 15)
    k = max(k, 2)
    interiors = k - 1
    remaining = n - interiors
    if remaining < 0:
        k = 1
        interiors = 0
        remaining = n
    # Remaining mids must fit in k * 14.
    while remaining > k * MAX_INTERMEDIATES:
        k += 1
        interiors = k - 1
        remaining = n - interiors
    quot, rem = divmod(remaining, k)
    counts = [quot + (1 if i < rem else 0) for i in range(k)]
    anchor_indices = evenly_spaced_indices(FRAMES_PER_PASS, k + 1)
    passes = [
        PassSpec(
            index=i + 1,  # 0 is the scout
            take_indices=[j + 1 for j in evenly_spaced_indices(MAX_INTERMEDIATES, counts[i])],
            include_end_anchor=(i < k - 1),
            source="scout_anchor",
        )
        for i in range(k)
    ]
    n_passes = 1 + k
    return FramePlan(
        n_intermediates=n,
        n_passes=n_passes,
        scout=True,
        k_segments=k,
        passes=passes,
        description=(
            f"{n_passes} real ToonCrafter passes: 1 scout to pick {k + 1} generated anchors, "
            f"then {k} interpolations between them. No duplicated or blended frames."
        ),
        anchor_indices=anchor_indices,
    )


def select_from_clip(clip_len: int, take_indices: list[int]) -> list[int]:
    """Validate take indices against a 16-frame clip (0 = start, 15 = end)."""
    out = []
    for index in take_indices:
        if index < 0 or index >= clip_len:
            raise IndexError(f"take index {index} outside clip of length {clip_len}")
        out.append(index)
    if len(out) != len(set(out)):
        raise ValueError("Planner produced duplicate frame indices.")
    return out


def with_pass_index(plan: FramePlan, index: int, **kwargs) -> PassSpec:
    spec = plan.passes[index]
    return replace(spec, **kwargs)
