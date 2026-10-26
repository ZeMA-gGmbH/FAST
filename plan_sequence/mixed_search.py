"""Deterministic breadth/depth scheduling for budgeted label collection."""

import hashlib
import math


def stratified_parts(ranked_parts):
    """Interleave upper, middle, and lower heuristic-rank thirds."""
    ranked_parts = list(ranked_parts)
    width = int(math.ceil(len(ranked_parts) / 3.0))
    groups = [ranked_parts[index : index + width] for index in range(0, len(ranked_parts), width)]
    return [
        group[offset]
        for offset in range(width)
        for group in groups
        if offset < len(group)
    ]


def seeded_tie(seed, state, part):
    payload = "{}:{}:{}".format(seed, ",".join(state), part).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_next(root, discovered, attempted, spine_state, balanced_step, ranker):
    """Return ``(lane, state, part, rank)`` or ``None`` when exhausted."""
    eligible = [
        state
        for state in discovered
        if len(state) > 1 and len(attempted.get(state, ())) < len(state)
    ]
    if not eligible:
        return None

    if spine_state in eligible:
        state = spine_state
        lane = "spine"
    else:
        preferred = "breadth" if balanced_step % 2 == 0 else "depth"
        depth = lambda value: len(root) - len(value)
        if preferred == "breadth":
            state = min(
                eligible,
                key=lambda value: (
                    depth(value),
                    len(attempted.get(value, ())),
                    value,
                ),
            )
        else:
            state = min(
                eligible,
                key=lambda value: (
                    -depth(value),
                    len(attempted.get(value, ())),
                    value,
                ),
            )
        lane = preferred

    ranked = list(ranker(state))
    if sorted(ranked) != sorted(state) or len(set(ranked)) != len(state):
        raise ValueError("heur-out ranker must return every state part exactly once")
    tried = set(attempted.get(state, ()))
    order = ranked if lane in {"spine", "depth"} else stratified_parts(ranked)
    part = next(part for part in order if part not in tried)
    return lane, state, part, ranked.index(part)


def select_next_adaptive(
    root, discovered, attempted, spine_state, balanced_step, ranker, seed
):
    """Use heur-out on spine/depth and seeded-uniform parts on breadth."""
    selected = select_next(
        root, discovered, attempted, spine_state, balanced_step, ranker
    )
    if selected is None:
        return None
    lane, state, part, rank = selected
    if lane != "breadth":
        return lane, state, part, rank, "heur-out", 1.0

    ranked = list(ranker(state))
    untried = [candidate for candidate in state if candidate not in attempted.get(state, ())]
    part = min(untried, key=lambda candidate: seeded_tie(seed, state, candidate))
    return lane, state, part, ranked.index(part), "seeded_uniform", 1.0 / len(untried)
