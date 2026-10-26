"""Deterministic ASAP-style DFS candidate selection."""


def select_next(path, attempted, ranker):
    """Return deepest untried heur-out candidate, backtracking in place."""
    while path:
        state = path[-1]
        if len(state) <= 1:
            path.pop()
            continue
        ranked = list(ranker(state))
        if sorted(ranked) != sorted(state) or len(set(ranked)) != len(state):
            raise ValueError("heur-out ranker must return every state part exactly once")
        tried = set(attempted.get(state, ()))
        for rank, part in enumerate(ranked):
            if part not in tried:
                return "dfs", state, part, rank
        path.pop()
    return None
