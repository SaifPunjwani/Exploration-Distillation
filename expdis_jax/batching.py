"""Bound sequence shapes without discarding active tokens."""


def sequence_bucket(length: int, limit: int) -> int:
    """Power-of-two padding, capped at the configured token limit.

    At most ten shapes cover the paper's context window. Small test budgets
    retain their smaller cap. Attention handles its own TPU block alignment.
    """
    if limit < 1 or length < 0 or length > limit:
        raise ValueError(f"invalid sequence length {length} for limit {limit}")
    return min(limit, max(128, 1 << (max(1, length) - 1).bit_length()))
