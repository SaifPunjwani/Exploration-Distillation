"""Pure budget and lineage rules for fixed-total Exploration-Distillation campaigns.

This module deliberately has no JAX dependency.  The paper's compute table,
the launchers, and the training pipeline all need to agree on the same integer
partition, including remainder handling when a budget is not divisible by the
number of rounds or Scouts.

The fixed-total RL contract is:

* 200 Scout RL updates across the whole campaign;
* 100 Central-model RL updates across the whole campaign;
* the Scout share is split across rounds, then across the K Scouts in a round;
* the Central share is split across rounds;
* the SFT cap is per round and is not part of these RL totals.
"""

from __future__ import annotations

from dataclasses import dataclass


def split_budget(total: int, parts: int) -> list[int]:
    """Split an integer budget exactly, front-loading at most one remainder.

    ``sum(split_budget(total, parts)) == total`` always holds.  Invalid or
    zero-sized partitions fail rather than silently dropping an update.
    """

    total = int(total)
    parts = int(parts)
    if total < 0:
        raise ValueError(f"total={total!r} must be >= 0")
    if parts < 1:
        raise ValueError(f"parts={parts!r} must be >= 1")
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def round_scout_budgets(
    total_scout_updates: int,
    rounds: int,
    scouts_per_round: int,
    round_index: int,
) -> list[int]:
    """Return every Scout's RL-update allocation in one 1-based round."""

    round_allocations = split_budget(total_scout_updates, rounds)
    round_index = int(round_index)
    if not 1 <= round_index <= len(round_allocations):
        raise IndexError(
            f"round_index={round_index!r} outside 1..{len(round_allocations)}"
        )
    return split_budget(
        round_allocations[round_index - 1],
        int(scouts_per_round),
    )


def round_central_budget(
    total_central_updates: int,
    rounds: int,
    round_index: int,
) -> int:
    """Return the Central model's RL-update allocation in one 1-based round."""

    allocations = split_budget(total_central_updates, rounds)
    round_index = int(round_index)
    if not 1 <= round_index <= len(allocations):
        raise IndexError(f"round_index={round_index!r} outside 1..{len(allocations)}")
    return allocations[round_index - 1]


@dataclass(frozen=True)
class CampaignGeometry:
    """Auditable fixed-total RL geometry for one K-by-R configuration."""

    scouts_per_round: int
    rounds: int
    scout_updates_by_round: tuple[tuple[int, ...], ...]
    central_updates_by_round: tuple[int, ...]
    rows_per_rl_update: int

    @property
    def total_scout_updates(self) -> int:
        return sum(sum(per_scout) for per_scout in self.scout_updates_by_round)

    @property
    def total_central_updates(self) -> int:
        return sum(self.central_updates_by_round)

    @property
    def total_rl_updates(self) -> int:
        return self.total_scout_updates + self.total_central_updates

    @property
    def update_consumed_rollout_rows(self) -> int:
        return self.total_rl_updates * int(self.rows_per_rl_update)


def fixed_total_geometry(
    *,
    scouts_per_round: int,
    rounds: int,
    total_scout_updates: int = 200,
    total_central_updates: int = 100,
    prompts_per_update: int = 4,
    generations_per_prompt: int = 16,
) -> CampaignGeometry:
    """Build and validate the fixed-total geometry used by the paper table."""

    scouts_per_round = int(scouts_per_round)
    rounds = int(rounds)
    prompts_per_update = int(prompts_per_update)
    generations_per_prompt = int(generations_per_prompt)
    if scouts_per_round < 1:
        raise ValueError("scouts_per_round must be >= 1")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if prompts_per_update < 1 or generations_per_prompt < 1:
        raise ValueError("prompt and generation counts must both be >= 1")

    scout_updates = tuple(
        tuple(
            round_scout_budgets(
                total_scout_updates,
                rounds,
                scouts_per_round,
                round_index,
            )
        )
        for round_index in range(1, rounds + 1)
    )
    central_updates = tuple(
        round_central_budget(total_central_updates, rounds, round_index)
        for round_index in range(1, rounds + 1)
    )
    geometry = CampaignGeometry(
        scouts_per_round=scouts_per_round,
        rounds=rounds,
        scout_updates_by_round=scout_updates,
        central_updates_by_round=central_updates,
        rows_per_rl_update=prompts_per_update * generations_per_prompt,
    )
    if geometry.total_scout_updates != int(total_scout_updates):
        raise AssertionError("Scout partition lost or created RL updates")
    if geometry.total_central_updates != int(total_central_updates):
        raise AssertionError("Central partition lost or created RL updates")
    return geometry


def parse_round_schedule(spec: str, rounds: int) -> tuple[float, ...]:
    """Parse a comma-separated per-round novelty schedule.

    An empty string means that the caller should keep its scalar novelty
    weight.  A non-empty schedule must provide exactly one finite,
    non-negative value per round; silently recycling a short schedule would
    change the method.
    """

    import math

    text = str(spec or "").strip()
    if not text:
        return ()
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if len(values) != int(rounds):
        raise ValueError(
            f"round novelty schedule has {len(values)} values; expected {int(rounds)}"
        )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("round novelty weights must be finite and non-negative")
    return values


def novelty_weight_for_round(
    *,
    scalar_weight: float,
    schedule: str,
    rounds: int,
    round_index: int,
) -> float:
    """Resolve the Scout novelty weight for one 1-based round."""

    values = parse_round_schedule(schedule, rounds)
    if not values:
        value = float(scalar_weight)
        if value < 0.0:
            raise ValueError("scalar novelty weight must be non-negative")
        return value
    round_index = int(round_index)
    if not 1 <= round_index <= len(values):
        raise IndexError(f"round_index={round_index!r} outside 1..{len(values)}")
    return values[round_index - 1]
