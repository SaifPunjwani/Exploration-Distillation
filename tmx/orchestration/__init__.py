"""Production orchestration layer for TMX experiments.

The goal of this package is to keep infrastructure choices separate from
research logic. Backends can submit through XPK/GKE/MaxText/Tunix or a raw TPU
fallback, while experiment specs continue to describe rewards, novelty, data,
and evaluation in one place.
"""

from .contract import validate_dapo_drgrpo_contract
from .spec import ExperimentSpec, load_experiment_spec

__all__ = ["ExperimentSpec", "load_experiment_spec", "validate_dapo_drgrpo_contract"]
