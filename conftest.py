"""Make expdis_jax importable in tests from a fresh clone without installation."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
