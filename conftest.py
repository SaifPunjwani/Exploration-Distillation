"""Pytest rootdir anchor: puts the repo root on sys.path so tests import
expdis_torch / expdis_jax from a fresh clone without an install step."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
