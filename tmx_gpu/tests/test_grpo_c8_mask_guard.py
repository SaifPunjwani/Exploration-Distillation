"""c8 must fail closed on Overlong Filtering until baseline consistency exists.

The measured c7 defect: with --mask-truncated, capped rows contribute no
gradient but their penalty rewards stay in the group baseline (advantages
inflate as clip rises; wrong survivors sign-flip positive) and eligibility is
judged pre-mask (documented in the c8 audit addendum, kept outside this repo).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tmx_gpu import grpo_gpu


def _args(*extra: str):
    return grpo_gpu.build_arg_parser().parse_args(
        ["--output-dir", "/tmp/grpo-c8-mask-guard", *extra])


def test_c8_rejects_overlong_filtering_without_baseline_consistency():
    with pytest.raises(ValueError, match="mask-truncated"):
        grpo_gpu.validate_rl_validity(
            _args("--rl-validity-mode", "c8", "--mask-truncated"),
            SimpleNamespace(managed=True))


def test_c8_default_unmasked_path_still_validates():
    grpo_gpu.validate_rl_validity(
        _args("--rl-validity-mode", "c8"), SimpleNamespace(managed=True))


def test_legacy_masking_replays_remain_allowed():
    # exact-c7 replays: legacy mode returns before any c8 gate.
    grpo_gpu.validate_rl_validity(
        _args("--mask-truncated"), SimpleNamespace(managed=False))
