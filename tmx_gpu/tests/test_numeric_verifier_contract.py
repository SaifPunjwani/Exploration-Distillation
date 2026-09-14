"""Golden and adversarial cases for the publication numeric verifier."""

from __future__ import annotations

from fractions import Fraction

import pytest

from tmx_jax import rewarding
from tmx_gpu import eval_gpu


@pytest.mark.parametrize(
    ("completion", "ground_truth", "canonical"),
    [
        (r"Thus \boxed{\frac{1}{2}}.", "1/2", "1/2"),
        (r"Thus \boxed { -\frac{ 1 }{ 2 } }.", "-0.5", "-1/2"),
        (r"Thus \boxed{\frac{-1}{2}}.", "-.5", "-1/2"),
        (r"Thus \boxed{\frac{1}{-2}}.", "-5e-1", "1/-2"),
        (r"Thus \boxed{\frac{\frac{1}{2}}{3}}.", "1/6", r"\frac{1/2}{3}"),
        ("The final answer is 4.5e33.", "4.5E+33", "4.5e33"),
        ("Final answer: -1.25E-3.", "-0.00125", "-1.25e-3"),
        (r"The answer is \frac{3}{ 4 }.", "0.75", "3/4"),
        (r"Answer: 4.5 \times 10^{33}.", "4.5e33", r"4.5\times10^{33}"),
        ("Answer: .5.", "1/2", ".5"),
        ("#### 1,024", "1024", "1024"),
    ],
)
def test_numeric_verifier_golden_equivalences(
    completion: str, ground_truth: str, canonical: str,
) -> None:
    assert rewarding.extract_presented_answer(completion) == canonical
    assert rewarding.is_correct_completion(completion, ground_truth)
    prediction, correct = eval_gpu.grade(completion, ground_truth)
    assert prediction == canonical
    assert correct is True


@pytest.mark.parametrize(
    "completion",
    [
        "Final answer: 4.5e",
        "Final answer: 1/2/3",
        "Final answer: 1e3foo",
        "Final answer: x/2",
        "Final answer: 1/0",
        r"Final answer: \frac{1}{0}",
    ],
)
def test_numeric_verifier_rejects_malformed_or_symbolic_prefixes(
    completion: str,
) -> None:
    assert rewarding.extract_presented_answer(completion) is None
    assert not rewarding.is_correct_completion(completion, "1")
    assert eval_gpu.grade(completion, "1") == (None, False)


def test_numeric_verifier_does_not_search_uncued_prose_for_a_number() -> None:
    completion = "There are 4.5e33 possibilities in this intermediate estimate."
    assert not rewarding.is_correct_completion(completion, "4.5e33")
    assert eval_gpu.grade(completion, "4.5e33")[1] is False


def test_protocol_fraction_normalization_is_literal_a_over_b() -> None:
    assert rewarding._normalize_text_answer(r"\frac{ 1 }{ 2 }") == "1/2"


@pytest.mark.parametrize(
    ("completion", "ground_truth"),
    [
        (r"Final answer: 9007199254740993", "9007199254740992"),
        (
            r"Final answer: 1000000000000000000000000000001",
            "1e30",
        ),
        (r"Final answer: \frac{1}{3}", "0.333331"),
        ("Final answer: 1.0000011", "1"),
    ],
)
def test_exact_numeric_representation_rejects_float_collapse_and_outside_tolerance(
    completion: str, ground_truth: str,
) -> None:
    assert not rewarding.is_correct_completion(completion, ground_truth)
    assert eval_gpu.grade(completion, ground_truth)[1] is False


@pytest.mark.parametrize(
    ("completion", "ground_truth"),
    [
        (r"Final answer: \frac{9007199254740993}{3}", "3002399751580331"),
        (r"Final answer: \frac{1}{10}", "0.1"),
        (r"Final answer: \frac{1}{3}", "0.333333"),
        ("Final answer: 1.000001", "1"),
    ],
)
def test_exact_fractions_and_documented_absolute_tolerance(
    completion: str, ground_truth: str,
) -> None:
    assert rewarding.is_correct_completion(completion, ground_truth)
    assert eval_gpu.grade(completion, ground_truth)[1] is True


def test_numeric_parser_returns_exact_fraction_not_binary_float() -> None:
    assert rewarding._parse_simple_numeric("0.1") == Fraction(1, 10)
    assert rewarding._parse_simple_numeric("1e-3") == Fraction(1, 1000)
    assert rewarding._parse_simple_numeric("9007199254740993") == Fraction(
        9007199254740993, 1)


def test_unbounded_scientific_literal_is_rejected_before_bigint_expansion() -> None:
    completion = "Final answer: 1e1001"
    assert rewarding.extract_presented_answer(completion) is None
    assert eval_gpu.grade(completion, "1") == (None, False)


def test_verifier_version_identifies_post_float_fix_contract() -> None:
    assert eval_gpu.REWARDING_CONTRACT_VERSION == (
        "tmx_jax.rewarding@gpu_repro_v2_exact_rational")
