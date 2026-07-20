"""Tests for the exact reference distribution, decoding, and sampling.

The reference distribution is the ground truth the whole experiment rests on,
so it is checked against independently-derived properties rather than against
stored numbers.
"""

import random

import pytest

from regev.reference import (
    bitstring_to_regev_vector,
    counts_to_distribution,
    ideal_regev_distribution,
    total_variation_distance,
)
from regev.sampling import ideal_distribution_cached, sample_ideal, sample_uniform

N77, D77, M77 = 77, 3, 32
BASES77 = [4, 9, 25]


@pytest.fixture(scope="module")
def dist77():
    return ideal_regev_distribution(BASES77, N77, D77, M77)


def test_distribution_is_normalised(dist77):
    assert sum(dist77.values()) == pytest.approx(1.0, abs=1e-9)


def test_distribution_is_nonnegative(dist77):
    assert min(dist77.values()) >= -1e-15


def test_distribution_covers_full_support(dist77):
    assert len(dist77) == M77 ** D77


def test_zero_vector_is_the_mode(dist77):
    """w = 0 is always the most likely outcome, and carries no information.

    Every group contributes |sum of ones|^2 at w = 0, so the mode is a
    structural fact, not an accident. A sampler that never returns it is not
    modelling the real circuit -- which is exactly the flaw in using a
    hand-picked top-k list.
    """
    mode = max(dist77, key=dist77.get)
    assert mode == (0, 0, 0)
    assert dist77[(0, 0, 0)] > 10 * (1.0 / M77 ** D77)


def test_distribution_far_from_uniform(dist77):
    """The whole premise: the ideal distribution is not the control."""
    uniform = 1.0 / M77 ** D77
    tvd = 0.5 * sum(abs(p - uniform) for p in dist77.values())
    assert tvd > 0.5, f"ideal distribution is near-uniform (TVD {tvd})"


def test_tvd_identical_is_zero(dist77):
    assert total_variation_distance(dist77, dist77) == pytest.approx(0.0, abs=1e-12)


def test_tvd_disjoint_is_one():
    assert total_variation_distance({(0,): 1.0}, {(1,): 1.0}) == pytest.approx(1.0)


def test_tvd_is_symmetric(dist77):
    q = {w: 1.0 / M77 ** D77 for w in dist77}
    assert total_variation_distance(dist77, q) == pytest.approx(
        total_variation_distance(q, dist77)
    )


# --- bitstring decoding -------------------------------------------------

def test_bitstring_little_endian_register_order():
    """Register 0 is the RIGHTMOST chunk in Qiskit's bit ordering."""
    # chunks left-to-right are registers 2, 1, 0
    assert bitstring_to_regev_vector("00010" "00001" "11111", 3, 5) == (31, 1, 2)


def test_bitstring_ignores_register_spaces():
    assert (bitstring_to_regev_vector("00010 00001 11111", 3, 5)
            == bitstring_to_regev_vector("000100000111111", 3, 5))


def test_bitstring_zero_and_max():
    assert bitstring_to_regev_vector("0" * 15, 3, 5) == (0, 0, 0)
    assert bitstring_to_regev_vector("1" * 15, 3, 5) == (31, 31, 31)


def test_bitstring_length_mismatch_raises():
    with pytest.raises(ValueError):
        bitstring_to_regev_vector("0101", 3, 5)


def test_counts_to_distribution_normalises():
    counts = {"0" * 15: 3, "1" * 15: 1}
    dist = counts_to_distribution(counts, 3, 5)
    assert dist[(0, 0, 0)] == pytest.approx(0.75)
    assert dist[(31, 31, 31)] == pytest.approx(0.25)
    assert sum(dist.values()) == pytest.approx(1.0)


# --- sampling -----------------------------------------------------------

def test_sample_ideal_shape_and_range():
    rng = random.Random(0)
    s = sample_ideal(BASES77, N77, D77, M77, 7, rng)
    assert len(s) == 7
    assert all(len(w) == D77 and all(0 <= x < M77 for x in w) for w in s)


def test_sample_ideal_is_reproducible():
    a = sample_ideal(BASES77, N77, D77, M77, 20, random.Random(42))
    b = sample_ideal(BASES77, N77, D77, M77, 20, random.Random(42))
    assert a == b


def test_sample_ideal_differs_across_seeds():
    a = sample_ideal(BASES77, N77, D77, M77, 30, random.Random(1))
    b = sample_ideal(BASES77, N77, D77, M77, 30, random.Random(2))
    assert a != b


def test_sample_ideal_matches_distribution(dist77):
    """Empirical distribution of many draws converges to the exact one."""
    rng = random.Random(11)
    n = 20000
    draws = sample_ideal(BASES77, N77, D77, M77, n, rng)
    emp = {}
    for w in draws:
        emp[w] = emp.get(w, 0) + 1 / n
    assert total_variation_distance(emp, dist77) < 0.10


def test_sample_ideal_does_draw_the_zero_vector():
    """A faithful sampler must produce the uninformative outcome sometimes."""
    draws = sample_ideal(BASES77, N77, D77, M77, 500, random.Random(3))
    assert (0, 0, 0) in draws


def test_cache_returns_consistent_data():
    a = ideal_distribution_cached(BASES77, N77, D77, M77)
    b = ideal_distribution_cached(tuple(BASES77), N77, D77, M77)
    assert a[0] == b[0] and a[1] == b[1]


def test_sample_uniform_shape_and_reproducibility():
    a = sample_uniform(D77, M77, 10, random.Random(5))
    b = sample_uniform(D77, M77, 10, random.Random(5))
    assert a == b and len(a) == 10
    assert all(len(w) == D77 and all(0 <= x < M77 for x in w) for w in a)
