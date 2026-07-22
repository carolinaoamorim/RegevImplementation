"""Tests for the circuit-free depolarising noise model.

The model is p_noisy = (1 - lam) p_ideal + lam * uniform. lam = 0 must recover
the ideal sampler exactly and lam = 1 the uniform control, so the noise sweep
genuinely interpolates the two arms of the main experiment.
"""

import random

import pytest

from regev.bases import generate_regev_bases
from regev.noise import (
    depolarizing_lambda,
    noisy_distribution,
    noisy_tvd_from_ideal,
    sample_noisy,
)
from regev.reference import total_variation_distance
from regev.sampling import ideal_probability_array, sample_ideal, sample_uniform

N77, D77, M77 = 77, 3, 32
BASES77 = [4, 9, 25]


# --- depolarizing_lambda -----------------------------------------------

def test_lambda_zero_error_is_zero():
    assert depolarizing_lambda(0.0, 30) == 0.0


def test_lambda_grows_with_qubits():
    a = depolarizing_lambda(0.01, 10)
    b = depolarizing_lambda(0.01, 30)
    assert 0 < a < b < 1


def test_lambda_matches_formula():
    assert depolarizing_lambda(0.1, 3) == pytest.approx(1 - 0.9 ** 3)


def test_lambda_rejects_bad_input():
    with pytest.raises(ValueError):
        depolarizing_lambda(1.5, 4)
    with pytest.raises(ValueError):
        depolarizing_lambda(0.1, -1)
    with pytest.raises(TypeError):
        depolarizing_lambda(True, 4)
    with pytest.raises(TypeError):
        noisy_distribution(BASES77, N77, D77, M77, "0.5")


# --- noisy_distribution -------------------------------------------------

def test_noisy_distribution_normalised():
    for lam in (0.0, 0.3, 1.0):
        arr = noisy_distribution(BASES77, N77, D77, M77, lam)
        assert arr.sum() == pytest.approx(1.0, abs=1e-9)
        assert arr.min() >= 0.0


def test_lam_zero_is_ideal():
    import numpy as np
    a = noisy_distribution(BASES77, N77, D77, M77, 0.0)
    b = ideal_probability_array(BASES77, N77, D77, M77)
    assert np.array_equal(a, b)


def test_lam_one_is_uniform():
    arr = noisy_distribution(BASES77, N77, D77, M77, 1.0)
    assert arr.min() == pytest.approx(arr.max())
    assert arr.flat[0] == pytest.approx(1.0 / M77 ** D77)


def test_noisy_distribution_rejects_bad_lambda():
    with pytest.raises(ValueError):
        noisy_distribution(BASES77, N77, D77, M77, 1.1)


# --- TVD ----------------------------------------------------------------

def test_tvd_is_linear_in_lambda():
    """Closed form: TVD(noisy, ideal) = lam * TVD(uniform, ideal)."""
    full = noisy_tvd_from_ideal(BASES77, N77, D77, M77, 1.0)
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        got = noisy_tvd_from_ideal(BASES77, N77, D77, M77, lam)
        assert got == pytest.approx(lam * full, abs=1e-9)


def test_tvd_zero_at_no_noise():
    assert noisy_tvd_from_ideal(BASES77, N77, D77, M77, 0.0) == pytest.approx(0.0)


def test_zero_noise_tvd_does_not_build_the_distribution(monkeypatch):
    import regev.noise as noise

    def fail(*args, **kwargs):
        pytest.fail("zero noise has zero TVD without allocating a distribution")

    monkeypatch.setattr(noise, "ideal_probability_array", fail)
    assert noisy_tvd_from_ideal(BASES77, N77, D77, M77, 0.0) == 0.0


# --- sampling -----------------------------------------------------------

def test_sample_noisy_shape():
    s = sample_noisy(BASES77, N77, D77, M77, 9, 0.3, random.Random(0))
    assert len(s) == 9
    assert all(len(w) == D77 and all(0 <= x < M77 for x in w) for w in s)


def test_sample_noisy_lam_zero_matches_ideal():
    """The zero-noise endpoint is exactly the ideal seeded sampler."""
    a = sample_noisy(BASES77, N77, D77, M77, 20, 0.0, random.Random(5))
    b = sample_ideal(BASES77, N77, D77, M77, 20, random.Random(5))
    assert a == b


def test_sample_noisy_lam_one_matches_uniform():
    """The full-noise endpoint is exactly the seeded uniform control."""
    a = sample_noisy(BASES77, N77, D77, M77, 20, 1.0, random.Random(5))
    b = sample_uniform(D77, M77, 20, random.Random(5))
    assert a == b


def test_sample_noisy_reproducible():
    a = sample_noisy(BASES77, N77, D77, M77, 30, 0.4, random.Random(9))
    b = sample_noisy(BASES77, N77, D77, M77, 30, 0.4, random.Random(9))
    assert a == b


def test_sample_noisy_rejects_negative_count():
    with pytest.raises(ValueError, match="non-negative integer"):
        sample_noisy(BASES77, N77, D77, M77, -1, 0.4, random.Random(9))


@pytest.mark.parametrize("lam", [0.25, 0.75])
def test_sample_noisy_validates_before_random_branch(lam):
    """Malformed inputs must fail independently of the branch RNG selects."""
    with pytest.raises(ValueError, match=r"len\(bases\)"):
        sample_noisy([], N77, D77, M77, 1, lam, random.Random(9))
    with pytest.raises(ValueError, match="window"):
        sample_noisy(
            BASES77,
            N77,
            D77,
            M77,
            0,
            lam,
            random.Random(9),
            window="invalid",
        )


def test_sample_noisy_converges_to_mixture():
    """Empirical frequencies of the high-probability cells match the mixture.

    Full-support TVD is a poor convergence check for this near-uniform mixture:
    with 32768 cells the finite-sample TVD floor swamps the signal. The
    high-probability cells (the mode and its neighbours) converge fast and are
    where the mixture identity is checked instead.
    """
    n = 20000
    arr = noisy_distribution(BASES77, N77, D77, M77, 0.35)
    draws = sample_noisy(BASES77, N77, D77, M77, n, 0.35, random.Random(3))
    emp = {}
    for w in draws:
        emp[w] = emp.get(w, 0) + 1 / n

    # w = 0 and the two next-largest cells carry enough mass to estimate at n.
    for w in [(0, 0, 0), (4, 2, 17), (28, 30, 15)]:
        assert emp.get(w, 0.0) == pytest.approx(float(arr[w]), abs=0.01)


def test_more_noise_lowers_factoring_success():
    """The experiment this model exists for: success falls as lam rises."""
    from regev.postprocess import regev_factor

    def rate(lam, T=120):
        rng = random.Random(7)
        return sum(
            regev_factor(sample_noisy(BASES77, N77, D77, M77, 8, lam, rng),
                         M77, D77, N77, BASES77, [2, 3, 5], verbose=False) is not None
            for _ in range(T)
        ) / T

    assert rate(0.0) > rate(0.5) > rate(1.0)
