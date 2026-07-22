"""Tests for the exact reference distribution, decoding, and sampling.

The reference distribution is the ground truth the whole experiment rests on,
so it is checked against independently-derived properties rather than against
stored numbers.
"""

import math
import random

import pytest

from regev.reference import (
    bitstring_to_regev_vector,
    counts_to_distribution,
    ideal_regev_distribution,
    ideal_regev_distribution_array,
    total_variation_distance,
)
from regev.sampling import ideal_probability_array, sample_ideal, sample_uniform

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

    At zero frequency every positive Gaussian amplitude adds in phase; the
    triangle inequality bounds every other Fourier component by that value.
    The mode is therefore structural, not an accident.
    """
    mode = max(dist77, key=dist77.get)
    assert mode == (0, 0, 0)
    assert dist77[(0, 0, 0)] > 10 * (1.0 / M77 ** D77)


def test_distribution_far_from_uniform(dist77):
    """The whole premise: the ideal distribution is not the control."""
    uniform = 1.0 / M77 ** D77
    tvd = 0.5 * sum(abs(p - uniform) for p in dist77.values())
    assert tvd > 0.5, f"ideal distribution is near-uniform (TVD {tvd})"


def test_default_window_is_gaussian():
    """The public default is Regev's Gaussian, not the legacy rectangle."""
    import numpy as np

    radius = M77 / (2 * math.sqrt(D77))
    default = ideal_regev_distribution_array(BASES77, N77, D77, M77)
    explicit = ideal_regev_distribution_array(
        BASES77, N77, D77, M77, window="gaussian", radius=radius
    )
    uniform = ideal_regev_distribution_array(
        BASES77, N77, D77, M77, window="uniform"
    )
    assert np.array_equal(default, explicit)
    assert not np.allclose(default, uniform)


@pytest.mark.parametrize(
    "radius", [0.0, -1.0, float("inf"), float("nan"), True, "1.0"]
)
def test_gaussian_radius_must_be_finite_and_positive(radius):
    with pytest.raises((TypeError, ValueError)):
        ideal_regev_distribution_array(
            BASES77, N77, D77, M77, radius=radius
        )


def test_gaussian_radius_allows_sensitivity_outside_paper_window():
    max_radius = M77 / (2 * math.sqrt(D77))
    # The boundary is the default. Exact finite-window experiments may also
    # select radii on either side of Regev's preferred asymptotic interval.
    for radius in (max_radius, max_radius * 2, max_radius / 10):
        p = ideal_regev_distribution_array(
            BASES77, N77, D77, M77, radius=radius
        )
        assert p.sum() == pytest.approx(1.0, abs=1e-9)


def test_window_validation():
    with pytest.raises(ValueError):
        ideal_regev_distribution_array(
            BASES77, N77, D77, M77, window="rectangle"
        )
    with pytest.raises(ValueError):
        ideal_regev_distribution_array(
            BASES77, N77, D77, M77, window="uniform", radius=1.0
        )


def test_distribution_input_validation():
    with pytest.raises((TypeError, ValueError)):
        ideal_regev_distribution_array(BASES77, 1, D77, M77)
    with pytest.raises(TypeError):
        ideal_regev_distribution_array(BASES77, 77.0, D77, M77)
    with pytest.raises(ValueError):
        ideal_regev_distribution_array(BASES77, N77, 0, M77)
    with pytest.raises(ValueError):
        ideal_regev_distribution_array(BASES77, N77, D77, 0)
    with pytest.raises(ValueError):
        ideal_regev_distribution_array(BASES77 + [49], N77, D77, M77)
    with pytest.raises(ValueError):
        ideal_regev_distribution_array(BASES77[:-1], N77, D77, M77)
    with pytest.raises(TypeError):
        ideal_regev_distribution_array([4, 9, 2.5], N77, D77, M77)
    with pytest.raises(ValueError):
        ideal_regev_distribution_array([4, 9, 7], N77, D77, M77)


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


def test_bitstring_rejects_nonbinary_and_invalid_dimensions():
    with pytest.raises(ValueError):
        bitstring_to_regev_vector("0000x", 1, 5)
    with pytest.raises(ValueError):
        bitstring_to_regev_vector("00000", 0, 5)
    with pytest.raises(ValueError):
        bitstring_to_regev_vector("00000", 1, 0)
    with pytest.raises(TypeError):
        bitstring_to_regev_vector(0, 1, 1)


def test_counts_to_distribution_normalises():
    counts = {"0" * 15: 3, "1" * 15: 1}
    dist = counts_to_distribution(counts, 3, 5)
    assert dist[(0, 0, 0)] == pytest.approx(0.75)
    assert dist[(31, 31, 31)] == pytest.approx(0.25)
    assert sum(dist.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("counts", [
    {},
    {"0": 0},
    {"0": -1},
    {"0": 1.5},
    {"0": True},
])
def test_counts_to_distribution_rejects_invalid_counts(counts):
    with pytest.raises((TypeError, ValueError)):
        counts_to_distribution(counts, 1, 1)


def test_counts_to_distribution_rejects_nonmapping_and_bad_key():
    with pytest.raises(TypeError):
        counts_to_distribution([("0", 1)], 1, 1)
    with pytest.raises(ValueError):
        counts_to_distribution({"x": 1}, 1, 1)


# --- sampling -----------------------------------------------------------

def test_sample_ideal_shape_and_range():
    rng = random.Random(0)
    s = sample_ideal(BASES77, N77, D77, M77, 7, rng)
    assert len(s) == 7
    assert all(len(w) == D77 and all(0 <= x < M77 for x in w) for w in s)


def test_sampling_argument_validation():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        sample_ideal(BASES77, N77, D77, M77, -1, rng)
    with pytest.raises(TypeError):
        sample_ideal(BASES77, N77, D77, M77, 1.5, rng)
    with pytest.raises(TypeError):
        sample_ideal(BASES77, N77, D77, M77, 1, object())
    with pytest.raises(ValueError):
        sample_uniform(0, M77, 1, rng)
    with pytest.raises(ValueError):
        sample_uniform(D77, M77, -1, rng)
    with pytest.raises(TypeError):
        sample_uniform(D77, M77, 1, object())


def test_zero_samples_do_not_build_the_distribution(monkeypatch):
    import regev.sampling as sampling

    def fail(*args, **kwargs):
        pytest.fail("zero samples should not allocate a probability CDF")

    monkeypatch.setattr(sampling, "_cached_cdf", fail)
    assert sample_ideal(
        BASES77, N77, D77, M77, 0, random.Random(0)
    ) == []


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
    # The Gaussian output spreads mass across more cells than the old
    # rectangular model, so use enough draws to put the finite-support TVD
    # below this deliberately coarse convergence threshold reliably.
    n = 30000
    draws = sample_ideal(BASES77, N77, D77, M77, n, rng)
    emp = {}
    for w in draws:
        emp[w] = emp.get(w, 0) + 1 / n
    assert total_variation_distance(emp, dist77) < 0.10


def test_sample_ideal_does_draw_the_zero_vector():
    """A faithful sampler must produce the uninformative outcome sometimes."""
    draws = sample_ideal(BASES77, N77, D77, M77, 2000, random.Random(3))
    assert (0, 0, 0) in draws


def test_cache_returns_consistent_data():
    """List and tuple `bases` must hit the same cache entry."""
    import numpy as np

    a = ideal_probability_array(BASES77, N77, D77, M77)
    b = ideal_probability_array(tuple(BASES77), N77, D77, M77)
    assert np.array_equal(a, b)


def test_cached_probability_arrays_are_read_only_and_cache_is_bounded():
    from regev.sampling import _cached_cdf, _cached_probability

    p = ideal_probability_array(BASES77, N77, D77, M77)
    assert not p.flags.writeable
    with pytest.raises(ValueError):
        p.flat[0] = 0.0
    assert _cached_cdf.cache_info().maxsize == 2
    assert _cached_probability.cache_info().maxsize == 2


def test_array_and_dict_forms_agree(dist77):
    """The dict wrapper must not drift from the array it wraps."""
    p = ideal_probability_array(BASES77, N77, D77, M77)
    assert p.shape == (M77,) * D77
    assert max(abs(float(p[w]) - dist77[w]) for w in dist77) < 1e-15


def test_single_fft_matches_grouped_gaussian_construction():
    """Compare the fast lag autocorrelation to the Gaussian circuit directly.

    This oracle starts from Regev's actual Gaussian amplitudes, groups exponent
    vectors by modular-exponentiation image, and Fourier-transforms every group
    independently.  It therefore does not reuse the lag-autocorrelation
    identity implemented by the production path.
    """
    import numpy as np
    from itertools import product as iproduct

    bases, N, d, M = [4, 9, 25], 77, 3, 8
    radius = M / (2 * math.sqrt(d))
    zs = range(-M // 2, M // 2)
    one_dim_norm = sum(
        math.exp(-2 * math.pi * z * z / (radius * radius)) for z in zs
    )
    state_norm = one_dim_norm ** d
    groups = {}
    for z in iproduct(zs, repeat=d):
        v = 1
        for ai, zi in zip(bases, z):
            v = v * pow(ai, zi, N) % N
        amplitude = math.exp(
            -math.pi * sum(zi * zi for zi in z) / (radius * radius)
        ) / math.sqrt(state_norm)
        groups.setdefault(v, []).append(
            (tuple(zi + M // 2 for zi in z), amplitude)
        )

    expected = np.zeros((M,) * d)
    for entries in groups.values():
        arr = np.zeros((M,) * d, dtype=complex)
        for index, amplitude in entries:
            arr[index] = amplitude
        expected += np.abs(np.fft.fftn(arr, norm="ortho")) ** 2

    got = ideal_regev_distribution_array(
        bases, N, d, M, radius=radius
    )
    assert np.abs(expected - got).max() < 1e-12


def test_uniform_window_matches_legacy_per_class_construction():
    """The legacy rectangular model remains available only by explicit opt-in."""
    import numpy as np
    from itertools import product as iproduct

    bases, N, d, M = [4, 9, 25], 77, 3, 8
    groups = {}
    for x in iproduct(range(M), repeat=d):
        v = 1
        for ai, xi in zip(bases, x):
            v = v * pow(ai, xi, N) % N
        groups.setdefault(v, []).append(x)

    expected = np.zeros((M,) * d)
    for xs in groups.values():
        arr = np.zeros((M,) * d, dtype=complex)
        for x in xs:
            arr[x] = 1.0
        expected += np.abs(np.fft.fftn(arr, norm="ortho")) ** 2 / M ** d

    got = ideal_regev_distribution_array(
        bases, N, d, M, window="uniform"
    )
    assert np.abs(expected - got).max() < 1e-12


def test_wide_modulus_uses_exact_object_path():
    """N above the int64 threshold must still compute exactly, not raise.

    The residue products reach N^2 > 2^63, so the engine falls back to
    object-dtype Python ints. Correctness is cross-checked against the int64
    path by forcing the same small instance down both.
    """
    import numpy as np

    import regev.reference as R

    N = 300_000_000_007  # ~3e11, N^2 well past 2^63; coprime to 4 and 9
    assert N > R._MAX_MODULUS and N * N > 2 ** 63
    bases = [4, 9]  # coprime to N (N is not divisible by 2 or 3)
    p = R.ideal_regev_distribution_array(bases, N, 2, 8)
    assert p.shape == (8, 8)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert p.min() >= 0.0

    # object path must equal int64 path on a small, in-range instance
    fast = R.ideal_regev_distribution_array([4, 9], 77, 2, 16)
    saved = R._MAX_MODULUS
    try:
        R._MAX_MODULUS = 1  # force object dtype
        slow = R.ideal_regev_distribution_array([4, 9], 77, 2, 16)
    finally:
        R._MAX_MODULUS = saved
    assert np.abs(fast - slow).max() < 1e-15


def test_sample_uniform_shape_and_reproducibility():
    a = sample_uniform(D77, M77, 10, random.Random(5))
    b = sample_uniform(D77, M77, 10, random.Random(5))
    assert a == b and len(a) == 10
    assert all(len(w) == D77 and all(0 <= x < M77 for x in w) for w in a)


def test_large_instance_is_tractable():
    """The single-FFT engine must reach instances the old one could not.

    N = 2021 has d = 4, M = 64, so the support is 16.7M points. The previous
    per-residue-class construction needed one FFT per class over an array that
    size, plus an M^d Python loop inside each. This asserts the whole
    distribution is now built in one pass and is a valid probability vector.
    """
    import numpy as np

    from regev.reference import ideal_regev_distribution_array

    p = ideal_regev_distribution_array([4, 9, 25, 49], 2021, 4, 64)
    assert p.shape == (64,) * 4
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert p.min() >= 0.0
    assert np.unravel_index(int(p.argmax()), p.shape) == (0, 0, 0, 0)
