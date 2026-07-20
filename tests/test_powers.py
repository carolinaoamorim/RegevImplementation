"""Tests for perfect-power detection and the input validation it feeds.

Prime powers are the case Regev's difference-of-squares step cannot handle at
all: u^2 = 1 (mod p^k) has only the roots u = +-1, so there is no nontrivial
square root to take a gcd against. Accepting such an N produces an unbounded
run that never succeeds, which is worse than rejecting it.
"""

import random

import pytest

from regev.bases import (
    generate_regev_bases,
    integer_nth_root,
    is_prime_power,
    perfect_power,
)
from regev.parameters import regev_parameters, validate_factor_input
from regev.postprocess import regev_factor
from regev.sampling import sample_ideal


# --- integer_nth_root ---------------------------------------------------

@pytest.mark.parametrize("x,k,expected", [
    (0, 3, 0), (1, 5, 1), (8, 3, 2), (9, 2, 3), (26, 3, 2), (27, 3, 3),
    (1023, 10, 1), (1024, 10, 2), (10 ** 18, 3, 10 ** 6),
])
def test_integer_nth_root_values(x, k, expected):
    assert integer_nth_root(x, k) == expected


def test_integer_nth_root_is_exact_floor():
    """r**k <= x < (r+1)**k, checked on randomised input."""
    rng = random.Random(0)
    for _ in range(300):
        x = rng.randrange(0, 10 ** 12)
        k = rng.randint(1, 12)
        r = integer_nth_root(x, k)
        assert r ** k <= x < (r + 1) ** k


def test_integer_nth_root_beats_float_precision():
    """The reason for integer Newton rather than x ** (1/k).

    2^53 + 1 cubed is not exactly representable as a double, so the naive
    float version rounds to the wrong root.
    """
    base = 2 ** 53 + 1
    assert integer_nth_root(base ** 3, 3) == base


def test_integer_nth_root_rejects_bad_args():
    with pytest.raises(ValueError):
        integer_nth_root(-1, 2)
    with pytest.raises(ValueError):
        integer_nth_root(8, 0)


# --- perfect_power ------------------------------------------------------

@pytest.mark.parametrize("N,expected", [
    (4, (2, 2)), (8, (2, 3)), (9, (3, 2)), (49, (7, 2)), (121, (11, 2)),
    (225, (15, 2)), (1000, (10, 3)),
])
def test_perfect_power_detected(N, expected):
    assert perfect_power(N) == expected


def test_perfect_power_returns_smallest_base():
    """64 = 8^2 = 4^3 = 2^6; the fully-reduced form is wanted."""
    assert perfect_power(64) == (2, 6)
    assert perfect_power(729) == (3, 6)


@pytest.mark.parametrize("N", [2, 3, 5, 15, 21, 77, 91, 143, 1001])
def test_non_powers_rejected(N):
    assert perfect_power(N) is None


def test_perfect_power_roundtrips():
    rng = random.Random(1)
    for _ in range(200):
        base = rng.randint(2, 500)
        exp = rng.randint(2, 6)
        got = perfect_power(base ** exp)
        assert got is not None
        b, e = got
        assert b ** e == base ** exp


def test_is_prime_power():
    for N in (2, 3, 4, 8, 9, 25, 49, 121, 2 ** 10):
        assert is_prime_power(N), N
    for N in (1, 6, 15, 77, 225, 1000):
        assert not is_prime_power(N), N


# --- validation ---------------------------------------------------------

def test_prime_power_rejected_with_factor():
    """N = 49 and N = 121 used to be accepted, then fail forever."""
    for N, base in ((49, 7), (121, 11)):
        res = validate_factor_input(N)
        assert res["valid"] is False
        assert "prime power" in res["reason"]
        assert res["factor"] == base
        assert res["factor"] * res["other"] == N


def test_composite_perfect_power_rejected():
    """225 = 15^2 is splittable by Regev, but isqrt already did it."""
    res = validate_factor_input(225)
    assert res["valid"] is False
    assert "perfect power" in res["reason"]
    assert res["factor"] * res["other"] == 225


def test_valid_instances_still_valid():
    for N in (15, 21, 57, 77, 91, 143):
        assert validate_factor_input(N)["valid"] is True


def test_existing_rejections_unchanged():
    assert validate_factor_input(1)["valid"] is False
    assert validate_factor_input(12)["factor"] == 2
    assert validate_factor_input(13)["valid"] is False


def test_rejected_prime_power_would_indeed_have_failed():
    """The justification for the rejection, measured rather than asserted.

    N = 49 is uncontaminated -- bases 4, 9, 25 are all coprime to it -- so
    nothing else in the pipeline screens it out. It simply never succeeds.
    """
    N, k, trials = 49, 16, 25
    params = regev_parameters(N)
    d, M = params["d"], params["M"]
    bases, primes, trivial = generate_regev_bases(N, d)
    assert not trivial, "N=49 should be uncontaminated; that is the point"

    rng = random.Random(0)
    successes = sum(
        regev_factor(sample_ideal(bases, N, d, M, k, rng), M, d, N,
                     bases, primes, verbose=False) is not None
        for _ in range(trials)
    )
    assert successes == 0, (
        f"N=49 factored {successes}/{trials} times -- if this ever fires, the "
        "prime-power rejection in validate_factor_input is wrong"
    )
