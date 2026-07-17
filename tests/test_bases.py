"""Tests for base generation, including the contamination regression.

The key test here is test_old_contaminated_instances_are_flagged: it locks in
the fix for the bug where N = 15, 21, 57 were "factored" by trial division
during base selection, before the quantum circuit ran.
"""

from math import gcd

import pytest

from regev.bases import is_prime_basic, generate_regev_bases, is_contaminated


def test_is_prime_basic():
    assert not is_prime_basic(0)
    assert not is_prime_basic(1)
    assert is_prime_basic(2)
    assert is_prime_basic(3)
    assert not is_prime_basic(4)
    assert is_prime_basic(97)
    assert not is_prime_basic(91)  # 7 * 13
    assert is_prime_basic(7919)


def test_bases_are_squares_of_primes():
    bases, primes, _ = generate_regev_bases(77, 3)
    assert bases == [p ** 2 for p in primes]
    assert all(is_prime_basic(p) for p in primes)


def test_bases_coprime_to_N():
    for N in (77, 91, 143, 2021):
        for d in (2, 3, 4):
            bases, _, _ = generate_regev_bases(N, d)
            assert len(bases) == d
            for a in bases:
                assert gcd(a, N) == 1


@pytest.mark.parametrize("N,expected_trivial", [
    (15, [3, 5]),
    (21, [3, 7]),
    (57, [3]),
])
def test_old_contaminated_instances_are_flagged(N, expected_trivial):
    """REGRESSION: these N were previously reported as quantum successes.

    Trial division during base selection finds a factor of N before the
    circuit runs. generate_regev_bases must surface this rather than
    silently returning it as a 'lucky factor' result.
    """
    _, _, trivial = generate_regev_bases(N, 3)
    assert trivial, f"N={N} must be flagged as contaminated"
    for p in expected_trivial:
        if p in trivial:
            assert N % p == 0


def test_77_is_clean():
    """N = 77 = 7 x 11 is the smallest clean instance at d = 3."""
    bases, primes, trivial = generate_regev_bases(77, 3)
    assert trivial == []
    assert bases == [4, 9, 25]
    assert primes == [2, 3, 5]
    assert not is_contaminated(77, 3)


def test_contamination_helper():
    assert is_contaminated(15, 2)
    assert is_contaminated(57, 3)
    assert not is_contaminated(77, 3)
    assert not is_contaminated(2021, 4)
