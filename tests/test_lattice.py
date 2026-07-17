"""Tests for exact LLL reduction.

These check the mathematical invariants of LLL rather than specific outputs,
so they stay valid if the reduction parameters change.
"""

import random
from fractions import Fraction

import pytest

from regev.lattice import lll, gram_schmidt, nearest_int


def det(M):
    """Exact integer determinant by fraction-free elimination."""
    import copy
    A = [[Fraction(x) for x in row] for row in copy.deepcopy(M)]
    n = len(A)
    d = Fraction(1)
    for i in range(n):
        piv = next((r for r in range(i, n) if A[r][i] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != i:
            A[i], A[piv] = A[piv], A[i]
            d = -d
        d *= A[i][i]
        inv = 1 / A[i][i]
        A[i] = [x * inv for x in A[i]]
        for r in range(i + 1, n):
            f = A[r][i]
            if f != 0:
                A[r] = [x - f * y for x, y in zip(A[r], A[i])]
    return d


def norm2(v):
    return sum(x * x for x in v)


def test_nearest_int():
    assert nearest_int(Fraction(1, 2)) == 1
    assert nearest_int(Fraction(-1, 2)) == 0
    assert nearest_int(Fraction(3, 2)) == 2
    assert nearest_int(Fraction(7, 3)) == 2
    assert nearest_int(Fraction(5)) == 5


def test_lll_known_basis():
    """Classic textbook example reduces to a known short basis."""
    B = [[1, 1, 1], [-1, 0, 2], [3, 5, 6]]
    R = lll(B)
    assert all(norm2(r) <= 6 for r in R)


def test_lll_preserves_determinant():
    """LLL is a unimodular transform: |det| is invariant."""
    random.seed(3)
    checked = 0
    for _ in range(40):
        d = random.randint(2, 4)
        B = [[random.randint(-50, 50) for _ in range(d)] for _ in range(d)]
        dB = det(B)
        if dB == 0:
            continue
        R = lll(B)
        assert abs(det(R)) == abs(dB)
        checked += 1
    assert checked > 10


def test_lll_shortens():
    """Reduced basis contains a vector no longer than the original shortest."""
    random.seed(11)
    for _ in range(20):
        d = random.randint(2, 4)
        B = [[random.randint(-80, 80) for _ in range(d)] for _ in range(d)]
        if det(B) == 0:
            continue
        R = lll(B)
        assert min(norm2(r) for r in R) <= min(norm2(b) for b in B)


def test_lll_size_reduced():
    """Output satisfies the LLL size-reduction condition |mu_ij| <= 1/2."""
    random.seed(5)
    B = [[random.randint(-40, 40) for _ in range(3)] for _ in range(3)]
    if det(B) != 0:
        R = lll(B)
        _, mu = gram_schmidt(R)
        for i in range(len(R)):
            for j in range(i):
                assert abs(mu[i][j]) <= Fraction(1, 2)


def test_lll_single_vector():
    assert lll([[3, 4]]) == [[3, 4]]
