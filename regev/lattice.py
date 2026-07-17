"""Exact lattice basis reduction (LLL) over the rationals.

Deliberately dependency-free: no fpylll, no numpy, no Qiskit. For the
dimensions used here (d = 3..5 plus k sample rows) exact rational
arithmetic is instant and avoids floating-point drift entirely.
"""

from fractions import Fraction

__all__ = ["lll", "gram_schmidt", "nearest_int"]


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def gram_schmidt(B):
    """Exact Gram-Schmidt. Returns (orthogonal basis, mu coefficients)."""
    n = len(B)
    Bs, mu = [], [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        v = [Fraction(x) for x in B[i]]
        for j in range(i):
            dj = _dot(Bs[j], Bs[j])
            if dj == 0:
                continue
            mu[i][j] = _dot([Fraction(x) for x in B[i]], Bs[j]) / dj
            v = [vk - mu[i][j] * bk for vk, bk in zip(v, Bs[j])]
        Bs.append(v)
    return Bs, mu


def nearest_int(f: Fraction) -> int:
    """Nearest integer to a Fraction, exact, ties away from zero."""
    n, d = f.numerator, f.denominator
    q, r = divmod(n, d)
    if 2 * r >= d:
        q += 1
    return q


def lll(B, delta=Fraction(99, 100)):
    """Exact LLL reduction of an integer lattice basis given as row vectors.

    Args:
        B: list of integer row vectors.
        delta: Lovasz parameter in (1/4, 1). 0.99 gives stronger reduction
            than the classical 0.75; cost is irrelevant at these dimensions.

    Returns:
        Reduced basis, shortest vectors first.
    """
    B = [list(map(int, r)) for r in B]
    n = len(B)
    if n <= 1:
        return B
    Bs, mu = gram_schmidt(B)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            if abs(mu[k][j]) > Fraction(1, 2):
                q = nearest_int(mu[k][j])
                B[k] = [bk - q * bj for bk, bj in zip(B[k], B[j])]
                Bs, mu = gram_schmidt(B)
        if _dot(Bs[k], Bs[k]) >= (delta - mu[k][k - 1] ** 2) * _dot(Bs[k - 1], Bs[k - 1]):
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            Bs, mu = gram_schmidt(B)
            k = max(k - 1, 1)
    return B
