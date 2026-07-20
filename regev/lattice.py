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
    """Nearest integer to a Fraction, exact, ties toward +infinity.

    The tie rule is arbitrary for LLL's purposes -- both choices keep
    |mu| <= 1/2 -- but it is fixed here so reduction is deterministic.
    """
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
        Reduced basis. LLL guarantees the first vector is short (within a
        2^((n-1)/2) factor of the true shortest); it does NOT sort the rows,
        so callers that want them ordered must sort themselves.
    """
    B = [list(map(int, r)) for r in B]
    n = len(B)
    if n <= 1:
        return B

    Bs, mu = gram_schmidt(B)
    norms = [_dot(v, v) for v in Bs]
    half = Fraction(1, 2)

    k = 1
    while k < n:
        # Size-reduce row k against rows k-1..0. Subtracting q*B[j] from B[k]
        # changes only row k of mu, and does so in closed form, so the whole
        # sweep runs without recomputing Gram-Schmidt. (The Gram-Schmidt
        # vectors themselves are untouched: B[k] moves only within the span
        # of B[0..k], leaving Bs[k] and every norm invariant.)
        for j in range(k - 1, -1, -1):
            if abs(mu[k][j]) > half:
                q = nearest_int(mu[k][j])
                B[k] = [bk - q * bj for bk, bj in zip(B[k], B[j])]
                mu[k][j] -= q
                for i in range(j):
                    mu[k][i] -= q * mu[j][i]

        if norms[k] >= (delta - mu[k][k - 1] ** 2) * norms[k - 1]:
            k += 1
        else:
            # Swapping rows k-1 and k reorders the span, so the Gram-Schmidt
            # data genuinely changes -- but it changes in closed form too
            # (Cohen, *A Course in Computational Algebraic Number Theory*,
            # Alg. 2.6.3), so no recomputation is needed here either. Only
            # column pair (k-1, k) of mu and the two norms are affected.
            m = mu[k][k - 1]
            nk = norms[k] + m * m * norms[k - 1]
            if nk == 0:
                # B[k-1] and B[k] both lie in the span of earlier rows: the
                # input was linearly dependent. Fall back to a full recompute
                # rather than divide by zero.
                B[k], B[k - 1] = B[k - 1], B[k]
                Bs, mu = gram_schmidt(B)
                norms = [_dot(v, v) for v in Bs]
                k = max(k - 1, 1)
                continue

            mu[k][k - 1] = m * norms[k - 1] / nk
            norms[k] = norms[k - 1] * norms[k] / nk
            norms[k - 1] = nk

            B[k], B[k - 1] = B[k - 1], B[k]
            for j in range(k - 1):
                mu[k][j], mu[k - 1][j] = mu[k - 1][j], mu[k][j]
            for i in range(k + 1, n):
                t = mu[i][k]
                mu[i][k] = mu[i][k - 1] - m * t
                mu[i][k - 1] = t + mu[k][k - 1] * mu[i][k]

            k = max(k - 1, 1)
    return B
