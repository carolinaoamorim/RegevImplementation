"""Regev's classical post-processing: measured vectors -> relations -> factors.

This is the step that distinguishes Regev from d parallel runs of Shor. There
is no order-finding and no continued-fraction expansion anywhere in this file.

Each measured vector w in Z_M^d is approximately orthogonal to every exponent
vector e satisfying prod a_i^{e_i} = 1 (mod N). The relations therefore live in
the kernel lattice

    L = { e in Z^d : <w_j, e> ~= 0 (mod M) for all measured w_j }

and short vectors of L are the relations. A single short vector combines all d
dimensions at once -- that is the whole point of the algorithm.

Imports nothing from Qiskit, so the post-processing can be tested against
classically generated samples with no simulator present.
"""

from math import gcd, isfinite, sqrt
from numbers import Real
from operator import index
import warnings

from regev.lattice import lll

__all__ = [
    "regev_embedding_scale",
    "regev_relation_lattice",
    "eval_relation",
    "is_relation",
    "relation_to_congruence",
    "regev_factor",
]


def _integer_arg(value, name: str) -> int:
    """Return an integer argument as a built-in int, without truncation."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None


def _positive_int_arg(value, name: str, *, minimum: int = 1) -> int:
    value = _integer_arg(value, name)
    if value < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive")
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _integer_vector(values, name: str):
    try:
        values = list(values)
    except TypeError:
        raise TypeError(f"{name} must be an iterable of integers") from None
    return [_integer_arg(value, f"{name}[{i}]") for i, value in enumerate(values)]


def _relation_inputs(e, values, N, values_name: str):
    N = _positive_int_arg(N, "N", minimum=2)
    e = _integer_vector(e, "e")
    values = _integer_vector(values, values_name)
    if not e:
        raise ValueError("relations must have at least one dimension")
    if len(e) != len(values):
        raise ValueError(
            f"e and {values_name} must have the same length; "
            f"got {len(e)} and {len(values)}"
        )
    return e, values, N


def _measurement_vectors(vectors, d: int, M: int):
    try:
        vectors = list(vectors)
    except TypeError:
        raise TypeError("vectors must be an iterable of measured vectors") from None
    if not vectors:
        raise ValueError("vectors must contain at least one measured vector")

    validated = []
    for j, vector in enumerate(vectors):
        vector = _integer_vector(vector, f"vectors[{j}]")
        if len(vector) != d:
            raise ValueError(
                f"vectors[{j}] must have length d = {d}; got {len(vector)}"
            )
        for i, value in enumerate(vector):
            if not 0 <= value < M:
                raise ValueError(
                    f"vectors[{j}][{i}] must be in [0, M); got {value} for M = {M}"
                )
        validated.append(vector)
    return validated


def regev_embedding_scale(M: int, d: int, radius=None) -> int:
    """Return the integer exponent scale for Regev's lattice embedding.

    Regev's normalized embedding uses noisy dual samples ``w=t/M`` and a
    scale ``S=1/delta``, where ``delta=sqrt(d)/(sqrt(2)*R)``. Multiplying the
    whole basis by ``M/S`` gives the integer form used here, with exponent
    scale ``C=M*delta``. Rounding C is the only finite-precision choice.
    """
    M = _positive_int_arg(M, "M")
    d = _positive_int_arg(d, "d")
    if radius is None:
        radius = M / (2 * sqrt(d))
    if isinstance(radius, bool) or not isinstance(radius, Real):
        raise TypeError("radius must be a real number")
    radius = float(radius)
    if not isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")

    delta = sqrt(d) / (sqrt(2) * radius)
    scale = round(M * delta)
    if scale < 1:
        raise ValueError(
            "radius is too large for the integer embedding: M*delta rounds to "
            "zero; use a smaller radius or provide exponent_scale explicitly"
        )
    return scale


def regev_relation_lattice(
    vectors,
    M: int,
    d: int,
    weight=None,
    *,
    radius=None,
    exponent_scale=None,
):
    """Build and reduce the relation lattice from measured vectors.

    Because M = 2^nd and the multiplicative orders generally do not divide M,
    the orthogonality <w, e> = 0 (mod M) holds only APPROXIMATELY. Demanding
    exact congruence returns nothing but the trivial lattice M*Z^d.

    We use the integer-scaled form of Regev's embedding

        rows:  [  C e_i  |  w_j[i]       ]   for i = 0..d-1
               [    0    |  M (diagonal) ]   for j = 0..k-1

    where ``C = round(M*sqrt(d)/(sqrt(2)*R))`` by default. This is obtained
    from the paper's normalized samples and error scale by multiplying through
    to keep an integer basis. A short reduced vector has both a short exponent
    block and small near-orthogonality residuals.

    Args:
        vectors: measured vectors, each of length d, entries in [0, M).
        M: Fourier modulus per dimension (2^nd).
        d: number of dimensions.
        exponent_scale: optional positive integer C. By default it is derived
            from M, d, and radius using :func:`regev_embedding_scale`.
        radius: Gaussian radius used by the sampler. Defaults to
            ``M/(2*sqrt(d))`` and matters only when deriving C.
        weight: deprecated compatibility option for the old rectangular
            embedding. Supplying it exactly reproduces rows ``[I | weight*t]``
            and ``[0 | weight*M I]``. It cannot be combined with radius or
            exponent_scale.

    Returns:
        List of (L1_norm, e, residual) sorted by increasing norm.
    """
    M = _positive_int_arg(M, "M")
    d = _positive_int_arg(d, "d")
    legacy_weight = None
    if weight is not None:
        if exponent_scale is not None or radius is not None:
            raise ValueError("weight cannot be combined with radius or exponent_scale")
        legacy_weight = _positive_int_arg(weight, "weight")
        warnings.warn(
            "weight is deprecated; omit it to use Regev's Gaussian-error "
            "embedding or pass exponent_scale explicitly",
            DeprecationWarning,
            stacklevel=2,
        )
    elif exponent_scale is None:
        exponent_scale = regev_embedding_scale(M, d, radius)
    else:
        exponent_scale = _positive_int_arg(exponent_scale, "exponent_scale")
    vectors = _measurement_vectors(vectors, d, M)

    k = len(vectors)
    rows = []
    for i in range(d):
        if legacy_weight is None:
            left = [exponent_scale if t == i else 0 for t in range(d)]
            right = [vectors[j][i] for j in range(k)]
        else:
            left = [1 if t == i else 0 for t in range(d)]
            right = [legacy_weight * vectors[j][i] for j in range(k)]
        rows.append(left + right)
    for j in range(k):
        diagonal = M if legacy_weight is None else legacy_weight * M
        rows.append([0] * d + [diagonal if t == j else 0 for t in range(k)])

    reduced = lll(rows)

    cands = []
    for r in reduced:
        left, resid = r[:d], r[d:]
        if legacy_weight is None:
            if any(value % exponent_scale for value in left):
                raise ArithmeticError("LLL left the integer embedding lattice")
            e = [value // exponent_scale for value in left]
        else:
            e = left
        if any(e):
            cands.append((sum(abs(x) for x in e), e, resid))
    cands.sort()
    return cands


def eval_relation(e, bases, N: int):
    """Evaluate prod a_i^{e_i} mod N, handling negative exponents.

    Returns None if the negative part is not invertible mod N.
    """
    e, bases, N = _relation_inputs(e, bases, N, "bases")
    num, den = 1, 1
    for ai, ei in zip(bases, e, strict=True):
        if ei > 0:
            num = num * pow(ai, ei, N) % N
        elif ei < 0:
            den = den * pow(ai, -ei, N) % N
    if gcd(den, N) != 1:
        return None
    return num * pow(den, -1, N) % N


def is_relation(e, bases, N: int) -> bool:
    """True if prod a_i^{e_i} = 1 (mod N)."""
    return eval_relation(e, bases, N) == 1 % N


def relation_to_congruence(e, primes, N: int):
    """Turn a verified relation into a factor via difference of squares.

    This is why Regev uses squared bases. With a_i = p_i^2, a relation
    prod a_i^{e_i} = 1 mod N means prod p_i^{2 e_i} = 1 mod N, so

        u = prod p_i^{e_i}   satisfies   u^2 = 1 (mod N).

    A nontrivial square root of unity (u != +-1) splits N via gcd(u - 1, N).

    Returns (p, q) or None if the relation gives only a trivial square root.
    """
    e, primes, N = _relation_inputs(e, primes, N, "primes")
    num, den = 1, 1
    for p, ei in zip(primes, e, strict=True):
        if ei > 0:
            num = num * pow(p, ei, N) % N
        elif ei < 0:
            den = den * pow(p, -ei, N) % N

    if gcd(den, N) != 1:
        g = gcd(den, N)
        return (g, N // g) if 1 < g < N else None

    u = num * pow(den, -1, N) % N

    if u * u % N != 1 % N:
        return None  # relation did not lift to a square root of unity
    if u == 1 or u == N - 1:
        return None  # trivial square root; try another relation

    for cand in (gcd(u - 1, N), gcd(u + 1, N)):
        if 1 < cand < N:
            return (cand, N // cand)
    return None


def regev_factor(
    vectors,
    M,
    d,
    N,
    bases,
    primes,
    weight=None,
    verbose=True,
    *,
    radius=None,
    exponent_scale=None,
):
    """Full pipeline: vectors -> relation lattice -> LLL -> sqrt -> gcd."""
    N = _positive_int_arg(N, "N", minimum=2)
    d = _positive_int_arg(d, "d")
    bases = _integer_vector(bases, "bases")
    primes = _integer_vector(primes, "primes")
    if len(bases) != d or len(primes) != d:
        raise ValueError(
            "bases and primes must each have length d; "
            f"got d = {d}, len(bases) = {len(bases)}, len(primes) = {len(primes)}"
        )
    for i, (base, prime) in enumerate(zip(bases, primes, strict=True)):
        if base % N != pow(prime, 2, N):
            raise ValueError(
                f"bases[{i}] must equal primes[{i}] squared modulo N"
            )

    cands = regev_relation_lattice(
        vectors,
        M,
        d,
        weight=weight,
        exponent_scale=exponent_scale,
        radius=radius,
    )
    rels = [e for _, e, _ in cands if is_relation(e, bases, N)]

    if verbose:
        print("=== Regev lattice post-processing ===")
        print(f"N = {N} | d = {d} | M = {M} | bases = {bases}")
        print("LLL basis vectors:", [e for _, e, _ in cands])
        print("verified relations:", rels)

    # Check each relation once. The map e -> prod p_i**e_i (mod N) is a group
    # homomorphism. If two relations each yield a trivial root (+-1), their sum
    # yields the product of those roots, which is still +-1. Pairwise sums can
    # therefore never turn individually trivial roots into a factor.
    for e in rels:
        res = relation_to_congruence(e, primes, N)
        if res:
            p, q = res
            if p * q == N:
                if verbose:
                    print(f"\nSUCCESS: N = {N} = {p} x {q}")
                    print("via relation e =", e)
                return (p, q)

    if verbose:
        print("\nNo factors recovered from these samples.")
    return None
