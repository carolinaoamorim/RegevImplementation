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

from itertools import combinations
from math import gcd

from regev.lattice import lll

__all__ = [
    "regev_relation_lattice",
    "eval_relation",
    "is_relation",
    "relation_to_congruence",
    "regev_factor",
]


def regev_relation_lattice(vectors, M: int, d: int, weight: int = 4):
    """Build and reduce the relation lattice from measured vectors.

    Because M = 2^nd and the multiplicative orders generally do not divide M,
    the orthogonality <w, e> = 0 (mod M) holds only APPROXIMATELY. Demanding
    exact congruence returns nothing but the trivial lattice M*Z^d.

    Instead we use the weighted embedding

        rows:  [  e_i  |  weight * w_j[i]   ]   for i = 0..d-1
               [   0   |  weight * M (diag) ]   for j = 0..k-1

    and LLL-reduce. A short reduced vector has a small left block (a short
    exponent vector) AND a small right block (near-orthogonality to every
    measured sample), which is exactly a candidate relation.

    Args:
        vectors: measured vectors, each of length d, entries in [0, M).
        M: Fourier modulus per dimension (2^nd).
        d: number of dimensions.
        weight: scaling on the residual block. Larger values penalise
            non-orthogonality more strongly, at the cost of letting the
            exponent block grow.

            The choice matters more than "stable for 1..8" (the previous
            claim here) suggests. Measured on N = 77 over 1000 trials on
            held-out seeds, ideal-sampling success at k = 3 runs 95.7% (w=1),
            95.4% (w=2), 93.5% (w=4), 90.3% (w=8). No value dominates across
            k, though: w=2 wins at k=3, ties at k=5, and gives up ~8 points of
            separation against the random control at k=8 because it also lifts
            the control. The default stays 4 for that reason, not because the
            result is insensitive to it.

            Sweeping weights and taking any success is NOT a free win -- see
            the note in `regev_factor`.

    Returns:
        List of (L1_norm, e, residual) sorted by increasing norm.
    """
    k = len(vectors)
    rows = []
    for i in range(d):
        left = [1 if t == i else 0 for t in range(d)]
        right = [weight * (int(vectors[j][i]) % M) for j in range(k)]
        rows.append(left + right)
    for j in range(k):
        rows.append([0] * d + [weight * M if t == j else 0 for t in range(k)])

    reduced = lll(rows)

    cands = []
    for r in reduced:
        e, resid = r[:d], r[d:]
        if any(e):
            cands.append((sum(abs(x) for x in e), e, resid))
    cands.sort()
    return cands


def eval_relation(e, bases, N: int):
    """Evaluate prod a_i^{e_i} mod N, handling negative exponents.

    Returns None if the negative part is not invertible mod N.
    """
    num, den = 1, 1
    for ai, ei in zip(bases, e):
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
    num, den = 1, 1
    for p, ei in zip(primes, e):
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


def regev_factor(vectors, M, d, N, bases, primes, weight=4, verbose=True):
    """Full pipeline: vectors -> relation lattice -> LLL -> sqrt -> gcd."""
    cands = regev_relation_lattice(vectors, M, d, weight=weight)
    rels = [e for _, e, _ in cands if is_relation(e, bases, N)]

    if verbose:
        print("=== Regev lattice post-processing ===")
        print(f"N = {N} | d = {d} | M = {M} | bases = {bases}")
        print("LLL basis vectors:", [e for _, e, _ in cands])
        print("verified relations:", rels)

    # Single relations first, then pairwise sums -- a given relation often
    # yields only a trivial square root (u = +-1), so combinations are needed.
    # Unordered pairs suffice: r + s == s + r.
    #
    # Three tempting extensions were implemented, measured, and REJECTED. Each
    # raises the headline success rate; all three raise the RANDOM control by
    # as much or more, which means the extra work is being done classically
    # rather than by the quantum samples. Success rate alone is the wrong
    # metric here -- the separation against the control is the result.
    #
    #   pairwise differences   ideal 99.0 -> 99.0, no change at all (relations
    #                          form a group; the reduced basis already spans it)
    #   short-vector enumeration (all +-1 combinations of the 5 shortest basis
    #                          vectors)   ideal 99.0 -> 100.0, but random
    #                          26.2 -> 95.0. Separation collapses 73 -> 5
    #                          points: it brute-forces the relation space and
    #                          factors N with no quantum input at all.
    #   weight ensemble (retry over w = 1,2,4,8,16 and accept any success)
    #                          ideal 99.0 -> 99.8, random 26.2 -> 51.0.
    #                          Separation 73 -> 49. Multiple testing.
    #
    # Do not add these back without re-running scripts/control_experiment.py
    # on BOTH arms.
    pool = list(rels)
    for r, s in combinations(rels, 2):
        pool.append([x + y for x, y in zip(r, s)])

    for e in pool:
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
