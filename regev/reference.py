"""Classical ground truth for the Regev sampling distribution.

Computes the exact output distribution of the ideal (noiseless) Regev circuit
by brute force, so simulated counts can be validated against it via total
variation distance -- the same metric used in the noise analysis.

Only tractable for small M^d (e.g. N = 77 gives M^d = 32^3 = 32768).
"""

from itertools import product

__all__ = ["ideal_regev_distribution", "total_variation_distance", "counts_to_distribution"]


def ideal_regev_distribution(bases, N: int, d: int, M: int) -> dict:
    """Exact output distribution of the ideal Regev circuit.

    The state before measurement is

        |psi> = (1/sqrt(M^d)) sum_x |x> |prod a_i^{x_i} mod N>

    followed by an inverse QFT_M on each of the d x-registers. Grouping the x
    values by their modular-exponentiation image and FFT-ing each group gives
    the exact marginal on the x registers.

    Args:
        bases: [a_1, ..., a_d].
        N: modulus.
        d: number of dimensions.
        M: 2^nd, Fourier modulus per dimension.

    Returns:
        dict mapping w (tuple of length d) -> probability.
    """
    import numpy as np

    groups = {}
    for x in product(range(M), repeat=d):
        v = 1
        for ai, xi in zip(bases, x):
            v = v * pow(ai, xi, N) % N
        groups.setdefault(v, []).append(x)

    probs = {}
    total = M ** d
    for _, xs in groups.items():
        arr = np.zeros((M,) * d, dtype=complex)
        for x in xs:
            arr[x] = 1.0
        f = np.fft.fftn(arr, norm="ortho")
        p = np.abs(f) ** 2 / total
        for w in product(range(M), repeat=d):
            probs[w] = probs.get(w, 0.0) + float(p[w])
    return probs


def counts_to_distribution(counts, d: int, nd: int) -> dict:
    """Convert Aer counts into a w -> probability dict."""
    from regev.simulate import bitstring_to_regev_vector

    total = sum(counts.values())
    dist = {}
    for bitstring, c in counts.items():
        w = bitstring_to_regev_vector(bitstring, d, nd)
        dist[w] = dist.get(w, 0.0) + c / total
    return dist


def total_variation_distance(p: dict, q: dict) -> float:
    """TVD between two distributions given as dicts over the same support."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
