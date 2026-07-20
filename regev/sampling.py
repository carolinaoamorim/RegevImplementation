"""Sampling from the ideal (noiseless) Regev output distribution.

This module exists to close a methodological gap. Earlier versions of the
experiment fed post-processing a hard-coded list of the five
highest-probability vectors of the exact distribution. Those vectors were
genuine -- they really are the top of `reference.ideal_regev_distribution` --
but handing the pipeline the best case every time is not a measurement of the
algorithm. It reports the performance of an oracle, and it cannot produce
error bars, because the "quantum" arm has no randomness in it at all.

A real quantum computer samples w with probability p(w). So does this module.
Sampling faithfully means sometimes drawing w = 0 (probability 0.067 at
N = 77), which carries no information, and sometimes drawing from the long
tail. The measured success rate is therefore lower than the cherry-picked
one -- and it is the number that can honestly be compared against a control.

As it happens the honest number is still decisive: at N = 77 with k = 5
samples, ideal sampling factors 99% of the time against 27% for uniformly
random vectors.
"""

from functools import lru_cache

from regev.reference import ideal_regev_distribution_array

__all__ = ["ideal_probability_array", "sample_ideal", "sample_uniform"]


@lru_cache(maxsize=8)
def _cached_cdf(bases: tuple, N: int, d: int, M: int):
    """(cdf, probability array) for the exact distribution, computed once.

    Only the flat cumulative array is needed to sample, so the M^d outcome
    tuples are never materialised. That matters: at N = 2021 the support is
    64^4 = 16.7M points, and a list of 16.7M Python tuples costs gigabytes,
    while the CDF is a 134 MB float64 array.
    """
    import numpy as np

    p = ideal_regev_distribution_array(list(bases), N, d, M)
    cdf = np.cumsum(p.ravel())
    cdf /= cdf[-1]
    return cdf, p


def ideal_probability_array(bases, N: int, d: int, M: int):
    """Exact probability array of shape (M,)*d, memoised."""
    return _cached_cdf(tuple(bases), N, d, M)[1]


def sample_ideal(bases, N: int, d: int, M: int, k: int, rng):
    """Draw k independent samples from the exact ideal Regev distribution.

    This is what a noiseless quantum computer running the circuit would
    return, including the uninformative w = 0 outcome and the low-probability
    tail. No filtering, no reranking.

    Inverse-transform sampling on the cached CDF. Draws are taken one uniform
    at a time from `rng` and located with a binary search, which reproduces
    `random.choices(population, weights=...)` exactly -- the outcomes are in
    lexicographic order, which is also C order for `unravel_index`, so the
    same seed yields the same samples as before this was vectorised.

    Args:
        bases: [a_1, ..., a_d].
        N, d, M: as elsewhere.
        k: number of measured vectors to draw.
        rng: a `random.Random` instance; pass a seeded one for reproducibility.

    Returns:
        List of k tuples of length d.
    """
    import numpy as np

    cdf, _ = _cached_cdf(tuple(bases), N, d, M)
    hi = cdf.size - 1
    us = np.fromiter((rng.random() for _ in range(k)), dtype=np.float64, count=k)
    flat = np.minimum(np.searchsorted(cdf, us, side="right"), hi)
    return [tuple(int(c) for c in np.unravel_index(i, (M,) * d)) for i in flat]


def sample_uniform(d: int, M: int, k: int, rng):
    """Draw k vectors uniformly from Z_M^d -- the null-hypothesis control.

    If post-processing factors N as readily from these as from `sample_ideal`,
    the pipeline is not using the quantum samples for anything.
    """
    return [tuple(rng.randrange(M) for _ in range(d)) for _ in range(k)]
