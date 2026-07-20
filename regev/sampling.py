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

from regev.reference import ideal_regev_distribution

__all__ = ["ideal_distribution_cached", "sample_ideal", "sample_uniform"]


@lru_cache(maxsize=8)
def _cached(bases: tuple, N: int, d: int, M: int):
    """Distribution as parallel (outcomes, weights) tuples, computed once.

    The brute-force distribution costs a full FFT per residue class, so the
    control experiment would otherwise recompute it on every trial.
    """
    dist = ideal_regev_distribution(list(bases), N, d, M)
    outcomes = sorted(dist)
    return outcomes, [dist[w] for w in outcomes]


def ideal_distribution_cached(bases, N: int, d: int, M: int):
    """(outcomes, weights) for the exact ideal distribution, memoised."""
    return _cached(tuple(bases), N, d, M)


def sample_ideal(bases, N: int, d: int, M: int, k: int, rng):
    """Draw k independent samples from the exact ideal Regev distribution.

    This is what a noiseless quantum computer running the circuit would
    return, including the uninformative w = 0 outcome and the low-probability
    tail. No filtering, no reranking.

    Args:
        bases: [a_1, ..., a_d].
        N, d, M: as elsewhere.
        k: number of measured vectors to draw.
        rng: a `random.Random` instance; pass a seeded one for reproducibility.

    Returns:
        List of k tuples of length d.
    """
    outcomes, weights = ideal_distribution_cached(bases, N, d, M)
    return rng.choices(outcomes, weights=weights, k=k)


def sample_uniform(d: int, M: int, k: int, rng):
    """Draw k vectors uniformly from Z_M^d -- the null-hypothesis control.

    If post-processing factors N as readily from these as from `sample_ideal`,
    the pipeline is not using the quantum samples for anything.
    """
    return [tuple(rng.randrange(M) for _ in range(d)) for _ in range(k)]
