"""Sampling from the ideal (noiseless) Regev output distribution.

This module exists to close a methodological gap. Earlier versions of the
experiment fed post-processing a hard-coded list of the five
highest-probability vectors of the old rectangular-window distribution. Those
vectors were genuine for that model, but handing the pipeline the best case
every time is not a measurement of the algorithm. It reports the performance
of an oracle, and it cannot produce error bars, because the "quantum" arm has
no randomness in it at all.

A real quantum computer samples w with probability p(w). So does this module,
including the uninformative zero vector and the low-probability tail.  Regev's
discrete-Gaussian input window is the default; pass ``window="uniform"`` to
reproduce the legacy rectangular model explicitly.
"""

from functools import lru_cache

from regev.reference import (
    _as_int,
    _resolve_window,
    _validate_distribution_inputs,
    ideal_regev_distribution_array,
)

__all__ = ["ideal_probability_array", "sample_ideal", "sample_uniform"]


def _validate_rng(rng, *, needs_randrange: bool = False):
    if not callable(getattr(rng, "random", None)):
        raise TypeError("rng must provide a callable random() method")
    if needs_randrange and not callable(getattr(rng, "randrange", None)):
        raise TypeError("rng must provide a callable randrange() method")


@lru_cache(maxsize=2)
def _cached_probability(
    bases: tuple, N: int, d: int, M: int, window: str, radius
):
    """Read-only exact probability array, computed once per parameter set."""
    p = ideal_regev_distribution_array(
        list(bases), N, d, M, window=window, radius=radius
    )
    p.setflags(write=False)
    return p


@lru_cache(maxsize=2)
def _cached_cdf(
    bases: tuple, N: int, d: int, M: int, window: str, radius
):
    """Read-only flat CDF for inverse-transform sampling.

    Only the flat cumulative array is needed to sample, so the M^d outcome
    tuples are never materialised. That matters: at N = 2021 the support is
    64^4 = 16.7M points, and a list of 16.7M Python tuples costs gigabytes,
    while the CDF is a 134 MB float64 array.
    """
    import numpy as np

    p = _cached_probability(bases, N, d, M, window, radius)
    cdf = np.cumsum(p.ravel())
    cdf /= cdf[-1]
    cdf.setflags(write=False)
    return cdf


def ideal_probability_array(
    bases, N: int, d: int, M: int, *, window: str = "gaussian", radius=None
):
    """Exact read-only probability array of shape ``(M,)*d``, memoised.

    ``window`` and ``radius`` have the same meaning and validation as in
    :func:`regev.reference.ideal_regev_distribution_array`.
    """
    bases, N, d, M = _validate_distribution_inputs(bases, N, d, M)
    window, radius = _resolve_window(d, M, window, radius)
    return _cached_probability(bases, N, d, M, window, radius)


def sample_ideal(
    bases,
    N: int,
    d: int,
    M: int,
    k: int,
    rng,
    *,
    window: str = "gaussian",
    radius=None,
):
    """Draw k independent samples from the exact ideal Regev distribution.

    This is what a noiseless quantum computer running the circuit would
    return, including the uninformative w = 0 outcome and the low-probability
    tail. No filtering, no reranking.

    Inverse-transform sampling on the cached CDF. Draws are taken one uniform
    at a time from `rng` and located with a binary search, which reproduces
    `random.choices(population, weights=...)` exactly -- the outcomes are in
    lexicographic order, which is also C order for `unravel_index`, so the
    selected distribution is sampled reproducibly for a fixed seed.

    Args:
        bases: [a_1, ..., a_d].
        N, d, M: as elsewhere.
        k: number of measured vectors to draw.
        rng: a `random.Random` instance; pass a seeded one for reproducibility.
        window: ``"gaussian"`` (default) or legacy ``"uniform"``.
        radius: Gaussian radius R. Defaults to ``M/(2*sqrt(d))`` and must be
            finite and positive. Regev's paper uses
            ``M/(4*sqrt(d)) < R <= M/(2*sqrt(d))``, but values outside that
            interval are accepted for finite-model sensitivity studies. Omit
            it for the uniform window.

    Returns:
        List of k tuples of length d.
    """
    import numpy as np

    bases, N, d, M = _validate_distribution_inputs(bases, N, d, M)
    k = _as_int("k", k, 0)
    _validate_rng(rng)
    window, radius = _resolve_window(d, M, window, radius)
    if k == 0:
        return []
    cdf = _cached_cdf(bases, N, d, M, window, radius)
    hi = cdf.size - 1
    us = np.fromiter((rng.random() for _ in range(k)), dtype=np.float64, count=k)
    flat = np.minimum(np.searchsorted(cdf, us, side="right"), hi)
    return [tuple(int(c) for c in np.unravel_index(i, (M,) * d)) for i in flat]


def sample_uniform(d: int, M: int, k: int, rng):
    """Draw k vectors uniformly from Z_M^d -- the null-hypothesis control.

    If post-processing factors N as readily from these as from `sample_ideal`,
    the pipeline is not using the quantum samples for anything.
    """
    d = _as_int("d", d, 1)
    M = _as_int("M", M, 1)
    k = _as_int("k", k, 0)
    _validate_rng(rng, needs_randrange=True)
    return [tuple(rng.randrange(M) for _ in range(d)) for _ in range(k)]
