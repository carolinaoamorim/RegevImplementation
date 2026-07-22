"""A circuit-free noise model for the Regev sampler.

The exact reference distribution is noiseless by construction, and a full Aer
simulation of the circuit is out of reach (30+ qubits at the smallest clean
instance -- see the README). But the leading-order effect of hardware noise on
a sampling experiment can be modelled directly on the output distribution,
without ever building the circuit.

The **depolarising** model: with probability `lam` a run decoheres completely
and the measured register is a uniform random vector; with probability
`1 - lam` it returns the ideal outcome. The observed distribution is therefore

    p_noisy = (1 - lam) * p_ideal + lam * uniform.

This is an explicit all-or-nothing surrogate, not a gate-level hardware model.
It is exact for a global depolarising channel and can be read as "the entire
sampling run works with probability 1 - lam". Its whole point is that it
interpolates the two arms the control experiment already contrasts:
lam = 0 is the ideal quantum sampler, lam = 1 is the uniform-random control.
So the noise sweep measures exactly how much decoherence the classical
post-processing tolerates before the quantum signal is gone.

For sensitivity studies, `depolarizing_lambda` maps an independent per-qubit
*failure* probability to the pessimistic assumption that any such failure
fully randomises the output. It must not be interpreted as composing physical
single-qubit depolarising channels.
"""

import math
from numbers import Real

from regev.reference import _as_int, _resolve_window, _validate_distribution_inputs
from regev.sampling import (
    _validate_rng,
    ideal_probability_array,
    sample_ideal,
    sample_uniform,
)

__all__ = [
    "depolarizing_lambda",
    "noisy_distribution",
    "sample_noisy",
    "noisy_tvd_from_ideal",
]


def _probability(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def depolarizing_lambda(eps: float, num_qubits: int) -> float:
    """All-or-nothing mixture rate from an independent per-qubit failure rate.

    If each of `num_qubits` qubits survives with probability `1 - eps`, the run
    has no failures with probability `(1 - eps)^q`. This helper pessimistically
    maps every other run to a uniform output. It is a sensitivity conversion,
    not a physical composition law for local depolarising channels.
    """
    eps = _probability(eps, "eps")
    num_qubits = _as_int("num_qubits", num_qubits, 0)
    return 1.0 - (1.0 - eps) ** num_qubits


def noisy_distribution(
    bases, N: int, d: int, M: int, lam: float, *, window="gaussian", radius=None
):
    """Exact depolarised distribution array: (1-lam) p_ideal + lam * uniform.

    Args:
        lam: depolarising probability in [0, 1]. 0 is noiseless, 1 is uniform.

    Returns:
        numpy array of shape (M,)*d summing to 1.
    """
    lam = _probability(lam, "lam")
    p = ideal_probability_array(bases, N, d, M, window=window, radius=radius)
    uniform = 1.0 / float(M) ** d
    return (1.0 - lam) * p + lam * uniform


def sample_noisy(
    bases,
    N: int,
    d: int,
    M: int,
    k: int,
    lam: float,
    rng,
    *,
    window="gaussian",
    radius=None,
):
    """Draw k samples from the depolarised distribution.

    Each of the k measured vectors independently either comes from the ideal
    sampler (probability 1 - lam) or is a fresh uniform vector (probability
    lam). Drawing per-vector rather than from the mixed array keeps the two
    component samplers -- and hence their seeded reproducibility -- intact.
    """
    lam = _probability(lam, "lam")
    bases, N, d, M = _validate_distribution_inputs(bases, N, d, M)
    k = _as_int("k", k, 0)
    window, radius = _resolve_window(d, M, window, radius)
    _validate_rng(rng, needs_randrange=lam != 0.0)

    # Preserve the component samplers exactly at the endpoints, including the
    # seeded random stream. Intermediate mixtures need one branch draw per
    # output; the endpoints do not.
    if lam == 0.0:
        return sample_ideal(
            bases, N, d, M, k, rng, window=window, radius=radius
        )
    if lam == 1.0:
        return sample_uniform(d, M, k, rng)

    out = []
    for _ in range(k):
        if rng.random() < lam:
            out.extend(sample_uniform(d, M, 1, rng))
        else:
            out.extend(
                sample_ideal(
                    bases, N, d, M, 1, rng, window=window, radius=radius
                )
            )
    return out


def noisy_tvd_from_ideal(
    bases, N: int, d: int, M: int, lam: float, *, window="gaussian", radius=None
) -> float:
    """TVD between the depolarised and ideal distributions.

    Uses `lam * TVD(uniform, ideal)` directly and sums in bounded-size chunks.
    The previous implementation converted both dense arrays to tuple-keyed
    dictionaries, which needed multiple gigabytes for a 64^4 distribution.
    """
    lam = _probability(lam, "lam")

    if lam == 0.0:
        bases, N, d, M = _validate_distribution_inputs(bases, N, d, M)
        _resolve_window(d, M, window, radius)
        return 0.0

    import numpy as np

    p = ideal_probability_array(
        bases, N, d, M, window=window, radius=radius
    ).ravel()
    uniform = 1.0 / float(M) ** d
    block_size = 1_000_000
    distance = 0.0
    for start in range(0, p.size, block_size):
        block = p[start:start + block_size]
        distance += float(np.abs(block - uniform).sum())
    return lam * 0.5 * distance
