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

This is the standard first approximation and the one the README's open item
named. It is exact for a global depolarising channel and a reasonable
caricature of "the circuit works with probability 1 - lam". Its whole point is
that it interpolates the two arms the control experiment already contrasts:
lam = 0 is the ideal quantum sampler, lam = 1 is the uniform-random control.
So the noise sweep measures exactly how much decoherence the classical
post-processing tolerates before the quantum signal is gone.

`lam` relates to a per-qubit error rate `eps` over `q` measured qubits by
`lam = 1 - (1 - eps)^q`; `depolarizing_lambda` converts.
"""

import random

from regev.reference import total_variation_distance
from regev.sampling import ideal_probability_array, sample_ideal, sample_uniform

__all__ = [
    "depolarizing_lambda",
    "noisy_distribution",
    "sample_noisy",
    "noisy_tvd_from_ideal",
]


def depolarizing_lambda(eps: float, num_qubits: int) -> float:
    """Global decoherence probability from an independent per-qubit rate.

    If each of `num_qubits` qubits survives with probability `1 - eps`, the run
    is fully coherent with probability `(1 - eps)^q`, so it depolarises with
    `lam = 1 - (1 - eps)^q`.
    """
    if not 0.0 <= eps <= 1.0:
        raise ValueError("eps must be in [0, 1]")
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative")
    return 1.0 - (1.0 - eps) ** num_qubits


def noisy_distribution(bases, N: int, d: int, M: int, lam: float):
    """Exact depolarised distribution array: (1-lam) p_ideal + lam * uniform.

    Args:
        lam: depolarising probability in [0, 1]. 0 is noiseless, 1 is uniform.

    Returns:
        numpy array of shape (M,)*d summing to 1.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    p = ideal_probability_array(bases, N, d, M)
    uniform = 1.0 / float(M) ** d
    return (1.0 - lam) * p + lam * uniform


def sample_noisy(bases, N: int, d: int, M: int, k: int, lam: float, rng):
    """Draw k samples from the depolarised distribution.

    Each of the k measured vectors independently either comes from the ideal
    sampler (probability 1 - lam) or is a fresh uniform vector (probability
    lam). Drawing per-vector rather than from the mixed array keeps the two
    component samplers -- and hence their seeded reproducibility -- intact.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    out = []
    for _ in range(k):
        if rng.random() < lam:
            out.extend(sample_uniform(d, M, 1, rng))
        else:
            out.extend(sample_ideal(bases, N, d, M, 1, rng))
    return out


def noisy_tvd_from_ideal(bases, N: int, d: int, M: int, lam: float) -> float:
    """TVD between the depolarised and ideal distributions.

    Closed form is `lam * TVD(uniform, ideal)`, but it is computed here so the
    identity is checked rather than assumed.
    """
    p = ideal_probability_array(bases, N, d, M)
    ideal = {w: float(v) for w, v in _enumerate(p, M, d)}
    noisy_arr = noisy_distribution(bases, N, d, M, lam)
    noisy = {w: float(v) for w, v in _enumerate(noisy_arr, M, d)}
    return total_variation_distance(noisy, ideal)


def _enumerate(arr, M: int, d: int):
    from itertools import product
    for w in product(range(M), repeat=d):
        yield w, arr[w]
