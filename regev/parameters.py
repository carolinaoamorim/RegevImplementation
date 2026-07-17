"""Input validation and Regev parameter/register sizing.

Pure classical: no Qiskit import, so this stays importable and testable
without a simulator.
"""

import math

__all__ = ["validate_factor_input", "regev_parameters"]


def _is_prime(num: int) -> bool:
    from regev.bases import is_prime_basic
    return is_prime_basic(num)


def validate_factor_input(N: int) -> dict:
    """Check whether N is a useful input for the factoring simulation."""
    if N <= 1:
        return {"valid": False, "reason": "N must be greater than 1.",
                "factor": None, "other": None}
    if N % 2 == 0:
        return {"valid": False, "reason": "N is even, so factoring is trivial.",
                "factor": 2, "other": N // 2}
    if _is_prime(N):
        return {"valid": False, "reason": "N is prime, so it has no nontrivial factors.",
                "factor": None, "other": None}
    return {"valid": True, "reason": "N is an odd composite number.",
            "factor": None, "other": None}


def regev_parameters(N: int, mode: str = "notebook") -> dict:
    """Choose register sizes for the Regev-style simulation.

    n  = bit length of N
    d  = number of dimensions (number of x-registers)
    nd = qubits per x-register
    M  = 2^nd, the Fourier modulus per dimension

    Args:
        mode: "cover_2n" sets d*nd >= 2n, matching Shor's total exponent
            qubit count -- which gives away Regev's asymptotic gate-count
            advantage and is only useful as a controlled comparison.
            "notebook" uses floor(n/d + d), closer to Regev's true parameter
            regime. Default is "notebook" for that reason.
    """
    n = N.bit_length()
    d = math.ceil(math.sqrt(n))

    if mode == "cover_2n":
        nd = math.ceil((2 * n) / d)
    elif mode == "notebook":
        nd = math.floor((n / d) + d)
    else:
        raise ValueError("mode must be either 'cover_2n' or 'notebook'.")

    return {
        "N": N,
        "n": n,
        "d": d,
        "nd": nd,
        "M": 2 ** nd,
        "total_x_qubits": d * nd,
        # x registers + y (n) + aux (n+1); the multiplier's internal flag
        # qubit is included in aux.
        "estimated_total_qubits": d * nd + n + (n + 1),
    }
