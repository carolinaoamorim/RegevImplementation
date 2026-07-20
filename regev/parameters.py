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
    """Check whether N is a useful input for the factoring simulation.

    Rejects perfect powers as well as the obvious cases. Both reasons matter:

    1. A perfect power N = m^k is factored by integer root extraction in
       microseconds, so reporting it as a quantum factoring success would be
       the same error as the contamination bug -- a classical step doing the
       work the quantum sampling was supposed to do.

    2. For an odd PRIME power p^k the quantum path cannot work at all. The
       congruence u^2 = 1 (mod p^k) has only the solutions u = +-1, so the
       difference-of-squares step in `relation_to_congruence` can never find a
       nontrivial square root, no matter how many samples are drawn. Before
       this check, N = 49 and N = 121 were accepted as valid, produced clean
       (uncontaminated) bases, and then failed 0/50 while the pipeline
       reported "no factors recovered from these samples" -- wording that
       implies more samples would help. They would not.

    Returns a dict with `valid`, `reason`, and `factor`/`other` populated when
    the rejection itself reveals a factorisation.
    """
    from regev.bases import perfect_power, is_prime_basic

    if N <= 1:
        return {"valid": False, "reason": "N must be greater than 1.",
                "factor": None, "other": None}
    if N % 2 == 0:
        return {"valid": False, "reason": "N is even, so factoring is trivial.",
                "factor": 2, "other": N // 2}
    if _is_prime(N):
        return {"valid": False, "reason": "N is prime, so it has no nontrivial factors.",
                "factor": None, "other": None}

    pp = perfect_power(N)
    if pp is not None:
        base, exp = pp
        if is_prime_basic(base):
            reason = (
                f"N = {base}^{exp} is an odd prime power. u^2 = 1 (mod N) has "
                "only the trivial roots u = +-1, so Regev's difference-of-"
                "squares step can never split it. Excluded as impossible, "
                "not merely unlucky."
            )
        else:
            reason = (
                f"N = {base}^{exp} is a perfect power, factored classically by "
                "integer root extraction. Excluded so a classical shortcut is "
                "not reported as a quantum result."
            )
        return {"valid": False, "reason": reason,
                "factor": base, "other": N // base}

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
