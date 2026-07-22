"""Input validation and Regev parameter/register sizing.

Pure classical: no Qiskit import, so this stays importable and testable
without a simulator.
"""

import math
from operator import index
import warnings

__all__ = ["validate_factor_input", "regev_parameters"]


def _integer(value, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None


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

    try:
        N = _integer(N, "N")
    except TypeError:
        return {
            "valid": False,
            "reason": "N must be an integer greater than 1.",
            "factor": None,
            "other": None,
        }

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


def regev_parameters(N: int, mode: str = "regev") -> dict:
    """Choose register sizes for the Regev-style simulation.

    n  = bit length of N
    d  = number of dimensions (number of x-registers)
    nd = qubits per x-register
    M  = 2^nd, the Fourier modulus per dimension

    Args:
        mode: "cover_2n" sets d*nd >= 2n as a larger comparison budget.
            "regev" uses floor(n/d + d), a finite-size heuristic inspired by
            the d + n/d terms in Regev's asymptotic analysis. The paper hides
            constants and a bound on the sought relation, so this mode must
            not be read as a theorem-derived parameter set. ("notebook" is a
            deprecated alias retained for compatibility.)
            "axis_order" enlarges nd until M exceeds the orders of the chosen
            generators. This can be a useful finite-box diagnostic, but it is
            NOT a correctness requirement of Regev's multidimensional
            algorithm: short cross-coordinate relations can exist even when
            every individual order exceeds M. "resolved" is retained as a
            deprecated alias for this mode.

    The returned Gaussian radius is R = M/(2*sqrt(d)), the largest value that
    satisfies the paper's finite-window condition M >= 2*sqrt(d)*R. These tiny
    instances remain demonstrations of the finite model, not evidence for the
    paper's asymptotic guarantees.
    """
    N = _integer(N, "N")
    if N <= 1:
        raise ValueError("N must be an integer greater than 1")

    if mode == "notebook":
        warnings.warn(
            "mode='notebook' is deprecated; use mode='regev'",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = "regev"
    elif mode == "resolved":
        warnings.warn(
            "mode='resolved' was based on an invalid one-dimensional resolution "
            "criterion; use mode='axis_order' only as a sizing diagnostic",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = "axis_order"

    n = N.bit_length()
    d = math.ceil(math.sqrt(n))

    if mode == "cover_2n":
        nd = math.ceil((2 * n) / d)
    elif mode == "regev":
        nd = math.floor((n / d) + d)
    elif mode == "axis_order":
        from regev.bases import generate_regev_bases, multiplicative_order

        nd = math.floor((n / d) + d)
        bases, _, trivial = generate_regev_bases(N, d)
        if not trivial:
            max_order = max(multiplicative_order(a, N) for a in bases)
            # smallest nd with 2^nd > max_order, never shrinking below notebook
            while (2 ** nd) <= max_order:
                nd += 1
    else:
        raise ValueError(
            "mode must be 'regev', 'cover_2n', or 'axis_order' "
            "(deprecated aliases: 'notebook', 'resolved')."
        )

    M = 2 ** nd

    params = {
        "N": N,
        "n": n,
        "d": d,
        "nd": nd,
        "M": M,
        "D": M,
        "R": M / (2 * math.sqrt(d)),
        "total_x_qubits": d * nd,
        "mode": mode,
    }

    # Individual generator orders are diagnostic only. They are not a
    # multidimensional Nyquist/correctness condition.
    from regev.bases import generate_regev_bases, order_report

    bases, _, trivial = generate_regev_bases(N, d)
    if trivial:
        params["axis_orders_covered"] = None
    else:
        report = order_report(bases, N, params["M"])
        params["axis_orders_covered"] = report["covers_axis_orders"]
    return params
