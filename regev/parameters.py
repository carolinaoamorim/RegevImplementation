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
            "resolved" enlarges nd until M exceeds every base order, so the
            near-orthogonality the post-processing relies on actually holds.
            It requires computing the orders (a simulation-only quantity) and,
            like "cover_2n", gives up Regev's register-size advantage. Use it
            when the goal is a clean demonstration rather than a faithful
            parameter regime -- see the caution below.

    Caution: the "notebook" formula is Regev-asymptotic and at these tiny n it
    routinely produces M <= max base order (e.g. N = 299, 323, 391, 437), where
    at least one register is under-resolved and the orthogonality argument is
    unsound. `regev.bases.resolution_report` diagnoses this, and "resolved"
    mode avoids it. The returned dict always carries a `resolved` flag.
    """
    n = N.bit_length()
    d = math.ceil(math.sqrt(n))

    if mode == "cover_2n":
        nd = math.ceil((2 * n) / d)
    elif mode == "notebook":
        nd = math.floor((n / d) + d)
    elif mode == "resolved":
        from regev.bases import generate_regev_bases, multiplicative_order

        nd = math.floor((n / d) + d)  # start from the notebook size
        bases, _, trivial = generate_regev_bases(N, d)
        if not trivial:
            max_order = max(multiplicative_order(a, N) for a in bases)
            # smallest nd with 2^nd > max_order, never shrinking below notebook
            while (2 ** nd) <= max_order:
                nd += 1
    else:
        raise ValueError("mode must be 'notebook', 'cover_2n', or 'resolved'.")

    params = {
        "N": N,
        "n": n,
        "d": d,
        "nd": nd,
        "M": 2 ** nd,
        "total_x_qubits": d * nd,
        # x registers + y (n) + aux (n+1); the multiplier's internal flag
        # qubit is included in aux.
        "estimated_total_qubits": d * nd + n + (n + 1),
        "mode": mode,
    }

    # Attach a resolution flag when the instance is clean enough to size one.
    from regev.bases import generate_regev_bases, resolution_report

    bases, _, trivial = generate_regev_bases(N, d)
    if trivial:
        params["resolved"] = None  # contaminated: resolution is moot
    else:
        params["resolved"] = resolution_report(bases, N, params["M"])["resolved"]
    return params
