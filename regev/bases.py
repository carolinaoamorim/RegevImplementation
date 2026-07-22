"""Generation of the Regev bases a_i = p_i^2.

Contains the fix for the contamination bug: the previous implementation
recorded a "lucky factor" whenever a candidate prime divided N, which meant
the pipeline factored N by trial division while *choosing bases*, before the
quantum circuit ever ran. Every instance with a small prime factor
(N = 15, 21, 57, ...) was reported as a success on that basis alone.

Here a prime dividing N is treated as a SETUP FAILURE for the experiment and
surfaced separately so the instance can be excluded from results.
"""

from math import gcd, isqrt
from operator import index
import warnings

__all__ = [
    "is_prime_basic",
    "generate_regev_bases",
    "is_contaminated",
    "integer_nth_root",
    "perfect_power",
    "is_prime_power",
    "multiplicative_order",
    "order_report",
    "resolution_report",
]


def _integer_arg(value, name: str) -> int:
    """Return an integer argument as a built-in int, without truncation."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None


def _positive_int_arg(value, name: str, *, minimum: int = 1) -> int:
    value = _integer_arg(value, name)
    if value < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive")
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def is_prime_basic(num: int) -> bool:
    """Trial-division primality test. For small simulation inputs only."""
    num = _integer_arg(num, "num")
    if num < 2:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    k = 3
    while k * k <= num:
        if num % k == 0:
            return False
        k += 2
    return True


def integer_nth_root(x: int, k: int) -> int:
    """Exact floor(x ** (1/k)) for x >= 0, k >= 1, with no float rounding.

    Newton's method on integers. `round(x ** (1.0 / k))` is wrong for large x
    -- double precision runs out well before these numbers do -- so the result
    is computed exactly and then corrected.
    """
    x = _integer_arg(x, "x")
    k = _integer_arg(k, "k")
    if x < 0:
        raise ValueError("x must be non-negative")
    if k < 1:
        raise ValueError("k must be at least 1")
    if x < 2 or k == 1:
        return x
    if k == 2:
        return isqrt(x)

    r = 1 << ((x.bit_length() + k - 1) // k)  # >= true root
    while True:
        nxt = ((k - 1) * r + x // r ** (k - 1)) // k
        if nxt >= r:
            break
        r = nxt
    while r ** k > x:
        r -= 1
    while (r + 1) ** k <= x:
        r += 1
    return r


def perfect_power(N: int):
    """Return (base, exponent) with base**exponent == N and exponent > 1.

    Returns the SMALLEST base (equivalently the largest exponent), so
    64 -> (2, 6) rather than (8, 2). None if N is not a perfect power.

    Only prime exponents need testing: if N = m^(ab) then N is also a perfect
    a-th power.
    """
    N = _integer_arg(N, "N")
    if N < 4:
        return None
    for k in range(2, N.bit_length() + 1):
        if not is_prime_basic(k):
            continue
        r = integer_nth_root(N, k)
        if r > 1 and r ** k == N:
            deeper = perfect_power(r)
            if deeper:
                return deeper[0], deeper[1] * k
            return r, k
    return None


def is_prime_power(N: int) -> bool:
    """True if N = p^k for a prime p and k >= 1."""
    N = _integer_arg(N, "N")
    if N < 2:
        return False
    if is_prime_basic(N):
        return True
    pp = perfect_power(N)
    return pp is not None and is_prime_basic(pp[0])


def generate_regev_bases(N: int, d: int):
    """Generate d squared prime bases coprime to N.

    Regev uses squares deliberately: a relation prod a_i^{e_i} = 1 mod N with
    a_i = p_i^2 yields u = prod p_i^{e_i} satisfying u^2 = 1 mod N, which is
    exactly the difference-of-squares structure the factoring step needs.

    Args:
        N: modulus to factor.
        d: number of dimensions / bases required.

    Returns:
        (bases, primes, trivially_found) where
        bases  = [p_1^2, ..., p_d^2], each coprime to N
        primes = [p_1, ..., p_d], needed for the square-root step
        trivially_found = primes dividing N that were skipped.

        A non-empty trivially_found means the instance is CONTAMINATED:
        trial division already revealed a factor of N. Report it; do not
        count such a run as a factoring success.
    """
    N = _positive_int_arg(N, "N", minimum=2)
    d = _positive_int_arg(d, "d")

    bases, primes, trivial = [], [], []
    candidate = 2
    while len(bases) < d:
        if is_prime_basic(candidate):
            if N % candidate == 0:
                trivial.append(candidate)  # setup failure, NOT a result
            else:
                bases.append(candidate ** 2)
                primes.append(candidate)
        candidate += 1
    return bases, primes, trivial


def is_contaminated(N: int, d: int) -> bool:
    """True if any of the first d primes divides N.

    Useful for selecting valid test instances. N = 15, 21, 57 are all
    contaminated; N = 77 = 7 x 11 is the smallest clean instance with d = 3.
    """
    _, _, trivial = generate_regev_bases(N, d)
    return len(trivial) > 0


def multiplicative_order(a: int, N: int) -> int:
    """Smallest k >= 1 with a^k = 1 (mod N). Requires gcd(a, N) = 1.

    A simulation-only diagnostic: on real hardware the order is part of what
    the quantum step is there to hide. Individual generator orders are not a
    correctness criterion for Regev's multidimensional sampling procedure.
    """
    a = _integer_arg(a, "a")
    N = _positive_int_arg(N, "N", minimum=2)
    if gcd(a % N, N) != 1:
        raise ValueError(f"a = {a} is not invertible mod N = {N}")
    k, x = 1, a % N
    while x != 1:
        x = x * a % N
        k += 1
    return k


def order_report(bases, N: int, M: int) -> dict:
    """Report whether M exceeds each individual base order.

    This is a finite-box sizing diagnostic, not a correctness condition for
    Regev's multidimensional algorithm. Short cross-coordinate relations can
    exist even when an individual generator order exceeds M. The random control
    remains the meaningful test of whether post-processing uses the samples.

    Args:
        bases: [a_1, ..., a_d], each coprime to N.
        N: modulus.
        M: Fourier modulus per dimension (2^nd).

    Returns:
        dict with orders, max_order, ratio = M / max_order, and
        `covers_axis_orders`.
    """
    N = _positive_int_arg(N, "N", minimum=2)
    M = _positive_int_arg(M, "M")
    try:
        bases = list(bases)
    except TypeError:
        raise TypeError("bases must be an iterable of integers") from None
    if not bases:
        raise ValueError("bases must contain at least one value")

    orders = [multiplicative_order(a, N) for a in bases]
    max_order = max(orders)
    covers_axis_orders = M > max_order
    if covers_axis_orders:
        reason = f"M = {M} exceeds every individual base order (max {max_order})."
    else:
        reason = (
            f"M = {M} does not exceed the largest base order ({max_order}): "
            "the individual-axis sizing diagnostic is not covered. This alone "
            "does not determine multidimensional correctness."
        )
    return {
        "orders": orders,
        "max_order": max_order,
        "M": M,
        "ratio": M / max_order,
        "covers_axis_orders": covers_axis_orders,
        "reason": reason,
    }


def resolution_report(bases, N: int, M: int) -> dict:
    """Deprecated alias for :func:`order_report`.

    The old name implied a one-dimensional correctness criterion that does not
    apply to Regev's multidimensional relation lattice.
    """
    warnings.warn(
        "resolution_report() is deprecated; use order_report()",
        DeprecationWarning,
        stacklevel=2,
    )
    report = order_report(bases, N, M)
    # Compatibility for callers of the pre-0.2 API. New code should use the
    # explicitly named `covers_axis_orders` field.
    return {**report, "resolved": report["covers_axis_orders"]}
