"""Generation of the Regev bases a_i = p_i^2.

Contains the fix for the contamination bug: the previous implementation
recorded a "lucky factor" whenever a candidate prime divided N, which meant
the pipeline factored N by trial division while *choosing bases*, before the
quantum circuit ever ran. Every instance with a small prime factor
(N = 15, 21, 57, ...) was reported as a success on that basis alone.

Here a prime dividing N is treated as a SETUP FAILURE for the experiment and
surfaced separately so the instance can be excluded from results.
"""

from math import gcd

__all__ = ["is_prime_basic", "generate_regev_bases", "is_contaminated"]


def is_prime_basic(num: int) -> bool:
    """Trial-division primality test. For small simulation inputs only."""
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
