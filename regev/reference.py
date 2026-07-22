"""Classical ground truth for the Regev sampling distribution.

Computes the exact output distribution of the ideal (noiseless) Regev circuit,
so simulated counts can be validated against it via total variation distance.

The distribution is obtained from a single FFT of the relation-lattice
autocorrelation rather than one FFT per residue class -- see
`ideal_regev_distribution_array` for the derivation.  The default model uses
Regev's discrete-Gaussian input amplitudes.  The old uniform rectangular
window remains available explicitly as ``window="uniform"`` for comparisons.
"""

from collections.abc import Mapping
from itertools import product
import math
from numbers import Real
import operator

__all__ = [
    "ideal_regev_distribution",
    "ideal_regev_distribution_array",
    "total_variation_distance",
    "counts_to_distribution",
    "bitstring_to_regev_vector",
]

# Residue products reach N^2; int64 holds them while N^2 < 2^63. At or below
# this N the fast int64 path is used, above it an exact object-dtype fallback.
_MAX_MODULUS = 3_037_000_499  # floor(sqrt(2^63 - 1))


def _as_int(name: str, value, minimum=None) -> int:
    """Return an integer-like value as ``int``, rejecting bools and fractions."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    value = int(value)
    if minimum is not None and value < minimum:
        if minimum == 0:
            qualifier = "a non-negative integer"
        elif minimum == 1:
            qualifier = "positive"
        else:
            qualifier = f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _validate_distribution_inputs(bases, N, d, D):
    """Validate and canonicalise the inputs shared by reference and sampling."""
    N = _as_int("N", N, 2)
    d = _as_int("d", d, 1)
    D = _as_int("M", D, 1)

    try:
        bases = tuple(bases)
    except TypeError as exc:
        raise TypeError("bases must be an iterable of integers") from exc
    if len(bases) != d:
        raise ValueError(f"len(bases) must equal d = {d}, got {len(bases)}")

    canonical_bases = []
    for i, base in enumerate(bases):
        base = _as_int(f"bases[{i}]", base)
        if math.gcd(base, N) != 1:
            raise ValueError(f"bases[{i}] = {base} is not invertible modulo N = {N}")
        canonical_bases.append(base)
    return tuple(canonical_bases), N, d, D


def _resolve_window(d: int, D: int, window: str, radius):
    """Validate a window choice and return a canonical ``(window, radius)``.

    Regev's paper chooses ``D/(4*sqrt(d)) < R <= D/(2*sqrt(d))``.  The default
    is the upper boundary.  Explicit finite positive radii outside the paper's
    interval remain valid exact finite-window experiments and are accepted for
    sensitivity analysis.
    """
    if d < 1:
        raise ValueError("d must be positive")
    if D < 1:
        raise ValueError("D must be positive")
    if window not in ("gaussian", "uniform"):
        raise ValueError("window must be 'gaussian' or 'uniform'")

    if window == "uniform":
        if radius is not None:
            raise ValueError("radius is only defined for window='gaussian'")
        return window, None

    default_radius = D / (2.0 * math.sqrt(d))
    if radius is None:
        radius = default_radius
    elif isinstance(radius, bool) or not isinstance(radius, Real):
        raise TypeError("radius must be a real number")
    else:
        radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    return window, radius


def _lag_autocorrelation(d: int, D: int, window: str, radius):
    """Return one-dimensional lag weights and the d-dimensional state norm."""
    import numpy as np

    if window == "uniform":
        amplitudes = np.ones(D, dtype=np.float64)
    else:
        # Regev's register represents {-D/2, ..., D/2-1}.  For odd D this
        # natural integer analogue still has exactly D entries; production
        # parameters use powers of two.
        z = np.arange(D, dtype=np.float64) - (D // 2)
        amplitudes = np.exp(-np.pi * (z / radius) ** 2)

    # At lag e, this is sum_x g(x)g(x-e), over x for which both terms lie in
    # the finite register window.  For a uniform window it reduces exactly to
    # D-|e|, the triangular weights used by the legacy model.
    lag_weights = np.correlate(amplitudes, amplitudes, mode="full")
    state_norm = float(np.dot(amplitudes, amplitudes)) ** d
    return lag_weights, state_norm


def _relation_autocorrelation(
    bases, N: int, d: int, D: int, *, window: str, radius
):
    """Weighted relation-lattice autocorrelation, folded into Z_D^d.

    Returns ``(A, Z)`` where ``Z = sum_z |g(z)|^2`` is the normalization of
    the d-dimensional input state and

        A[e mod D] = sum over e in L, |e_i| < D, of prod_i c(e_i),

    where L = { e in Z^d : prod a_i^{e_i} = 1 (mod N) } is the relation
    lattice and ``c`` is the one-dimensional lag autocorrelation of the input
    amplitudes.  For the legacy uniform window, ``c(e) = D - |e|``.
    """
    import numpy as np

    span = 2 * D - 1
    off = D - 1

    # Residue products reach N^2. int64 holds that while N^2 < 2^63; beyond
    # that, fall back to object arrays of Python ints, which are exact at any
    # width (slower, but M^d binds long before N does).
    res_dtype = np.int64 if N <= _MAX_MODULUS else object

    # tables[i][j] = a_i^(j - off) mod N, covering exponents -(M-1) .. M-1.
    tables = []
    for a in bases:
        inv_a = pow(a, -1, N)
        t = np.empty(span, dtype=res_dtype)
        t[off] = 1
        v = 1
        for e in range(1, D):
            v = v * a % N
            t[off + e] = v
        v = 1
        for e in range(1, D):
            v = v * inv_a % N
            t[off - e] = v
        tables.append(t)

    weights, state_norm = _lag_autocorrelation(d, D, window, radius)
    folded = (np.arange(span) - off) % D

    # Residues and weights over dimensions 1..d-1, held as flat arrays. The
    # first dimension is iterated instead of materialised, which bounds peak
    # memory at span^(d-1) rather than span^d.
    if d == 1:
        rest_res = np.ones(1, dtype=res_dtype)
        rest_w = np.ones(1, dtype=np.float64)
        rest_idx = ()
    else:
        shape_rest = (span,) * (d - 1)
        rest_res = np.ones(shape_rest, dtype=res_dtype)
        rest_w = np.ones(shape_rest, dtype=np.float64)
        for i in range(1, d):
            sh = [1] * (d - 1)
            sh[i - 1] = span
            rest_res = rest_res * tables[i].reshape(sh) % N
            rest_w = rest_w * weights.reshape(sh)
        grids = np.meshgrid(*[folded] * (d - 1), indexing="ij")
        rest_idx = tuple(g.ravel() for g in grids)
        rest_res = rest_res.ravel()
        rest_w = rest_w.ravel()

    A = np.zeros((D,) * d, dtype=np.float64)
    for j0 in range(span):
        hits = (rest_res * int(tables[0][j0]) % N) == 1
        if not hits.any():
            continue
        idx = (np.full(int(hits.sum()), folded[j0]),) + tuple(g[hits] for g in rest_idx)
        np.add.at(A, idx, weights[j0] * rest_w[hits])
    return A, state_norm


def ideal_regev_distribution_array(
    bases, N: int, d: int, M: int, *, window: str = "gaussian", radius=None
):
    """Exact output distribution of the ideal Regev circuit, as an array.

    Here ``M`` is Regev's Fourier modulus ``D``.  The default input state is

        |psi> = (1/sqrt(Z)) sum_z rho_R(z) |z> |prod a_i^z_i mod N>,

    over ``z in {-D/2, ..., D/2-1}^d``, with
    ``rho_R(z) = exp(-pi ||z||^2/R^2)``.  If omitted, ``radius`` defaults to
    ``D/(2*sqrt(d))``, the boundary of Regev's requirement
    ``D >= 2*sqrt(d)*R``.  Regev's paper uses radii in
    ``D/(4*sqrt(d)) < R <= D/(2*sqrt(d))``; any finite positive explicit
    radius is accepted so callers can study sensitivity outside that regime.
    Pass ``window="uniform"`` to recover the legacy state with equal amplitudes
    over the register window; ``radius`` must then be omitted.

    Applying an inverse QFT_D on each of the d exponent registers gives

        p(w) = (1 / (Z D^d)) * sum_y
               |sum_{z : f(z) = y} rho_R(z) omega^{-<w,z>}|^2

    with ``f(z) = prod a_i^z_i mod N`` and ``omega = exp(2 pi i / D)``.

    Expanding the square turns the sum over residue classes into a sum over
    PAIRS, and f(x) = f(x') holds exactly when e = x - x' is a relation:

        sum_y |...|^2 = sum_{e in L, |e_i| < D}
                        prod_i c_R(e_i) omega^{-<w,e>},

    where ``c_R`` is the lag autocorrelation of the Gaussian amplitudes.  Thus
    p is one FFT of the relation-lattice autocorrelation. The naive route
    -- group the M^d exponent vectors by image, FFT each group -- computes the
    same numbers with one transform per residue class, and the class count
    grows with the order of the subgroup generated by the bases. Here the cost
    is a single transform regardless.

    Returns:
        numpy array of shape (M,)*d, indexed by w, summing to 1.
    """
    import numpy as np

    bases, N, d, M = _validate_distribution_inputs(bases, N, d, M)
    window, radius = _resolve_window(d, M, window, radius)
    A, state_norm = _relation_autocorrelation(
        bases, N, d, M, window=window, radius=radius
    )
    p = np.fft.fftn(A).real / (state_norm * float(M) ** d)
    # The lattice is symmetric (e in L iff -e in L) with symmetric weights, so
    # the transform is real up to rounding; negatives are float noise only.
    np.clip(p, 0.0, None, out=p)
    return p


def ideal_regev_distribution(
    bases, N: int, d: int, M: int, *, window: str = "gaussian", radius=None
) -> dict:
    """Exact output distribution as a dict mapping w -> probability.

    Thin wrapper over `ideal_regev_distribution_array`. Materialising all M^d
    tuple keys costs far more memory than the array itself, so prefer the
    array form for anything large; this exists for readability and for
    `total_variation_distance`.
    """
    p = ideal_regev_distribution_array(
        bases, N, d, M, window=window, radius=radius
    )
    support = product(*(range(size) for size in p.shape))
    return {w: float(p[w]) for w in support}


def bitstring_to_regev_vector(bitstring: str, d: int, nd: int) -> tuple:
    """Decode one Qiskit measurement bitstring into a vector w in Z_M^d.

    Qiskit prints classical bits little-endian: the leftmost character is the
    highest-index clbit. Register 0 therefore occupies the RIGHTMOST nd
    characters, so the chunks are reversed after splitting.

    Spaces (which Qiskit inserts between separate classical registers) are
    stripped, so both "01010 11100" and "0101011100" decode identically.

    Defined here rather than in a simulator module so that `reference` stays
    importable with no Qiskit present.
    """
    d = _as_int("d", d, 1)
    nd = _as_int("nd", nd, 1)
    if not isinstance(bitstring, str):
        raise TypeError("bitstring must be a string")
    bits = bitstring.replace(" ", "")
    if any(bit not in "01" for bit in bits):
        raise ValueError("bitstring must contain only binary digits and spaces")
    if len(bits) != d * nd:
        raise ValueError(
            f"bitstring of length {len(bits)} does not match d*nd = {d * nd}"
        )
    chunks = [bits[i * nd:(i + 1) * nd] for i in range(d)]
    return tuple(int(c, 2) for c in reversed(chunks))


def counts_to_distribution(counts, d: int, nd: int) -> dict:
    """Convert Aer counts into a w -> probability dict."""
    d = _as_int("d", d, 1)
    nd = _as_int("nd", nd, 1)
    if not isinstance(counts, Mapping):
        raise TypeError("counts must be a mapping from bitstrings to counts")
    if not counts:
        raise ValueError("counts must be non-empty")

    checked = []
    total = 0
    for bitstring, count in counts.items():
        try:
            count = _as_int(f"count for {bitstring!r}", count, 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("count values must be non-negative integers") from exc
        checked.append((bitstring, count))
        total += count
    if total <= 0:
        raise ValueError("counts must have a positive total")

    dist = {}
    for bitstring, c in checked:
        w = bitstring_to_regev_vector(bitstring, d, nd)
        dist[w] = dist.get(w, 0.0) + c / total
    return dist


def total_variation_distance(p: dict, q: dict) -> float:
    """TVD between two distributions given as dicts over the same support."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
