# Regev's Multidimensional Factoring Algorithm

Simulation and analysis of Regev's 2023 multidimensional factoring algorithm,
with comparison against Shor's algorithm and noise analysis.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Layout

```
regev/
  lattice.py       Exact LLL over Fraction. No Qiskit, no numpy, no fpylll.
  bases.py         Base generation a_i = p_i^2 (+ contamination detection).
  parameters.py    Input validation and register sizing (n, d, nd, M).
  postprocess.py   Relation lattice -> LLL -> square root -> gcd.
  reference.py     Exact ideal distribution + TVD, for validating the circuit.
  circuit.py       Qiskit circuit construction.
  simulate.py      Aer execution and bitstring decoding.
  arithmetic.py    Modular exponentiation gates (vendored).
notebooks/         Thin experiment drivers; import from `regev`, hold no logic.
tests/             pytest suite.
```

`lattice.py` and `postprocess.py` deliberately import nothing from Qiskit, so
the classical half of the algorithm is testable with no simulator present.

## The contamination bug

The earlier notebook implementation reported factoring N = 15, 21, and 57
successfully. Those results were **trial division, not quantum computation**.

`generate_regev_bases` tested `gcd(candidate, N)` for each candidate prime
while selecting bases. For N = 57 it reached 3, found `gcd(3, 57) = 3`, and
recorded `lucky_factors = [(3, 19)]`. The post-processing then checked
`lucky_factors` *first*, returning it before reading a single measured vector.

Every N whose smallest prime factor is among the first `d` primes is affected —
which is most small N, exactly the regime that is simulable.

`generate_regev_bases` now returns a third value, `trivially_found`. A
non-empty value means the instance is contaminated and must be excluded from
results rather than reported as a success. `tests/test_bases.py` locks this in.

**N = 77 = 7 x 11** is the smallest clean instance at d = 3: bases 4, 9, 25 are
all coprime to 77.

## Post-processing: Regev, not Shor

Regev's classical stage contains no order-finding and no continued fractions.
Each measured vector `w` in `Z_M^d` is approximately orthogonal to every
exponent vector `e` with `prod a_i^{e_i} = 1 (mod N)`. Those relations are the
short vectors of the kernel lattice

```
L = { e in Z^d : <w_j, e> ~= 0 (mod M) for all measured w_j }
```

A single short vector combines all `d` dimensions at once — that is what makes
the algorithm Regev's rather than `d` parallel runs of Shor.

Two implementation notes:

1. Orthogonality is only **approximate**. `M = 2^nd` and the multiplicative
   orders generally do not divide `M`, so demanding exact congruence returns
   only the trivial lattice `M*Z^d`. `regev_relation_lattice` uses a weighted
   embedding so LLL finds vectors short in both the exponent block and the
   residual block.

2. The **squared bases** are what make factoring work. With `a_i = p_i^2`, a
   relation gives `u = prod p_i^{e_i}` with `u^2 = 1 (mod N)`; a nontrivial
   square root of unity splits N by gcd.

Worked path for N = 77: `9^1 * 25^-2 = 1 (mod 77)` → `u = 3 * 5^-2` →
`gcd(u - 1, 77) = 11`.

## Validation: quantum vs. random control

Running the identical post-processing on uniformly random vectors is what makes
the end-to-end claim defensible. On N = 77, 30 trials each, seed 7
(reproduce with `python scripts/control_experiment.py`):

| samples          | success | relations per LLL basis | mean min \|e\|_1 |
|------------------|---------|-------------------------|------------------|
| quantum (ideal)  | 30/30   | 3.13                    | 3.0              |
| uniform random   | 9/30    | 0.80                    | 16.7             |

Every LLL basis vector derived from quantum samples is a relation. Random
vectors need long vectors and mostly fail. Random succeeds occasionally because
at N = 77 a small lattice sometimes contains a relation by chance — that is a
limitation of the instance size, and should be stated as such.

The LLL is separately validated against its mathematical invariants
(determinant preservation under unimodular transform, size-reduction condition
`|mu_ij| <= 1/2`, non-increase of the shortest vector) in `tests/test_lattice.py`.

## Parameter modes

`regev_parameters(N, mode=...)`:

- `notebook` (default) — `nd = floor(n/d + d)`, closer to Regev's true regime.
- `cover_2n` — `nd = ceil(2n/d)`, matching Shor's total exponent qubit count.
  This gives away Regev's asymptotic gate-count advantage and is useful only as
  a controlled comparison.

## Open items

- Verify aux-register uncomputation: run the modexp on `|x>|1>|0>` with
  `Statevector` and assert aux returns to `|0>`. If it does not, the measured
  distribution is entangled with garbage.
- Validate the Aer circuit against `reference.ideal_regev_distribution` via TVD.
- Extend noise analysis from Shor's N = 15 to Regev on N = 77.
