# Finite-model Regev factoring simulator

This project studies the sampling and lattice post-processing at the heart of
Oded Regev's multidimensional factoring algorithm. It computes the output
distribution of a finite, ideal Gaussian-state circuit exactly, samples from
that distribution, and compares the factoring pipeline with a uniform-random
control.

> This is a classical research simulator, not a scalable factoring tool or a
> gate-level quantum implementation. Its small-instance parameters are
> heuristic, and its results do not establish a practical quantum advantage.

The implementation follows the discrete-Gaussian state in [Regev's original
paper](https://arxiv.org/abs/2308.06572). The repository's former uniform-box
model is still available as `window="uniform"`, but is no longer called the
ideal Regev distribution.

## What is implemented

- Exact finite Gaussian-state output probabilities via one multidimensional
  FFT of a weighted relation-lattice autocorrelation.
- Faithful seeded sampling, including zero and low-probability outcomes.
- Regev-style integer lattice embedding and exact rational LLL reduction.
- Relation verification, nontrivial square-root extraction, and GCD factoring.
- A matched uniform-random control with independent random streams and Wilson
  confidence intervals.
- An explicit all-or-nothing depolarising sensitivity model.
- Guards against invalid, prime-power, classically contaminated, and malformed
  inputs.

There is deliberately no Qiskit dependency. The current experiment studies
the ideal finite distribution directly; it does not build modular-arithmetic
circuits or model coherent gate errors.

## Quick start

From a source checkout, Python 3.10 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python scripts/control_experiment.py --trials 300
```

The experiment defaults to `N = 77` and `k = d + 4` measured vectors, matching
the paper's sample-count threshold. A typical run prints both arms, Wilson 95%
intervals, verified relations per reduced basis, and the mean shortest
relation norm. Exact counts depend on the selected seed and model parameters;
the script output, rather than a copied table, is the reproducible result.

Useful variants:

```bash
# Compare several sample counts.
.venv/bin/python scripts/control_experiment.py --sweep --trials 300

# Compare finite instances. This can use hundreds of MiB of memory.
.venv/bin/python scripts/control_experiment.py --scale --trials 100

# Interpolate between ideal Gaussian samples and the uniform control.
.venv/bin/python scripts/control_experiment.py --noise --trials 100

# Inspect a larger window that covers every individual generator order.
# This is a diagnostic, not a correctness requirement.
.venv/bin/python scripts/control_experiment.py --mode axis_order --trials 100
```

The command-line runner also works directly from a fresh source checkout, so
`python3 scripts/control_experiment.py --help` does not depend on an editable
install.

## Model

For an odd composite, non-perfect-power modulus `N`, choose small primes
`b_1, ..., b_d` and squared bases `a_i = b_i²`. The relation lattice is

```text
L = { e in Z^d : product_i a_i^e_i = 1 mod N }.
```

The default finite parameters are:

```text
n = bit_length(N)
d = ceil(sqrt(n))
D = M = 2^floor(n/d + d)
R = D / (2 sqrt(d))
```

The `D` formula is a small-instance heuristic inspired by the asymptotic
`d + n/d` terms in the paper; hidden constants and the short-relation bound
are not known here. The default radius lies on the boundary of Regev's window
condition. Explicit positive radii are supported for reference-distribution
sensitivity studies. If an extremely broad radius makes `D delta` round to
zero, the integer post-processor asks for a smaller radius or an explicit
`exponent_scale` instead of silently clamping the documented value.

The simulated input amplitudes are

```text
rho_R(z) = exp(-pi ||z||² / R²),
z in {-D/2, ..., D/2 - 1}^d.
```

After modular exponentiation, discarding the output register, and applying a
QFT over each exponent register, the probability of `w` can be written as one
FFT of a relation autocorrelation. For each one-dimensional lag `e`, the
Gaussian weight is

```text
c_R(e) = sum_x rho_R(x) rho_R(x - e).
```

The legacy uniform window is the special case `rho_R(x) = 1`, where
`c_R(e) = D - |e|`. Tests compare the optimized Gaussian FFT against a direct,
per-residue-class construction on a small oracle instance.

### Lattice post-processing

Measured integers `t_j in Z_D^d` approximate noisy dual-lattice samples. The
integer embedding reduced by LLL is

```text
[ C I_d | t_1 ... t_k ]
[   0   |   D I_k      ],
```

with

```text
C = round(D delta),       delta = sqrt(d) / (sqrt(2) R).
```

This is the integer-scaled form of Regev's normalized embedding. The code
checks the short candidates exactly against `L`, turns a verified relation
into a square root of unity using the unsquared primes, and applies a GCD.
The old residual `weight=` embedding remains as an explicitly deprecated
compatibility path; it is not silently reinterpreted as the new scale.

The finite implementation is Regev-style rather than a proof-level
reproduction of every asymptotic recovery step: it does not implement the
paper's full Gram-Schmidt cutoff argument, and LLL quality at these tiny
dimensions is measured empirically against the random control.

### Why individual orders are not a validity test

`order_report` computes each `ord_N(a_i)` and reports whether `D` exceeds all
of them. This is only an axis-aligned window diagnostic. It is not a Nyquist
condition for a multidimensional hidden lattice: two generators can each have
order greater than `D` while a short cross-coordinate relation such as
`(1, -1)` remains visible.

The old `resolution_report` name is retained as a deprecated compatibility
alias. Likewise, parameter mode `resolved` is a deprecated alias for the more
honest name `axis_order`.

## Programmatic use

```python
import random

from regev import (
    generate_regev_bases,
    regev_factor,
    regev_parameters,
    sample_ideal,
    validate_factor_input,
)

N = 77
assert validate_factor_input(N)["valid"]

params = regev_parameters(N)
d, D, R = params["d"], params["D"], params["R"]
bases, primes, contamination = generate_regev_bases(N, d)
assert not contamination

samples = sample_ideal(
    bases, N, d, D, d + 4, random.Random(7), radius=R
)
factorization = regev_factor(
    samples, D, d, N, bases, primes, radius=R, verbose=False
)
print(factorization)
```

For an explicit comparison with the historical surrogate:

```python
legacy = sample_ideal(
    bases, N, d, D, d + 4, random.Random(7), window="uniform"
)
```

## Experimental guardrails

- **No cherry-picking.** Every trial resamples from the selected distribution.
- **Matched control.** Identical post-processing is applied to uniform vectors.
- **Independent arms.** The ideal and control arms use separate seeded random
  streams.
- **No setup factoring counted as success.** If base selection encounters a
  prime divisor of `N`, the instance is marked contaminated and excluded.
- **Perfect powers rejected.** They are classically recognizable; odd prime
  powers also have no nontrivial square root of unity for this factoring step.
- **Strict public inputs.** Dimension mismatches, ragged LLL matrices,
  fractional values, invalid counts, and non-invertible bases fail explicitly
  instead of hanging, truncating, or silently ignoring data.
- **No ineffective relation sums.** Products of trivial roots `±1` remain
  trivial, so pairwise sums cannot rescue failed verified relations.

## Noise model

The optional sensitivity model is

```text
p_noisy = (1 - lambda) p_ideal + lambda p_uniform.
```

It is exact for a global depolarising channel and useful as an all-or-nothing
loss-of-signal model. It is not a composition law for local gate errors, does
not model coherent error, and should not be presented as hardware fidelity.
The `lambda = 0` and `lambda = 1` samplers exactly reuse the seeded ideal and
uniform samplers.

## Performance and limits

The exact distribution has `D^d` cells. The implementation avoids Python
outcome tuples and per-residue FFTs, but the dense probability array and its
sampling CDF remain exponential in `d log D`.

At `D = 64, d = 4`, each float64 array contains 16,777,216 values and occupies
128 MiB. Probability-only operations cache just that array; sampling adds a
CDF, bringing the parameter set to about 256 MiB. Cached arrays are read-only,
and only two parameter sets are retained (so two sampled sets can approach
512 MiB). The TVD noise calculation works in bounded-size array chunks rather
than constructing multi-gigabyte tuple-key dictionaries.

Consequences:

- This simulator is appropriate for small research instances, not RSA sizes.
- It provides exact ideal finite-model probabilities, not a quantum speedup.
- Coherent/gate-level noise and reversible auxiliary-register cleanup require
  a real circuit and are outside the current scope.
- The object-integer fallback supports wide moduli exactly, but state-space
  memory normally becomes the limiting factor first.

## Repository layout

| Path | Purpose |
|---|---|
| `regev/reference.py` | Gaussian/uniform exact distributions and decoding |
| `regev/sampling.py` | Cached, seeded ideal and uniform samplers |
| `regev/postprocess.py` | Regev embedding, relation checks, and factoring |
| `regev/lattice.py` | Exact rational LLL |
| `regev/bases.py` | Base selection, number theory, and order diagnostics |
| `regev/parameters.py` | Input checks and finite parameter heuristics |
| `regev/noise.py` | Global-depolarising sensitivity model |
| `scripts/control_experiment.py` | Reproducible experiment CLI |
| `tests/` | Mathematical, regression, validation, and CLI tests |

The GitHub Actions workflow runs the suite on the minimum and newest supported
Python versions.

## Attribution and license

Created by Carolina Amorim, Owen Barnes, and Summer Malik during summer
research on quantum computing at the University of Illinois
Urbana-Champaign. Released under the [MIT License](LICENSE).
