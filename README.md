# Improvements to the Regev Factoring Simulation

What changed in this pass, why, and how each claim was verified.

Summary: one methodological bug that inflated the headline result, a ~96x
speedup of the lattice reduction, removal of dead and phantom code, and a test
suite grown from 29 to 49 tests.

---

## 1. The cherry-picking bug (most important)

The repository already documented a "contamination bug", where trial division
factored `N` during basis selection and the result was reported as quantum
computation. **The same class of error was still present, one layer up.**

The control table reported **30/30** for the quantum arm. That number came from
feeding post-processing a hard-coded list:

```python
N77_VECTORS = [(4, 2, 17), (28, 30, 15), (19, 26, 13), (13, 6, 19), (2, 17, 9)]
```

Those vectors were checked against `reference.ideal_regev_distribution` and are
genuine — they really are ranks 1–5 of the exact distribution. That is not the
problem.

The problem is that a real quantum computer **samples**. It draws `w = 0` about
6.7% of the time (the mode of the distribution, and information-free) and lands
in the low-probability tail regularly. Handing the pipeline the top 5 on every
trial measures an oracle, not the algorithm.

It also cannot produce error bars. The "30 trials" were 30 evaluations of the
same constant input; the observed variance was necessarily zero. The same flaw
was in `test_quantum_beats_random_control`, which re-ran one fixed input 20
times and called it 20 trials.

### The fix

`regev/sampling.py` (new) draws from the exact distribution faithfully —
`w = 0` and long tail included — via `sample_ideal`, with `sample_uniform` as
the null-hypothesis control. Both arms are now resampled every trial from a
seeded `random.Random`, so runs are reproducible without being constant.

### The honest result is stronger

N = 77, d = 3, M = 32, 300 trials per arm, seed 7
(`python scripts/control_experiment.py --sweep`):

| k  | arm            | success  | 95% CI (Wilson) | rel/basis | mean min \|e\|_1 |
|----|----------------|----------|-----------------|-----------|------------------|
| 3  | ideal quantum  | 280/300  | 89.9% – 95.6%   | 2.71      | 3.5              |
| 3  | uniform random | 50/300   | 12.9% – 21.3%   | 0.35      | 11.9             |
| 5  | ideal quantum  | 297/300  | 97.1% – 99.7%   | 3.33      | 3.5              |
| 5  | uniform random | 81/300   | 22.3% – 32.3%   | 0.59      | 18.1             |
| 8  | ideal quantum  | 299/300  | 98.1% – 99.9%   | 4.04      | 3.5              |
| 8  | uniform random | 81/300   | 22.3% – 32.3%   | 0.63      | 21.9             |
| 12 | ideal quantum  | 300/300  | 98.7% – 100%    | 5.57      | 3.4              |
| 12 | uniform random | 108/300  | 30.8% – 41.6%   | 0.81      | 21.2             |
| 16 | ideal quantum  | 300/300  | 98.7% – 100%    | 8.90      | 3.3              |
| 16 | uniform random | 128/300  | 37.2% – 48.3%   | 1.12      | 20.8             |

The intervals do not overlap at any `k`. Quantum samples put relations among
the **short** vectors of the reduced basis (mean min `|e|_1` ≈ 3.5, flat in
`k`); random vectors reach them only at length ~20, and usually not at all.

Losing 30/30 in exchange for 297/300 with a confidence interval and a
matched control is a net gain in defensibility.

### Supporting changes

- `scripts/control_experiment.py` rewritten: real sampling, Wilson score
  intervals (the normal approximation runs outside [0,1] at these rates),
  argparse for `-N/-k/--trials/--seed/--sweep`, and a hard exit if the chosen
  `N` is contaminated.
- `test_quantum_beats_random_control` now resamples both arms for 60 trials and
  asserts a separation rather than a fixed outcome. It also asserts the random
  control succeeds *sometimes* — a control that never fires is usually a
  control that is not running.

---

## 2. LLL reduction: ~96x faster

`regev/lattice.py` recomputed Gram–Schmidt from scratch **inside** the
size-reduction inner loop, and again after every swap.

Both updates are available in closed form (Cohen, *A Course in Computational
Algebraic Number Theory*, Alg. 2.6.3):

- **Size reduction** moves `B[k]` only within the span of `B[0..k]`, so the
  Gram–Schmidt vectors and all norms are invariant; only row `k` of `mu`
  changes.
- **Swaps** do reorder the span, but affect only column pair `(k-1, k)` of `mu`
  and the two norms.

Gram–Schmidt now runs exactly once per call. Exact `Fraction` arithmetic is
retained — there was never a reason to accept floating-point drift at these
dimensions.

| rows (d + k) | before   | after   | speedup |
|--------------|----------|---------|---------|
| 3 + 5        | 34 ms    | 1.6 ms  | 21x     |
| 3 + 8        | 208 ms   | 4.3 ms  | 48x     |
| 3 + 12       | 1292 ms  | 13.4 ms | 96x     |

Full test suite: **3.9 s → 0.58 s**. The 300-trial × 5-value sweep above went
from impractical to a ~40 s run — which is what made the honest experiment in
section 1 feasible at all.

### How it was verified

Not just "tests still pass". The new implementation was differential-tested
against the original naive one on 300 randomised bases — including rectangular
and rank-deficient inputs — and produces **bit-identical output** on every one.
A `nk == 0` guard falls back to full recomputation on linearly dependent input
rather than dividing by zero.

---

## 3. Dead code and phantom modules

`reference.counts_to_distribution` imported `regev.simulate`, **a module that
never existed**. The function had no test and would have raised `ImportError`
on any call.

Fixed by writing `bitstring_to_regev_vector` directly in `reference.py`, which
also keeps the module importable with no Qiskit present. Qiskit's little-endian
clbit ordering (register 0 is the **rightmost** chunk) is now documented and
pinned by tests, so any future circuit module has a convention to match.

The README also documented `circuit.py`, `simulate.py`, and `arithmetic.py` in
its layout section. None existed.

### Why they were not written

Building them was the obvious "max effort" move. The numbers say otherwise.

At N = 77: `n = 7`, `d = 3`, `nd = 5`, so the register budget is
`15 + 7 + 8 = 30` qubits. A dense statevector of 30 qubits is `2^30` amplitudes
≈ **17 GB**, before the modular-exponentiation circuit's depth is considered.
And N = 77 is the *smallest* clean instance — anything larger is worse.

Meanwhile `ideal_regev_distribution` computes the noiseless output distribution
**exactly** in ~0.1 s. It is not an approximation of what Aer would produce; it
is what Aer converges to given infinite shots and no noise.

So the phantom modules were removed from the docs and replaced with a section
explaining why the exact distribution is the correct tool here. A circuit is
genuinely needed only for **noise**, which the exact computation cannot model —
that is now stated as the real open item.

---

## 4. Dependencies

`pyproject.toml` declared `qiskit`, `qiskit-aer`, `matplotlib`, and
`pylatexenc` as hard dependencies. **Nothing in the package imports any of
them.**

`numpy` is now the only requirement (it backs the FFT in the exact reference
distribution), with the rest moved to `[plots]` and `[quantum]` extras.
`import regev` no longer pulls in numpy at import time either — it is imported
lazily inside the one function that needs it.

`requirements.txt` updated to match. Added a `.gitignore` for
`__pycache__/`, `*.egg-info/`, and local virtualenvs.

---

## 5. Algorithmic change tested and rejected

`regev_factor` combined relations pairwise by sum only. Pairwise **differences**
(and doubling) were implemented and benchmarked at k = 3 and k = 5, 300 trials
each: **identical success rates**, 280/300 and 297/300 in both variants.

Relations form a group and the reduced basis already spans it, so differences
add nothing. The change was **not** adopted — the code comment now records that
it was tried and why it does not help, so nobody re-derives it.

One real fix did land: the loop iterated ordered pairs, computing every sum
twice (`r + s` and `s + r`). Now `itertools.combinations`.

---

## 6. Smaller corrections

- `nearest_int` documented "ties away from zero"; it actually rounds ties
  toward `+infinity` (`nearest_int(-1/2) == 0`, which the existing test
  asserted). Docstring corrected to match behaviour and tests.
- `lll` claimed to return vectors "shortest first". LLL guarantees the first
  vector is short but does **not** sort the basis. Docstring corrected, with a
  note that callers wanting an order must sort — `postprocess` already does.
- The hard-coded `N77_VECTORS` are kept in `tests/test_postprocess.py`, but
  relabelled as a deterministic **fixture** for unit tests, explicitly not an
  experiment.
- `regev/__init__.py` now exports the reference and sampling functions.

---

## 7. Test suite: 29 → 49

New `tests/test_reference.py` covers the exact distribution, bitstring
decoding, and sampling. Tests assert independently-derived properties rather
than stored numbers:

- distribution is normalised, non-negative, and covers the full `M^d` support
- `w = 0` is the mode — a structural fact, and the specific thing the
  cherry-picked sampler was hiding
- distribution is far from uniform (TVD > 0.5) — the premise of the experiment
- empirical distribution of 20k draws converges to the exact one (TVD < 0.10)
- the sampler **does** draw `w = 0` — a direct regression test against the bug
  in section 1
- TVD is symmetric, zero on identical inputs, one on disjoint support
- bitstring decoding: register order, space handling, bounds, length mismatch

Verification commands:

```bash
pytest                                          # 49 passed in 0.58s
python scripts/control_experiment.py --sweep    # regenerates the table above
```

---

## Open items

- **Noise analysis.** Requires an actual circuit, since the exact reference
  distribution is noiseless by construction. Applying a depolarising channel to
  the ideal distribution is a cheaper first approximation that would not need
  30 qubits.
- **Larger clean instances.** At N = 77 the random control still succeeds ~27%
  of the time, because a rank-3 lattice sometimes contains a relation by
  chance. A larger `N` would sharpen the separation, at `M^d` cost in the exact
  distribution.
- **Aux-register uncomputation**, if a circuit is ever written: run the modexp
  on `|x>|1>|0>` and assert aux returns to `|0>`, or the measured distribution
  is entangled with garbage.
- The `regev_parameters(mode="notebook")` string now refers to a `notebooks/`
  directory that is empty. The API string was left alone to avoid breaking
  callers, but it is a dangling name.
