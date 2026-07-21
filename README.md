# Improvements to the Regev Factoring Simulation

Made by Carolina Amorim, Owen Barnes, and Summer Malik during summer research on quantum computing at the University of Illinois at Urbana Champaign.

What changed in this pass, why, and how each claim was verified.

Summary: two correctness bugs (one methodological bug that inflated the
headline result, one instance class that could never have succeeded), two large
speedups (~96x on the lattice reduction, ~1250x on the reference distribution),
removal of dead and phantom code, three tempting "improvements" measured and
rejected, a resolution guard for a fourth silent-failure mode, and a test suite
grown from 29 to 112 tests.

The speedups are not cosmetic: they are what made the honest experiment
affordable and what raised the demonstrated result from 99% vs 27% at N = 77
to **99.3% vs 1.0% at N = 2021**.

---

## 1. The cherry-picking bug (most important)

The repository documented a "contamination bug", where trial division
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
**exactly** in ~0.08 s — and after section 7, in 0.79 s even at N = 2021, an
instance whose circuit would need well over 40 qubits. It is not an
approximation of what Aer would produce; it is what Aer converges to given
infinite shots and no noise.

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

## 5. Prime powers: an instance class that could never succeed

`validate_factor_input` accepted **N = 49** and **N = 121** as valid inputs.
They are odd, composite, and uncontaminated — bases 4, 9, 25 are coprime to
both — so nothing in the pipeline screened them out.

They fail **0/50** even at k = 16, and they always will:

```text
sqrt(1) mod 49  = {1, 48}      nontrivial: none
sqrt(1) mod 121 = {1, 120}     nontrivial: none
```

For an odd prime power `p^k`, the congruence `u^2 = 1 (mod p^k)` has only the
roots `u = ±1`. Regev's entire factoring step is "find a nontrivial square root
of unity, take a gcd" — so there is nothing to find. No number of samples
helps. Worse, the pipeline reported *"No factors recovered from these
samples"*, wording that implies more samples would fix it.

### The fix

Added `integer_nth_root`, `perfect_power`, and `is_prime_power` to `bases.py`,
and wired perfect-power detection into `validate_factor_input`, which now
rejects these up front with the reason and the classically-extracted factor.

Two distinct rejections, because the reasons differ:

- **Prime power** (49, 121, 343) — the quantum path is *impossible*, not
  unlucky.
- **Composite perfect power** (225 = 15²) — Regev *could* split it, but
  `isqrt` already did, in microseconds. Reporting it as a quantum success would
  be the contamination bug again.

`integer_nth_root` uses integer Newton iteration rather than `x ** (1.0 / k)`;
a test pins the case `(2^53 + 1)^3`, where double precision gives the wrong
root.

---

## 6. Three algorithmic changes tested and rejected

Each of these **raises the headline success rate**. All three were rejected,
because each raises the *random control* by as much or more — meaning the extra
work is being done classically, not by the quantum samples.

| change | ideal | random | separation |
|--------|-------|--------|------------|
| baseline (k=5) | 99.0% | 26.2% | **72.8 pts** |
| + pairwise differences | 99.0% | 26.2% | 72.8 pts |
| + short-vector enumeration | **100.0%** | 95.0% | **5.0 pts** |
| + weight ensemble (w = 1,2,4,8,16) | 99.8% | 51.0% | 48.8 pts |

- **Pairwise differences** change nothing at all. Relations form a group and
  the reduced basis already spans it.
- **Short-vector enumeration** (all ±1 combinations of the 5 shortest basis
  vectors) hits a perfect 100% — and lifts random from 26% to 95%. It
  brute-forces the relation space and factors N = 77 with *no quantum input*.
  This is the contamination pattern for a third time, and 100% is exactly what
  makes it tempting.
- **Weight ensemble** (retry across weights, accept any success) is plain
  multiple testing: five chances instead of one, for both arms.

Success rate alone is the wrong metric. The separation is the result. These are
documented in a comment in `regev_factor` so they are not re-derived.

One real fix did land: the pool loop iterated *ordered* pairs, computing every
sum twice (`r + s` and `s + r`). Now `itertools.combinations`.

### Weight sensitivity

`regev_relation_lattice`'s docstring claimed results were "stable for 1..8".
Measured over 1000 trials on held-out seeds, they are not:

| k | w=1 | w=2 | w=4 (default) | w=8 |
|---|-----|-----|---------------|-----|
| 3 | 95.7% | 95.4% | 93.5% | 90.3% |
| 5 | 99.7% | 99.6% | 98.8% | 96.1% |
| 8 | 99.9% | 100.0% | 99.8% | 99.2% |

`w=2` beats the default at k=3 and ties at k=5 — but gives up ~8 points of
control separation at k=8, because it lifts the control too. **No value
dominates across k, so the default was left at 4** and the docstring corrected
to report the real numbers. Changing it on the k=3 result alone would have been
overfitting; the table above is from seeds held out from the tuning run.

---

## 7. The reference distribution: one FFT instead of one per residue class

This is what unblocked everything in section 8.

`ideal_regev_distribution` grouped the `M^d` exponent vectors by their
modular-exponentiation image, ran an FFT per group, and accumulated with a
Python loop over all `M^d` outcomes *inside each group*. Cost scaled with the
number of residue classes, which grows with the order of the subgroup the
bases generate. N = 299 took **13.7 s**, and it was getting worse fast.

### The identity

The probability is

```text
p(w) = (1 / M^2d) * sum_y |sum_{x : f(x) = y} w^(-<w,x>)|^2
```

Expanding the square turns the sum over classes into a sum over **pairs**
`(x, x')`. And `f(x) = f(x')` holds exactly when `e = x - x'` satisfies
`prod a_i^{e_i} = 1 (mod N)` — that is, when `e` is a **relation**. The number
of ordered pairs in the box `[0,M)^d` differing by `e` is `prod_i (M - |e_i|)`,
so

```text
sum_y |...|^2 = sum over e in L, |e_i| < M, of  prod_i (M - |e_i|) * w^(-<w,e>)
```

The distribution is therefore a *single* FFT of the relation-lattice
autocorrelation — no grouping, no per-class transform. The relation lattice is
the same object `postprocess.py` searches, which is a pleasing symmetry: the
forward simulation and the classical post-processing are both governed by `L`.

### Result

| instance | support `M^d` | before | after |
|----------|---------------|--------|-------|
| N = 77   | 32,768        | 0.62 s | 0.083 s |
| N = 299  | 262,144       | 13.7 s | 0.011 s |
| N = 2021 | 16,777,216    | not attemptable | 0.79 s |

**1250x at N = 299.** Larger N is often *faster*, because a bigger modulus
makes the relation lattice sparser — fewer points to accumulate.

Verified exact: max deviation from the original construction is 2.8e-17
(float noise) across several `N`, `d`, `M`, and base sets. The old
per-class construction is kept in the test suite as an oracle
(`test_single_fft_matches_per_class_construction`).

`sample_ideal` was the next bottleneck — it called `sorted()` on all `M^d`
outcome tuples, hopeless at 16.7M. It now inverse-transform-samples from a
cached CDF and never materialises the outcome tuples. It is **bit-identical**
to the previous sampler on the same seed, which is why none of the numbers in
section 1 changed.

An `N > 3.04e9` guard rejects moduli where `int64` residue products would
silently overflow.

---

## 8. Larger instances: the main caveat, removed

The previous open items said a larger `N` would sharpen the separation, and
that N = 77's ~27% random success rate was "a limitation of the instance size".
Section 7 made that testable. It was correct:

`python scripts/control_experiment.py --scale` (300 trials per arm, seed 7):

| N    | n  | d | M  | k  | ideal quantum | uniform random | separation |
|------|----|---|----|----|---------------|----------------|------------|
| 77   | 7  | 3 | 32 | 5  | 99.0%         | 27.0%          | 72.0 pts   |
| 323  | 9  | 3 | 64 | 5  | 93.7%         | 6.7%           | 87.0 pts   |
| 713  | 10 | 4 | 64 | 5  | 96.3%         | 3.7%           | 92.7 pts   |
| 2021 | 11 | 4 | 64 | 5  | 90.0%         | 1.7%           | 88.3 pts   |
| 2021 | 11 | 4 | 64 | 8  | 97.7%         | 1.3%           | 96.3 pts   |
| 2021 | 11 | 4 | 64 | 12 | **99.3%**     | **1.0%**       | **98.3 pts** |

The random control falls from 27% to 1% as `N` grows — at N = 77 a rank-3
lattice over a 32-point range often contains a relation by luck, and that stops
being true quickly. Feeding the larger instance a few more samples recovers the
ideal arm to 99.3%.

So the strongest defensible statement is no longer "99% vs 27% at N = 77" but
**99.3% vs 1.0% at N = 2021**, with the caveat that motivated the original
hedge now measured away rather than argued away.

---

## 9. Under-resolution: a fourth silent-invalid-experiment mode

Regev's post-processing needs each measured vector to be *approximately*
orthogonal to the relations. That approximation comes from `QFT_M`
concentrating near multiples of `M / ord(a_i)`. An M-point DFT cannot resolve a
period `>= M` — plain aliasing — so if any base order reaches `M`, that
register is degenerate: `Z_M^d` fills with **exact** relations that exist
independently of the measured data.

The symptom is the now-familiar one: the uniform-random control starts
factoring as often as the quantum arm. At **N = 221 with M = 8** (an order is
exactly 8), random succeeds **83.5%** of the time — separation collapses from
~90 points to 14. The experiment would look like a success and demonstrate
nothing. This is the same failure family as contamination (section on the
contamination bug) and cherry-picking (section 1), reached a different way.

### It affects the default parameters

`M > max_order` is a **sound** necessary condition (it is just the Nyquist
limit), and the Regev-asymptotic `notebook` formula violates it for a majority
of the clean simulable instances:

```text
   N   d nd   M   max order   resolved?
  77   3  5   32     15        yes
 143   3  5   32     30        yes
 187   3  5   32     40        NO
 299   3  6   64     66        NO
 323   3  6   64     72        NO
 437   3  6   64     99        NO
```

These still often "work" — usually only the largest order exceeds `M`, and the
other registers carry the result — but they run on an unsound orthogonality
assumption, and both ideal success and separation measurably degrade.

### The fix

- `regev.bases.multiplicative_order` and `resolution_report(bases, N, M)`
  compute the orders (a simulation-only quantity) and return the `M / max_order`
  ratio and a `resolved` flag.
- `regev_parameters` now carries that `resolved` flag, and a new
  `mode="resolved"` enlarges `nd` until `M` exceeds every base order. Like
  `cover_2n`, it gives up Regev's register-size advantage, and is documented as
  a clean-demonstration mode rather than a faithful regime.
- `scripts/control_experiment.py` prints the orders and ratio, and warns when
  the chosen instance is under-resolved.

Sizing up to `resolved` improves every previously-under-resolved instance
(300-trial measurement, k = 5):

| N   | notebook M | ideal / rand / sep | resolved M | ideal / rand / sep |
|-----|-----------|--------------------|-----------|--------------------|
| 187 | 32        | 88.5 / 13.5 / 75.0 | 64        | 94.5 / 12.5 / 82.0 |
| 299 | 64        | 96.0 /  4.5 / 91.5 | 128       | 99.0 /  5.5 / 93.5 |
| 437 | 64        | 96.0 /  2.5 / 93.5 | 128       | 98.5 /  4.5 / 94.0 |

The default was **not** changed: `notebook` is Regev's intended regime and the
right thing to model. But it is no longer silent about being under-resolved,
and there is a principled one-flag path to a resolved instance.

---

## 10. Smaller corrections

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

## 11. Test suite: 29 → 112

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
- ~~**Larger clean instances.**~~ Done — see section 8. Now reaches N = 2021,
  where the random control succeeds 1.0% of the time instead of 27%.
- **Beyond N ≈ 2021.** The next parameter step is `M = 128, d = 4`, i.e. `M^d`
  = 268M cells ≈ 2 GB for the output array alone. The single-FFT engine is no
  longer the constraint; storing a dense distribution over the whole of
  `Z_M^d` is. Sampling does not actually need the dense array — walking the
  relation lattice directly, or sampling one register at a time from
  conditional marginals, would lift the ceiling again.
- **Aux-register uncomputation**, if a circuit is ever written: run the modexp
  on `|x>|1>|0>` and assert aux returns to `|0>`, or the measured distribution
  is entangled with garbage.
- **Wider moduli.** `_relation_autocorrelation` keeps residues in `int64`, so
  `N` is capped at ~3.04e9 by a guard. Object dtype or a split-multiply would
  lift it, though `M^d` binds long before that does.
- The `regev_parameters(mode="notebook")` string now refers to a `notebooks/`
  directory that is empty. The API string was left alone to avoid breaking
  callers, but it is a dangling name.