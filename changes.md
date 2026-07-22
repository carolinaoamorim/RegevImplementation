# Changes

Building on the original finite-model Regev factoring simulator, version 0.2.0
extends the scientific model, improves validation and performance, and adds
documentation and testing while preserving compatibility where practical.

## Scientific model

- Updated the default exponent state to Regev's discrete-Gaussian state.
- Preserved the original uniform model through
  `window="uniform"` for comparisons and reproducibility.
- Added configurable Gaussian radius support, with
  `R = D / (2 sqrt(d))` as the default.
- Updated the probability normalization for the Gaussian model and implemented
  Gaussian lag autocorrelation weights.
- Verified the optimized FFT distribution against an independent construction
  that groups exponent vectors by their modular-exponentiation output.
- Updated the default lattice scaling to use the integer-scaled Regev
  embedding
  `C = round(D sqrt(d) / (sqrt(2) R))`.
- Retained the original `weight=` embedding as a deprecated
  compatibility option.

## Experiment and methodology

- Extended the experiment runner so every trial draws fresh samples from the
  selected distribution.
- Kept the ideal and uniform-control arms on independent, reproducible random
  streams.
- Simplified relation processing by checking each verified relation once;
  pairwise sums of trivial square roots remain trivial and do not add useful
  factoring candidates.
- Added a check that each factoring base matches its corresponding prime
  squared modulo `N`.
- Clarified that individual multiplicative orders provide an optional
  axis-order sizing diagnostic rather than a complete condition for the
  multidimensional relation lattice.
- Renamed the diagnostic API to `order_report`; the old
  `resolution_report` name remains as a deprecated compatibility alias.
- Made `regev` the default parameter mode and retained `notebook` as a
  deprecated alias.
- Renamed the former `resolved` mode to the clearer `axis_order`, while
  preserving `resolved` as a deprecated alias.

## Validation and reliability

- Added strict validation for moduli, dimensions, sample counts, bases,
  measurement vectors, Gaussian radii, probabilities, bitstrings, and count
  dictionaries.
- Added clear errors for booleans and fractional values where integer inputs
  are required.
- Added validation for ragged or non-integer lattice matrices before LLL
  reduction.
- Added explicit handling for unsupported, prime, even, perfect-power,
  prime-power, and classically contaminated factoring inputs.
- Added early checks for number-theory input requirements.
- Added exact support for wide moduli through a Python-integer fallback when
  residue products no longer fit safely in `int64`.

## Sampling, noise, and performance

- Added read-only, bounded caches for exact probability arrays and sampling
  CDFs.
- Avoided constructing millions of Python outcome tuples during sampling.
- Made zero-sample requests return without allocating a distribution or CDF.
- Reworked total-variation-distance calculations to use bounded NumPy chunks
  instead of large tuple-keyed dictionaries.
- Made zero-noise TVD return after validation without building the ideal
  distribution.
- Documented the noise model as an all-or-nothing global depolarizing
  sensitivity model rather than a gate-level hardware simulation.
- Preserved exact seeded behavior at the ideal and uniform noise endpoints.
- Reused each LLL reduction in the experiment runner to reduce repeated
  post-processing work.

## Command-line experiment

- Made the script runnable directly from a fresh source checkout.
- Added positive-integer argument validation and mutually exclusive experiment
  modes.
- Set the default number of samples to `d + 4`.
- Added Wilson 95% confidence intervals for reported success rates.
- Added single-run, sample-count sweep, finite-instance scale, noise, and
  axis-order diagnostic modes.
- Applied factoring-input validation consistently across every mode.
- Added clear guidance when `--scale` is combined with `-N`, `-k`, or a mode
  outside its predefined sweep configuration.
- Added the Gaussian radius and axis-order diagnostic to the reported output.

## Documentation and packaging

- Expanded `README.md` into a project overview, setup guide, model description,
  API example, limitations section, and experiment guide.
- Clarified the scope of the finite simulator and distinguished it from a
  scalable gate-level implementation.
- Documented the exponential memory cost of the exact finite distribution.
- Updated package metadata and the project version to 0.2.0.
- Reduced runtime dependencies to NumPy and provided a `dev` extra for pytest.
- Added GitHub Actions coverage for Python 3.10 and Python 3.14.

## Verification

- The complete suite contains 223 passing tests.
- The suite also passes with Python warnings treated as errors.
- `git diff --check` reports no whitespace errors.
- Wheel and source distributions build without warnings.
- The wheel installs cleanly in an isolated Python 3.14 environment and
  passes `pip check`.
- The installed API example factors `77` into `7 × 11`.
- Direct help, single-run, scale, noise, and validation CLI smoke tests pass.

The project remains focused on classical exact simulation of small finite
instances, providing a foundation for further experiments and future circuit
work.
