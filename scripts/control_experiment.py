"""Quantum-vs-random control experiment for Regev post-processing.

Reproduces the table in the README:

    python scripts/control_experiment.py            # single k
    python scripts/control_experiment.py --sweep    # k = 3, 5, 8, 12, 16

Both arms are genuinely sampled. The "ideal quantum" arm draws from the exact
output distribution of the noiseless circuit (`regev.reference`), including
the uninformative w = 0 outcome and the low-probability tail; the control arm
draws uniformly from Z_M^d. Identical post-processing runs on both, so any
separation is attributable to the distribution and nothing else.

Earlier versions handed the quantum arm a fixed list of the five
highest-probability vectors. See `regev/sampling.py` for why that was replaced.
"""

import argparse
import math
import random

from regev.bases import generate_regev_bases
from regev.parameters import regev_parameters, validate_factor_input
from regev.postprocess import is_relation, regev_factor, regev_relation_lattice
from regev.sampling import sample_ideal, sample_uniform


def wilson_interval(successes: int, trials: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation here because the success rates
    sit near 0 and 1, where the naive interval runs outside [0, 1].
    """
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_arm(sampler, label, *, N, d, M, bases, primes, k, trials, seed):
    """Run `trials` independent factoring attempts and summarise them."""
    rng = random.Random(seed)
    successes = 0
    relation_count = 0
    min_norms = []

    for _ in range(trials):
        vectors = sampler(rng, k)
        cands = regev_relation_lattice(vectors, M, d)
        rels = [(nm, e) for nm, e, _ in cands if is_relation(e, bases, N)]
        relation_count += len(rels)
        if rels:
            min_norms.append(min(nm for nm, _ in rels))
        if regev_factor(vectors, M, d, N, bases, primes, verbose=False):
            successes += 1

    lo, hi = wilson_interval(successes, trials)
    mean_norm = sum(min_norms) / len(min_norms) if min_norms else float("nan")
    print(
        f"  {label:16s} {successes:4d}/{trials}  "
        f"[{lo * 100:5.1f}%, {hi * 100:5.1f}%]  "
        f"{relation_count / trials:9.2f}  "
        f"{mean_norm:9.1f}"
    )
    return successes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-N", type=int, default=77, help="modulus to factor")
    ap.add_argument("-k", type=int, default=5, help="measured vectors per trial")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sweep", action="store_true", help="sweep k rather than fix it")
    args = ap.parse_args()

    check = validate_factor_input(args.N)
    if not check["valid"]:
        raise SystemExit(f"N = {args.N} unusable: {check['reason']}")

    params = regev_parameters(args.N)
    d, M = params["d"], params["M"]
    bases, primes, trivial = generate_regev_bases(args.N, d)
    if trivial:
        raise SystemExit(
            f"N = {args.N} is CONTAMINATED: prime(s) {trivial} divide it, so "
            "trial division would factor it during basis selection. Excluded."
        )

    print(f"N = {args.N} | n = {params['n']} | d = {d} | M = {M} | bases = {bases}")
    print(f"trials = {args.trials} per arm, seed = {args.seed}\n")
    print("  arm                success  95% CI (Wilson)  rel/basis  mean min |e|_1")

    for k in ([3, 5, 8, 12, 16] if args.sweep else [args.k]):
        print(f"  -- k = {k} measured vectors --")
        common = dict(N=args.N, d=d, M=M, bases=bases, primes=primes,
                      k=k, trials=args.trials, seed=args.seed)
        run_arm(lambda rng, kk: sample_ideal(bases, args.N, d, M, kk, rng),
                "ideal quantum", **common)
        run_arm(lambda rng, kk: sample_uniform(d, M, kk, rng),
                "uniform random", **common)


if __name__ == "__main__":
    main()
