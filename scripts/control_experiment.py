"""Quantum-vs-random control experiment for Regev post-processing.

Reproduces the tables in the README:

    python scripts/control_experiment.py            # single k
    python scripts/control_experiment.py --sweep    # k = 3, 5, 8, 12, 16
    python scripts/control_experiment.py --scale    # sweep N
    python scripts/control_experiment.py --noise    # sweep depolarising rate

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

from regev.bases import generate_regev_bases, resolution_report
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


# (N, k) pairs spanning the instance sizes the exact distribution can reach.
# All are odd, composite, non-perfect-power, and coprime to the first d primes.
SCALE_INSTANCES = [(77, 5), (323, 5), (713, 5), (2021, 5), (2021, 8), (2021, 12)]


def run_scale(trials: int, seed: int):
    """Show how the quantum/random separation sharpens with instance size.

    At N = 77 the random control succeeds ~27% of the time, because a rank-3
    lattice over a 32-point range often contains a relation by chance. That
    was previously the headline caveat on the whole experiment. Larger N
    shrinks it to noise.
    """
    print(f"trials = {trials} per arm, seed = {seed}\n")
    print("      N   n  d    M   k |     ideal quantum |    uniform random | separation")
    for N, k in SCALE_INSTANCES:
        params = regev_parameters(N)
        d, M = params["d"], params["M"]
        bases, primes, trivial = generate_regev_bases(N, d)
        if trivial or not validate_factor_input(N)["valid"]:
            print(f"  {N}: skipped (contaminated or invalid)")
            continue

        counts = {}
        for name, sampler in (
            ("q", lambda rng, kk=k: sample_ideal(bases, N, d, M, kk, rng)),
            ("u", lambda rng, kk=k: sample_uniform(d, M, kk, rng)),
        ):
            rng = random.Random(seed)
            counts[name] = sum(
                regev_factor(sampler(rng), M, d, N, bases, primes, verbose=False)
                is not None
                for _ in range(trials)
            )

        ql, qh = wilson_interval(counts["q"], trials)
        ul, uh = wilson_interval(counts["u"], trials)
        print(
            f"  {N:5d} {params['n']:3d}  {d}  {M:3d} {k:3d} | "
            f"{counts['q'] * 100 / trials:5.1f}% [{ql * 100:4.1f},{qh * 100:5.1f}] | "
            f"{counts['u'] * 100 / trials:5.1f}% [{ul * 100:4.1f},{uh * 100:4.1f}] | "
            f"{(counts['q'] - counts['u']) * 100 / trials:6.1f} pts"
        )


NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0]


def run_noise(N: int, k: int, trials: int, seed: int):
    """Sweep the depolarising rate: how much noise the post-processing tolerates.

    lam = 0 is the ideal quantum sampler, lam = 1 is the uniform-random control
    (the two arms of the main experiment), so this interpolates between them.
    """
    from regev.noise import noisy_tvd_from_ideal, sample_noisy

    params = regev_parameters(N, mode="resolved")
    d, M = params["d"], params["M"]
    bases, primes, trivial = generate_regev_bases(N, d)
    if trivial:
        raise SystemExit(f"N = {N} is contaminated; pick a clean instance.")

    print(f"N = {N} | d = {d} | M = {M} (resolved) | k = {k} | "
        f"trials = {trials}, seed = {seed}")
    print("  depolarising rate -> success (identical post-processing)\n")
    print("  lam    success   95% CI (Wilson)   TVD from ideal")
    for lam in NOISE_LEVELS:
        rng = random.Random(seed)
        succ = sum(
            regev_factor(sample_noisy(bases, N, d, M, k, lam, rng),
                        M, d, N, bases, primes, verbose=False) is not None
            for _ in range(trials)
        )
        lo, hi = wilson_interval(succ, trials)
        tvd = noisy_tvd_from_ideal(bases, N, d, M, lam)
        print(f"  {lam:0.2f}  {succ * 100 / trials:6.1f}%   "
              f"[{lo * 100:5.1f}, {hi * 100:5.1f}]      {tvd:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-N", type=int, default=77, help="modulus to factor")
    ap.add_argument("-k", type=int, default=5, help="measured vectors per trial")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mode", default="regev",
                    choices=["regev", "notebook", "cover_2n", "resolved"],
                    help="register-sizing mode (see regev.parameters)")
    ap.add_argument("--sweep", action="store_true", help="sweep k rather than fix it")
    ap.add_argument(
        "--scale",
        action="store_true",
        help="sweep N instead, showing how separation sharpens with instance size",
    )
    ap.add_argument(
        "--noise",
        action="store_true",
        help="sweep the depolarising rate at fixed N, k (circuit-free noise model)",
    )
    args = ap.parse_args()

    if args.scale:
        run_scale(args.trials, args.seed)
        return
    if args.noise:
        run_noise(args.N, args.k, args.trials, args.seed)
        return

    check = validate_factor_input(args.N)
    if not check["valid"]:
        raise SystemExit(f"N = {args.N} unusable: {check['reason']}")

    params = regev_parameters(args.N, mode=args.mode)
    d, M = params["d"], params["M"]
    bases, primes, trivial = generate_regev_bases(args.N, d)
    if trivial:
        raise SystemExit(
            f"N = {args.N} is CONTAMINATED: prime(s) {trivial} divide it, so "
            "trial division would factor it during basis selection. Excluded."
        )

    res = resolution_report(bases, args.N, M)
    print(f"N = {args.N} | n = {params['n']} | d = {d} | M = {M} "
            f"| mode = {args.mode} | bases = {bases}")
    print(f"base orders = {res['orders']} | M/max_order = {res['ratio']:.2f}")
    if not res["resolved"]:
        print("  WARNING: under-resolved -- " + res["reason"])
        print("  (re-run with --mode resolved to size M above every base order)")
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
