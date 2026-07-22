"""Quantum-vs-random control experiment for Regev post-processing.

Run the finite model directly:

    python scripts/control_experiment.py            # single k
    python scripts/control_experiment.py --sweep    # k = 3, 5, 8, 12, 16
    python scripts/control_experiment.py --scale    # sweep N
    python scripts/control_experiment.py --noise    # sweep depolarising rate

Both arms are genuinely sampled. The "ideal quantum" arm draws from the exact
output distribution of the finite Gaussian circuit (`regev.reference`), including
the uninformative w = 0 outcome and the low-probability tail; the control arm
draws uniformly from Z_M^d. Identical post-processing runs on both, so any
separation is attributable to the distribution and nothing else.

Earlier versions handed the quantum arm a fixed list of the five
highest-probability vectors. See `regev/sampling.py` for why that was replaced.
"""

import argparse
import math
from pathlib import Path
import random
import sys


# Direct execution sets sys.path[0] to ``scripts/``, so the sibling ``regev``
# package is otherwise invisible in a fresh source checkout. Resolve the path
# from this file (never from the caller's working directory) and only bootstrap
# it for direct execution; installed/module execution keeps normal import rules.
if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from regev.bases import generate_regev_bases, order_report
from regev.parameters import regev_parameters, validate_factor_input
from regev.postprocess import (
    is_relation,
    relation_to_congruence,
    regev_factor,
    regev_relation_lattice,
)
from regev.sampling import sample_ideal, sample_uniform


MODE_CHOICES = ("regev", "cover_2n", "axis_order", "notebook", "resolved")


def positive_int(value: str) -> int:
    """argparse converter for strictly positive integers."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} is not a positive integer")
    return parsed


def sample_count(k: int | None, d: int) -> int:
    """Resolve the paper-style default number of samples for dimension ``d``."""
    return d + 4 if k is None else k


def require_valid_factor_input(N: int) -> None:
    """Exit with a useful message when ``N`` is not a factoring instance."""
    check = validate_factor_input(N)
    if not check["valid"]:
        raise SystemExit(f"N = {N} unusable: {check['reason']}")


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
        # Reuse the candidates already reduced above. Calling regev_factor here
        # used to repeat the same LLL reduction a second time on every trial.
        if any(relation_to_congruence(e, primes, N) for _, e in rels):
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
    """Measure quantum/control separation across finite model instances."""
    print(
        f"trials = {trials} per arm, ideal seed = {seed}, "
        f"control seed = {seed + 1}\n"
    )
    print("      N   n  d    M   k |     ideal quantum |    uniform random | separation")
    for N, k in SCALE_INSTANCES:
        params = regev_parameters(N)
        d, M = params["d"], params["M"]
        bases, primes, trivial = generate_regev_bases(N, d)
        if trivial or not validate_factor_input(N)["valid"]:
            print(f"  {N}: skipped (contaminated or invalid)")
            continue

        counts = {}
        for name, arm_seed, sampler in (
            ("q", seed, lambda rng, kk=k: sample_ideal(bases, N, d, M, kk, rng)),
            ("u", seed + 1, lambda rng, kk=k: sample_uniform(d, M, kk, rng)),
        ):
            rng = random.Random(arm_seed)
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


def run_noise(N: int, k: int | None, trials: int, seed: int, mode: str = "regev"):
    """Sweep the depolarising rate: how much noise the post-processing tolerates.

    lam = 0 is the ideal quantum sampler, lam = 1 is the uniform-random control
    (the two arms of the main experiment), so this interpolates between them.
    """
    from regev.noise import noisy_tvd_from_ideal, sample_noisy

    require_valid_factor_input(N)
    params = regev_parameters(N, mode=mode)
    d, M = params["d"], params["M"]
    k = sample_count(k, d)
    bases, primes, trivial = generate_regev_bases(N, d)
    if trivial:
        raise SystemExit(f"N = {N} is contaminated; pick a clean instance.")

    print(
        f"N = {N} | d = {d} | M = {M} | R = {params['R']:.3f} | "
        f"mode = {params['mode']} | k = {k} | "
        f"trials = {trials}, seed = {seed}"
    )
    print("  depolarising rate -> success (identical post-processing)\n")
    print("  lam    success   95% CI (Wilson)   TVD from ideal")
    full_tvd = noisy_tvd_from_ideal(bases, N, d, M, 1.0)
    for lam in NOISE_LEVELS:
        rng = random.Random(seed)
        succ = sum(
            regev_factor(sample_noisy(bases, N, d, M, k, lam, rng),
                        M, d, N, bases, primes, verbose=False) is not None
            for _ in range(trials)
        )
        lo, hi = wilson_interval(succ, trials)
        tvd = lam * full_tvd
        print(f"  {lam:0.2f}  {succ * 100 / trials:6.1f}%   "
              f"[{lo * 100:5.1f}, {hi * 100:5.1f}]      {tvd:.3f}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-N",
        type=positive_int,
        default=None,
        help="modulus to factor (default: 77; unavailable with --scale)",
    )
    ap.add_argument(
        "-k",
        type=positive_int,
        default=None,
        help="measured vectors per trial (default: d + 4)",
    )
    ap.add_argument("--trials", type=positive_int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--mode",
        default="regev",
        choices=MODE_CHOICES,
        help=(
            "register-sizing mode; notebook/resolved are deprecated aliases "
            "(see regev.parameters)"
        ),
    )
    experiment = ap.add_mutually_exclusive_group()
    experiment.add_argument(
        "--sweep", action="store_true", help="sweep k rather than fix it"
    )
    experiment.add_argument(
        "--scale",
        action="store_true",
        help="compare quantum/control separation across finite instances",
    )
    experiment.add_argument(
        "--noise",
        action="store_true",
        help="sweep the depolarising rate at fixed N, k (circuit-free noise model)",
    )
    return ap


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.scale and args.N is not None:
        parser.error("--scale uses predefined N values; do not pass -N")
    if args.scale and args.k is not None:
        parser.error("--scale uses predefined (N, k) pairs; do not pass -k")
    if args.scale and args.mode != "regev":
        parser.error("--scale uses mode='regev'; do not pass --mode")

    if args.N is None:
        args.N = 77

    # Validate before dispatch so every execution mode, including --noise and
    # --scale, rejects unusable user-supplied factoring instances consistently.
    require_valid_factor_input(args.N)

    if args.scale:
        run_scale(args.trials, args.seed)
        return
    if args.noise:
        run_noise(args.N, args.k, args.trials, args.seed, mode=args.mode)
        return

    params = regev_parameters(args.N, mode=args.mode)
    d, M = params["d"], params["M"]
    bases, primes, trivial = generate_regev_bases(args.N, d)
    if trivial:
        raise SystemExit(
            f"N = {args.N} is CONTAMINATED: prime(s) {trivial} divide it, so "
            "trial division would factor it during basis selection. Excluded."
        )

    res = order_report(bases, args.N, M)
    print(f"N = {args.N} | n = {params['n']} | d = {d} | M = {M} "
            f"| R = {params['R']:.3f} "
            f"| mode = {params['mode']} | bases = {bases}")
    print(f"base orders = {res['orders']} | M/max_order = {res['ratio']:.2f}")
    if not res["covers_axis_orders"]:
        print("  axis-order diagnostic: " + res["reason"])
        print("  (--mode axis_order covers each individual order; this is optional)")
    print(
        f"trials = {args.trials} per arm, ideal seed = {args.seed}, "
        f"control seed = {args.seed + 1}\n"
    )
    print("  arm                success  95% CI (Wilson)  rel/basis  mean min |e|_1")

    ks = [3, 5, 8, 12, 16] if args.sweep else [sample_count(args.k, d)]
    for k in ks:
        print(f"  -- k = {k} measured vectors --")
        common = dict(
            N=args.N,
            d=d,
            M=M,
            bases=bases,
            primes=primes,
            k=k,
            trials=args.trials,
        )
        run_arm(lambda rng, kk: sample_ideal(bases, args.N, d, M, kk, rng),
                "ideal quantum", seed=args.seed, **common)
        run_arm(lambda rng, kk: sample_uniform(d, M, kk, rng),
                "uniform random", seed=args.seed + 1, **common)


if __name__ == "__main__":
    main()
