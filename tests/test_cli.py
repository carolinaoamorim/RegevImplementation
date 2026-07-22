"""Subprocess coverage for the documented control-experiment CLI."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.control_experiment as cli


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "control_experiment.py"


def run_cli(*args, cwd=None, without_site_packages=True):
    """Run the source-tree script with no PYTHONPATH assistance."""
    command = [sys.executable]
    if without_site_packages:
        # -S ensures an installed copy of this project cannot mask a broken
        # source-checkout bootstrap. --help and validation need no third-party
        # imports because numpy remains lazy.
        command.append("-S")
    command.extend((str(SCRIPT), *(str(arg) for arg in args)))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_direct_source_checkout_help_works(tmp_path):
    result = run_cli("--help", cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "axis_order" in result.stdout
    assert "default: d + 4" in result.stdout


@pytest.mark.parametrize(
    "option,value",
    [("-N", "0"), ("-k", "0"), ("--trials", "0"), ("--trials", "-2")],
)
def test_positive_integer_arguments_are_enforced(tmp_path, option, value):
    result = run_cli(option, value, cwd=tmp_path)

    assert result.returncode == 2
    assert "positive integer" in result.stderr


@pytest.mark.parametrize(
    "flags",
    [("--sweep", "--scale"), ("--sweep", "--noise"), ("--scale", "--noise")],
)
def test_experiment_modes_are_mutually_exclusive(tmp_path, flags):
    result = run_cli(*flags, cwd=tmp_path)

    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


@pytest.mark.parametrize("mode", [(), ("--sweep",), ("--noise",)])
def test_invalid_factor_input_is_rejected_before_every_mode(tmp_path, mode):
    result = run_cli(*mode, "-N", "49", "--trials", "1", cwd=tmp_path)

    assert result.returncode != 0
    assert "prime power" in result.stderr


@pytest.mark.parametrize(
    "args,message",
    [
        (("-N", "77"), "predefined N values"),
        (("-k", "5"), "predefined (N, k) pairs"),
        (("--mode", "cover_2n"), "uses mode='regev'"),
    ],
)
def test_scale_rejects_options_it_would_ignore(tmp_path, args, message):
    result = run_cli("--scale", *args, cwd=tmp_path)

    assert result.returncode == 2
    assert message in result.stderr


def test_default_sample_count_tracks_dimension():
    assert cli.sample_count(None, 3) == 7
    assert cli.sample_count(None, 4) == 8
    assert cli.sample_count(11, 4) == 11


def test_noise_run_uses_dimension_based_sample_default(monkeypatch):
    observed = []

    monkeypatch.setattr(
        cli,
        "regev_parameters",
        lambda N, mode: {"d": 4, "M": 16, "R": 4.0, "mode": mode},
    )
    monkeypatch.setattr(
        cli,
        "generate_regev_bases",
        lambda N, d: ([4, 9, 25, 49], [2, 3, 5, 7], []),
    )
    monkeypatch.setattr(cli, "NOISE_LEVELS", [0.0])
    monkeypatch.setattr(cli, "regev_factor", lambda *args, **kwargs: None)

    import regev.noise as noise

    def fake_sample_noisy(bases, N, d, M, k, lam, rng):
        observed.append(k)
        return [(0,) * d] * k

    monkeypatch.setattr(noise, "sample_noisy", fake_sample_noisy)
    monkeypatch.setattr(noise, "noisy_tvd_from_ideal", lambda *args: 0.0)

    cli.run_noise(77, None, trials=1, seed=7)

    assert observed == [8]


def test_single_run_uses_dimension_based_sample_default(tmp_path):
    result = run_cli("--trials", "1", cwd=tmp_path, without_site_packages=False)

    assert result.returncode == 0, result.stderr
    assert "-- k = 7 measured vectors --" in result.stdout


def test_run_arm_reuses_each_lattice_reduction(monkeypatch):
    reductions = 0

    def reduce_once(vectors, M, d):
        nonlocal reductions
        reductions += 1
        return [(1, [1], [])]

    monkeypatch.setattr(cli, "regev_relation_lattice", reduce_once)
    monkeypatch.setattr(cli, "is_relation", lambda e, bases, N: True)
    monkeypatch.setattr(cli, "relation_to_congruence", lambda e, primes, N: (7, 11))
    monkeypatch.setattr(
        cli,
        "regev_factor",
        lambda *args, **kwargs: pytest.fail("run_arm repeated post-processing"),
    )

    successes = cli.run_arm(
        lambda rng, k: [(0,)] * k,
        "test",
        N=77,
        d=1,
        M=2,
        bases=[1],
        primes=[1],
        k=1,
        trials=3,
        seed=7,
    )

    assert successes == 3
    assert reductions == 3
