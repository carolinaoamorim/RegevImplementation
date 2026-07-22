"""Tests for Regev's classical post-processing.

Includes the quantum-vs-random control: the experiment that establishes the
end-to-end factoring result is actually attributable to the quantum samples.
"""

import random

import pytest

from regev.bases import generate_regev_bases
from regev.postprocess import (
    regev_embedding_scale,
    regev_relation_lattice,
    eval_relation,
    is_relation,
    relation_to_congruence,
    regev_factor,
)
from regev.sampling import sample_ideal, sample_uniform

# A fixed known-good input for deterministic unit tests. These were prominent
# outcomes of the legacy rectangular model and remain relation-rich, but they
# are only a fixture; the experiment below samples the Gaussian model.
N77_VECTORS = [(4, 2, 17), (28, 30, 15), (19, 26, 13), (13, 6, 19), (2, 17, 9)]
N77 = 77
D77, M77 = 3, 32
BASES77, PRIMES77 = [4, 9, 25], [2, 3, 5]


def test_eval_relation_positive():
    assert eval_relation([1, 0, 0], [4, 9, 25], 77) == 4
    assert eval_relation([0, 0, 0], [4, 9, 25], 77) == 1


def test_eval_relation_negative_exponent():
    # 25^-1 mod 77 exists since gcd(25,77)=1
    v = eval_relation([0, 0, -1], [4, 9, 25], 77)
    assert v * 25 % 77 == 1


def test_is_relation_true_case():
    # 9^1 * 25^-2 = 1 mod 77
    assert is_relation([0, 1, -2], BASES77, N77)


def test_is_relation_false_case():
    assert not is_relation([1, 0, 0], BASES77, N77)


def test_orders_are_nontrivial():
    """Guard: bases must not have degenerate order 1."""
    for a in BASES77:
        assert a % N77 != 1


def test_relation_lattice_finds_relations():
    cands = regev_relation_lattice(N77_VECTORS, M77, D77)
    rels = [e for _, e, _ in cands if is_relation(e, BASES77, N77)]
    assert len(rels) >= 3, f"expected >=3 relations, got {rels}"


def test_relation_lattice_relations_are_short():
    """Relations from quantum samples should be the SHORT vectors."""
    cands = regev_relation_lattice(N77_VECTORS, M77, D77)
    rels = [(nm, e) for nm, e, _ in cands if is_relation(e, BASES77, N77)]
    assert min(nm for nm, _ in rels) <= 4


def test_default_embedding_scale_comes_from_gaussian_error():
    assert regev_embedding_scale(M77, D77) == round(2 ** 0.5 * D77)
    assert regev_embedding_scale(M77, D77, radius=8.0) == 5


def test_embedding_rejects_radius_that_rounds_scale_to_zero():
    with pytest.raises(ValueError, match="rounds to zero"):
        regev_embedding_scale(M77, D77, radius=10_000.0)


def test_legacy_weight_is_explicit_and_deprecated():
    with pytest.warns(DeprecationWarning, match="weight is deprecated"):
        cands = regev_relation_lattice(N77_VECTORS, M77, D77, weight=4)
    assert any(is_relation(e, BASES77, N77) for _, e, _ in cands)

    with pytest.raises(ValueError, match="cannot be combined"):
        regev_relation_lattice(
            N77_VECTORS, M77, D77, weight=4, exponent_scale=4
        )

    with pytest.warns(DeprecationWarning, match="weight is deprecated"):
        result = regev_factor(
            N77_VECTORS,
            M77,
            D77,
            N77,
            BASES77,
            PRIMES77,
            weight=4,
            verbose=False,
        )
    assert result is not None


@pytest.mark.parametrize("exponent_scale", [2, 4, 6, 8])
def test_relation_lattice_stable_near_derived_scale(exponent_scale):
    """The deterministic fixture remains useful near the derived scale."""
    cands = regev_relation_lattice(
        N77_VECTORS, M77, D77, exponent_scale=exponent_scale
    )
    rels = [e for _, e, _ in cands if is_relation(e, BASES77, N77)]
    assert len(rels) >= 2


def test_relation_to_congruence_splits_77():
    p, q = relation_to_congruence([0, 1, -2], PRIMES77, N77)
    assert {p, q} == {7, 11}


def test_relation_to_congruence_rejects_trivial():
    """The zero relation gives u = 1, a trivial square root."""
    assert relation_to_congruence([0, 0, 0], PRIMES77, N77) is None


def test_regev_factor_end_to_end():
    result = regev_factor(N77_VECTORS, M77, D77, N77, BASES77, PRIMES77, verbose=False)
    assert result is not None
    p, q = result
    assert p * q == N77
    assert {p, q} == {7, 11}


def test_quantum_beats_random_control():
    """THE control experiment.

    If uniformly random vectors factor N as often as the quantum samples do,
    the end-to-end result is not evidence the circuit works.

    Both arms are genuinely resampled every trial. An earlier version of this
    test re-ran the same fixed five vectors 20 times and called it 20 trials;
    that measures nothing but the variance of a constant, and it silently gave
    the quantum arm the best case in the distribution. The ideal arm here
    draws from the true distribution, w = 0 and long tail included.

    At N = 77 random vectors still succeed sometimes (a rank-3 lattice
    occasionally contains a relation by luck), so this asserts a separation,
    not that random never succeeds.
    """
    trials, k = 60, D77 + 4
    q_rng = random.Random(7)
    r_rng = random.Random(8)

    q_succ = q_rel = 0
    r_succ = r_rel = 0
    for _ in range(trials):
        qv = sample_ideal(BASES77, N77, D77, M77, k, q_rng)
        q_rel += len([e for _, e, _ in regev_relation_lattice(qv, M77, D77)
                      if is_relation(e, BASES77, N77)])
        q_succ += regev_factor(qv, M77, D77, N77, BASES77, PRIMES77,
                               verbose=False) is not None

        rv = sample_uniform(D77, M77, k, r_rng)
        r_rel += len([e for _, e, _ in regev_relation_lattice(rv, M77, D77)
                      if is_relation(e, BASES77, N77)])
        r_succ += regev_factor(rv, M77, D77, N77, BASES77, PRIMES77,
                               verbose=False) is not None

    q_avg, r_avg = q_rel / trials, r_rel / trials

    assert q_succ / trials >= 0.85, f"ideal sampling factored only {q_succ}/{trials}"
    assert r_succ < q_succ / 2, f"no separation: random {r_succ} vs ideal {q_succ}"
    assert q_avg >= 2.5, f"ideal samples gave only {q_avg} relations/basis"
    assert r_avg < 1.5, f"random gave {r_avg} relations/basis -- no separation"
    assert r_succ > 0, (
        "random control never succeeded -- suspicious at this instance size; "
        "check the control is actually running"
    )
