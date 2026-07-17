"""Tests for Regev's classical post-processing.

Includes the quantum-vs-random control: the experiment that establishes the
end-to-end factoring result is actually attributable to the quantum samples.
"""

import random

import pytest

from regev.bases import generate_regev_bases
from regev.postprocess import (
    regev_relation_lattice,
    eval_relation,
    is_relation,
    relation_to_congruence,
    regev_factor,
)

# Ideal Regev samples for N = 77, d = 3, M = 32, bases (4, 9, 25).
# Highest-probability vectors of the exact distribution (see regev.reference).
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


@pytest.mark.parametrize("weight", [1, 2, 4, 8])
def test_relation_lattice_stable_across_weights(weight):
    """Result must not be an artifact of the embedding weight."""
    cands = regev_relation_lattice(N77_VECTORS, M77, D77, weight=weight)
    rels = [e for _, e, _ in cands if is_relation(e, BASES77, N77)]
    assert len(rels) >= 3


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
    the end-to-end result is not evidence the circuit works. Quantum samples
    should yield a full basis of relations; random vectors should mostly not.

    Note: at N = 77 random vectors succeed occasionally (a small lattice
    sometimes contains a relation by luck), so this asserts a separation,
    not that random never succeeds.
    """
    random.seed(7)
    trials = 20

    q_rel = 0
    for _ in range(trials):
        cands = regev_relation_lattice(N77_VECTORS, M77, D77)
        q_rel += len([e for _, e, _ in cands if is_relation(e, BASES77, N77)])

    r_rel = 0
    r_succ = 0
    for _ in range(trials):
        rv = [tuple(random.randrange(M77) for _ in range(D77)) for _ in range(5)]
        cands = regev_relation_lattice(rv, M77, D77)
        r_rel += len([e for _, e, _ in cands if is_relation(e, BASES77, N77)])
        if regev_factor(rv, M77, D77, N77, BASES77, PRIMES77, verbose=False):
            r_succ += 1

    q_avg = q_rel / trials
    r_avg = r_rel / trials

    assert q_avg >= 2.5, f"quantum samples gave only {q_avg} relations/basis"
    assert r_avg < 1.5, f"random gave {r_avg} relations/basis -- no separation"
    assert q_avg > 2 * r_avg, f"insufficient separation: {q_avg} vs {r_avg}"
    assert r_succ < trials, "random control succeeded every time -- pipeline is vacuous"
