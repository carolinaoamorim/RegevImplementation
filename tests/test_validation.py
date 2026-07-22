"""Boundary-validation regressions for the public classical APIs."""

from fractions import Fraction

import pytest

import regev.postprocess as postprocess
from regev.bases import (
    generate_regev_bases,
    integer_nth_root,
    is_prime_basic,
    is_prime_power,
    multiplicative_order,
    order_report,
    perfect_power,
)
from regev.lattice import gram_schmidt, lll
from regev.postprocess import (
    eval_relation,
    relation_to_congruence,
    regev_factor,
    regev_relation_lattice,
)


@pytest.mark.parametrize(
    "function,args",
    [
        (is_prime_basic, (2.5,)),
        (integer_nth_root, (4.0, 2)),
        (integer_nth_root, (4, 2.0)),
        (perfect_power, (8.0,)),
        (is_prime_power, (8.0,)),
    ],
)
def test_number_theory_helpers_reject_non_integer_inputs(function, args):
    with pytest.raises(TypeError, match="must be an integer"):
        function(*args)


def test_number_theory_predicates_preserve_integral_negative_behavior():
    assert is_prime_basic(-3) is False
    assert perfect_power(-8) is None
    assert is_prime_power(-8) is False


@pytest.mark.parametrize("N", [-5, 0, 1])
def test_generate_bases_rejects_invalid_modulus_without_hanging(N):
    with pytest.raises(ValueError, match="N must be at least 2"):
        generate_regev_bases(N, 1)


@pytest.mark.parametrize("d", [-1, 0])
def test_generate_bases_requires_a_positive_dimension(d):
    with pytest.raises(ValueError, match="d must be positive"):
        generate_regev_bases(77, d)


def test_multiplicative_order_rejects_modulus_one_without_hanging():
    with pytest.raises(ValueError, match="N must be at least 2"):
        multiplicative_order(1, 1)


@pytest.mark.parametrize("M", [-1, 0])
def test_order_report_requires_positive_M(M):
    with pytest.raises(ValueError, match="M must be positive"):
        order_report([4], 77, M)


def test_order_report_requires_at_least_one_base():
    with pytest.raises(ValueError, match="at least one"):
        order_report([], 77, 32)


def test_order_report_exposes_honestly_named_axis_diagnostic():
    report = order_report([4, 9, 25], 77, 32)
    assert report["covers_axis_orders"] is True
    assert "resolved" not in report


@pytest.mark.parametrize("reducer", [lll, gram_schmidt])
def test_lattice_operations_reject_ragged_rows(reducer):
    with pytest.raises(ValueError, match="all have length"):
        reducer([[1, 0], [0]])


@pytest.mark.parametrize("value", [1.5, Fraction(3, 2), "1"])
def test_lll_rejects_non_integer_entries_instead_of_truncating(value):
    with pytest.raises(TypeError, match=r"B\[0\]\[0\] must be an integer"):
        lll([[value, 0], [0, 1]])


@pytest.mark.parametrize("delta", [0, Fraction(1, 4), 1, Fraction(5, 4)])
def test_lll_rejects_delta_outside_the_Lovasz_interval(delta):
    with pytest.raises(ValueError, match="strictly between"):
        lll([[1, 0], [0, 1]], delta=delta)


@pytest.mark.parametrize(
    "e,bases",
    [([], []), ([], [4]), ([1], [4, 9]), ([1, 2], [4])],
)
def test_eval_relation_requires_matching_nonempty_dimensions(e, bases):
    with pytest.raises(ValueError):
        eval_relation(e, bases, 77)


@pytest.mark.parametrize(
    "e,primes",
    [([], []), ([], [2]), ([1], [2, 3]), ([1, 2], [2])],
)
def test_relation_to_congruence_requires_matching_nonempty_dimensions(e, primes):
    with pytest.raises(ValueError):
        relation_to_congruence(e, primes, 77)


@pytest.mark.parametrize(
    "M,d,exponent_scale,match",
    [
        (0, 2, 4, "M must be positive"),
        (8, 0, 4, "d must be positive"),
        (8, 2, 0, "exponent_scale must be positive"),
    ],
)
def test_relation_lattice_requires_positive_scalar_arguments(
    M, d, exponent_scale, match
):
    with pytest.raises(ValueError, match=match):
        regev_relation_lattice(
            [(1, 2)], M, d, exponent_scale=exponent_scale
        )


def test_relation_lattice_requires_measurements():
    with pytest.raises(ValueError, match="at least one measured vector"):
        regev_relation_lattice([], 8, 2)


@pytest.mark.parametrize("vector", [(1,), (1, 2, 3)])
def test_relation_lattice_rejects_wrong_vector_dimension(vector):
    with pytest.raises(ValueError, match="length d"):
        regev_relation_lattice([vector], 8, 2)


def test_relation_lattice_rejects_non_integer_coordinates():
    with pytest.raises(TypeError, match="must be an integer"):
        regev_relation_lattice([(1.5, 2)], 8, 2)


@pytest.mark.parametrize("coordinate", [-1, 8])
def test_relation_lattice_rejects_coordinates_outside_Z_M(coordinate):
    with pytest.raises(ValueError, match=r"must be in \[0, M\)"):
        regev_relation_lattice([(coordinate, 2)], 8, 2)


@pytest.mark.parametrize(
    "bases,primes",
    [([4, 9], [2, 3, 5]), ([4, 9, 25], [2, 3])],
)
def test_regev_factor_requires_d_bases_and_primes(bases, primes):
    with pytest.raises(ValueError, match="must each have length d"):
        regev_factor([(1, 2, 3)], 8, 3, 77, bases, primes, verbose=False)


def test_regev_factor_requires_bases_to_match_squared_primes():
    with pytest.raises(ValueError, match="squared modulo N"):
        regev_factor(
            [(1, 2, 3)],
            8,
            3,
            77,
            [4, 9, 25],
            [2, 3, 7],
            verbose=False,
        )


def test_regev_factor_does_not_try_pairwise_relation_sums(monkeypatch):
    """Products of trivial roots remain trivial, so sums cannot reveal a factor."""
    candidates = [(1, [1], []), (2, [2], [])]
    checked = []

    monkeypatch.setattr(
        postprocess,
        "regev_relation_lattice",
        lambda vectors, M, d, exponent_scale=None, **kwargs: candidates,
    )
    monkeypatch.setattr(postprocess, "is_relation", lambda e, bases, N: True)

    def record_relation(e, primes, N):
        checked.append(tuple(e))
        return None

    monkeypatch.setattr(postprocess, "relation_to_congruence", record_relation)

    assert regev_factor([(0,)], 2, 1, 3, [1], [1], verbose=False) is None
    assert checked == [(1,), (2,)]
