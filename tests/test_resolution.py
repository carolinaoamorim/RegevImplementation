"""Tests for order computation, the resolution guard, and 'resolved' sizing.

The resolution guard is the same species as the contamination and cherry-
picking guards: it flags an experimental setup that would report a success
without demonstrating one. Here the failure is a register modulus M too small
to resolve a base's multiplicative order -- an aliasing limit -- which lets the
random control succeed as often as the quantum arm (e.g. N = 221 at M = 8).
"""

import pytest

from regev.bases import (
    generate_regev_bases,
    multiplicative_order,
    resolution_report,
)
from regev.parameters import regev_parameters


# --- multiplicative_order ----------------------------------------------

@pytest.mark.parametrize("a,N,expected", [
    (4, 77, 15), (2, 7, 3), (3, 7, 6), (10, 77, 6), (2, 15, 4), (1, 77, 1),
])
def test_multiplicative_order_values(a, N, expected):
    assert multiplicative_order(a, N) == expected


def test_multiplicative_order_definition():
    """a ** order == 1, and no smaller positive power does."""
    for a, N in ((4, 77), (9, 143), (25, 221)):
        o = multiplicative_order(a, N)
        assert pow(a, o, N) == 1
        assert all(pow(a, k, N) != 1 for k in range(1, o))


def test_multiplicative_order_requires_coprime():
    with pytest.raises(ValueError):
        multiplicative_order(7, 77)  # gcd(7, 77) = 7


# --- resolution_report --------------------------------------------------

def test_resolution_flags_undersized_M():
    """N = 221, M = 8: an order (24) exceeds M, so the instance is degenerate.

    This is the setup where the uniform-random control factors N ~83% of the
    time -- indistinguishable from the quantum arm, i.e. a vacuous experiment.
    """
    bases, _, _ = generate_regev_bases(221, 3)
    rep = resolution_report(bases, 221, 8)
    assert rep["resolved"] is False
    assert rep["max_order"] >= 8
    assert rep["ratio"] < 1


def test_resolution_passes_when_M_large():
    bases, _, _ = generate_regev_bases(221, 3)
    rep = resolution_report(bases, 221, 32)
    assert rep["resolved"] is True
    assert rep["ratio"] > 1


def test_resolution_boundary_is_strict():
    """M must strictly exceed the order; M == order still aliases."""
    bases, _, _ = generate_regev_bases(77, 3)  # all orders 15
    assert resolution_report(bases, 77, 15)["resolved"] is False
    assert resolution_report(bases, 77, 16)["resolved"] is True


def test_default_formula_can_be_under_resolved():
    """Documenting the deficiency: 'notebook' under-resolves several clean N.

    Not a bug in a single N -- a property of the Regev-asymptotic formula at
    simulable sizes. Locked in so the README's claim stays honest.
    """
    under = [N for N in (187, 209, 299, 323, 391, 437)
             if regev_parameters(N, "notebook")["resolved"] is False]
    assert under == [187, 209, 299, 323, 391, 437]


# --- 'resolved' mode ----------------------------------------------------

@pytest.mark.parametrize("N", [187, 209, 299, 323, 391, 437, 527])
def test_resolved_mode_is_resolved(N):
    params = regev_parameters(N, "resolved")
    assert params["resolved"] is True
    bases, _, _ = generate_regev_bases(N, params["d"])
    assert resolution_report(bases, N, params["M"])["resolved"]


def test_resolved_never_smaller_than_notebook():
    """'resolved' only ever enlarges nd; it must not shrink registers."""
    for N in (77, 143, 187, 299, 437, 527):
        assert (regev_parameters(N, "resolved")["nd"]
                >= regev_parameters(N, "notebook")["nd"])


def test_resolved_leaves_already_resolved_alone():
    """N = 77 is resolved at the notebook size, so 'resolved' changes nothing."""
    assert (regev_parameters(77, "resolved")["nd"]
            == regev_parameters(77, "notebook")["nd"])


def test_parameters_carry_resolution_flag():
    assert regev_parameters(77)["resolved"] is True
    assert regev_parameters(437)["resolved"] is False


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        regev_parameters(77, mode="nonsense")
