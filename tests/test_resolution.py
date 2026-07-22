"""Tests for generator-order diagnostics and finite-model parameters.

Individual generator orders are useful to inspect, but they are not a
one-dimensional Nyquist condition for Regev's multidimensional lattice.
"""

import math

import pytest

from regev.bases import (
    generate_regev_bases,
    multiplicative_order,
    order_report,
    resolution_report,
)
from regev.parameters import regev_parameters
from regev.postprocess import is_relation


@pytest.mark.parametrize(
    "a,N,expected",
    [(4, 77, 15), (2, 7, 3), (3, 7, 6), (10, 77, 6), (2, 15, 4), (1, 77, 1)],
)
def test_multiplicative_order_values(a, N, expected):
    assert multiplicative_order(a, N) == expected


def test_multiplicative_order_matches_definition():
    for a, N in ((4, 77), (9, 143), (25, 221)):
        order = multiplicative_order(a, N)
        assert pow(a, order, N) == 1
        assert all(pow(a, k, N) != 1 for k in range(1, order))


def test_multiplicative_order_requires_invertible_base():
    with pytest.raises(ValueError, match="not invertible"):
        multiplicative_order(7, 77)


def test_order_report_describes_axis_coverage():
    bases, _, _ = generate_regev_bases(221, 3)
    small = order_report(bases, 221, 8)
    large = order_report(bases, 221, 32)

    assert small["covers_axis_orders"] is False
    assert small["ratio"] < 1
    assert large["covers_axis_orders"] is True
    assert large["ratio"] > 1


def test_axis_coverage_boundary_is_strict():
    bases, _, _ = generate_regev_bases(77, 3)
    assert order_report(bases, 77, 15)["covers_axis_orders"] is False
    assert order_report(bases, 77, 16)["covers_axis_orders"] is True


def test_axis_orders_are_not_a_multidimensional_validity_guard():
    """A short cross-axis relation is visible below either axis order.

    Both generators have order 15 > M, yet (1, -1) is an exact relation. This
    is the simplest counterexample to the old per-register Nyquist claim.
    """
    bases = [4, 4]
    report = order_report(bases, 77, 8)
    assert report["covers_axis_orders"] is False
    assert is_relation([1, -1], bases, 77)
    assert "does not determine" in report["reason"]


def test_resolution_report_is_a_compatibility_alias():
    with pytest.warns(DeprecationWarning, match="order_report"):
        report = resolution_report([4, 9, 25], 77, 32)
    assert report["resolved"] is report["covers_axis_orders"]


def test_default_parameters_are_canonical_regev_mode():
    params = regev_parameters(77)
    assert params["mode"] == "regev"
    assert params["D"] == params["M"]
    assert params["R"] == pytest.approx(params["D"] / (2 * math.sqrt(params["d"])))


@pytest.mark.parametrize("N", [187, 209, 299, 323, 391, 437, 527])
def test_axis_order_mode_covers_individual_orders(N):
    params = regev_parameters(N, "axis_order")
    bases, _, _ = generate_regev_bases(N, params["d"])
    assert order_report(bases, N, params["M"])["covers_axis_orders"]
    assert params["axis_orders_covered"] is True


def test_axis_order_mode_never_shrinks_default_window():
    for N in (77, 143, 187, 299, 437, 527):
        assert regev_parameters(N, "axis_order")["nd"] >= regev_parameters(N)["nd"]


def test_axis_order_mode_leaves_already_covered_instance_unchanged():
    assert regev_parameters(77, "axis_order")["nd"] == regev_parameters(77)["nd"]


def test_default_carries_honestly_named_axis_diagnostic():
    assert regev_parameters(77)["axis_orders_covered"] is True
    assert regev_parameters(437)["axis_orders_covered"] is False


def test_deprecated_parameter_aliases_are_canonicalized():
    with pytest.warns(DeprecationWarning, match="notebook"):
        notebook = regev_parameters(437, "notebook")
    with pytest.warns(DeprecationWarning, match="resolved"):
        resolved = regev_parameters(437, "resolved")

    assert notebook["mode"] == "regev"
    assert resolved["mode"] == "axis_order"


def test_cover_2n_has_enough_exponent_bits():
    params = regev_parameters(77, "cover_2n")
    assert params["total_x_qubits"] >= 2 * params["n"]


def test_bad_mode_and_modulus_are_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        regev_parameters(77, mode="nonsense")
    with pytest.raises(ValueError, match="greater than 1"):
        regev_parameters(1)
