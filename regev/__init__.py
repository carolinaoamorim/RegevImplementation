"""Regev's multidimensional factoring algorithm: simulation and analysis."""

__version__ = "0.1.0"

from regev.parameters import regev_parameters, validate_factor_input
from regev.bases import generate_regev_bases, is_prime_basic, is_contaminated
from regev.lattice import lll
from regev.postprocess import (
    regev_relation_lattice,
    regev_factor,
    is_relation,
    eval_relation,
    relation_to_congruence,
)
from regev.reference import (
    ideal_regev_distribution,
    total_variation_distance,
    counts_to_distribution,
    bitstring_to_regev_vector,
)
from regev.sampling import sample_ideal, sample_uniform, ideal_distribution_cached

__all__ = [
    "ideal_regev_distribution",
    "total_variation_distance",
    "counts_to_distribution",
    "bitstring_to_regev_vector",
    "sample_ideal",
    "sample_uniform",
    "ideal_distribution_cached",
    "regev_parameters",
    "validate_factor_input",
    "generate_regev_bases",
    "is_prime_basic",
    "is_contaminated",
    "lll",
    "regev_relation_lattice",
    "regev_factor",
    "is_relation",
    "eval_relation",
    "relation_to_congruence",
]
