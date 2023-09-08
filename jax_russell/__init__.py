"""Top-level package for jax_russell."""

__author__ = """Sean Easter"""
__email__ = 'sean@easter.ai'
__version__ = '0.1.0'

from jax_russell.trees import CRRBinomialTree, ExerciseValuer, MaxValuer, SoftplusValuer, ValuationModel

__all__ = [
    "CRRBinomialTree",
    "ExerciseValuer",
    "MaxValuer",
    "SoftplusValuer",
    "ValuationModel",
]
