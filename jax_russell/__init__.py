"""Top-level package for jax_russell."""

__author__ = """Sean Easter"""
__email__ = 'sean@easter.ai'
__version__ = '0.2.0'


from jax_russell.base import StockOptionContinuousDividendMixin, StockOptionMixin, ValuationModel
from jax_russell.bsm import GeneralizedBlackScholesMerten
from jax_russell.trees import CRRBinomialTree, ExerciseValuer, MaxValuer, RendlemanBartterBinomialTree, SoftplusValuer


class StockOptionCRRTree(StockOptionMixin, CRRBinomialTree):  # type: ignore[misc]
    """Stock option CRR tree."""

    __doc__ += "" if StockOptionMixin.__doc__ is None else StockOptionMixin.__doc__


class StockOptionContinuousDividendCRRTree(StockOptionContinuousDividendMixin, CRRBinomialTree):  # type: ignore[misc]
    """Stock option CRR tree with a continuous dividend."""

    __doc__ += "" if StockOptionContinuousDividendMixin.__doc__ is None else StockOptionContinuousDividendMixin.__doc__


class StockOptionRBTree(StockOptionMixin, RendlemanBartterBinomialTree):  # type: ignore[misc]
    """Stock option Rendleman Bartter tree."""

    __doc__ += "" if StockOptionMixin.__doc__ is None else StockOptionMixin.__doc__


class StockOptionContinuousDividendRBTree(  # type: ignore[misc]
    StockOptionContinuousDividendMixin,
    RendlemanBartterBinomialTree,
):
    """Stock option Rendleman Bartter tree with a continuous dividend."""

    __doc__ += "" if StockOptionContinuousDividendMixin.__doc__ is None else StockOptionContinuousDividendMixin.__doc__


class StockOptionBSM(StockOptionMixin, GeneralizedBlackScholesMerten):  # type: ignore[misc]
    """Stock option Black Scholes Merten valuation."""

    __doc__ += "" if StockOptionMixin.__doc__ is None else StockOptionMixin.__doc__


__all__ = [
    "ExerciseValuer",
    "MaxValuer",
    "CRRBinomialTree",
    "SoftplusValuer",
    "ValuationModel",
    "StockOptionCRRTree",
    "StockOptionContinuousDividendCRRTree",
    "StockOptionRBTree",
    "StockOptionContinuousDividendRBTree",
]
