"""Top-level package for jax_russell."""

__author__ = """Sean Easter"""
__email__ = 'sean@easter.ai'
__version__ = '0.2.0'


from jax_russell.base import ValuationModel, stock_option_cls, stock_option_continuous_dividend_cls
from jax_russell.bsm import GeneralizedBlackScholesMerten
from jax_russell.trees import CRRBinomialTree, ExerciseValuer, MaxValuer, RendlemanBartterBinomialTree, SoftplusValuer


@stock_option_cls
class StockOptionCRRTree(CRRBinomialTree):  # type: ignore[misc]
    """Stock option CRR tree."""


@stock_option_continuous_dividend_cls
class StockOptionContinuousDividendCRRTree(CRRBinomialTree):  # type: ignore[misc]
    """Stock option CRR tree with a continuous dividend."""


@stock_option_cls
class StockOptionRBTree(RendlemanBartterBinomialTree):  # type: ignore[misc]
    """Stock option Rendleman Bartter tree."""


@stock_option_continuous_dividend_cls
class StockOptionContinuousDividendRBTree(RendlemanBartterBinomialTree):  # type: ignore[misc]
    """Stock option Rendleman Bartter tree with a continuous dividend."""


class StockOptionBSM(GeneralizedBlackScholesMerten):  # type: ignore[misc]
    """Stock option Black Scholes Merten valuation."""


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
