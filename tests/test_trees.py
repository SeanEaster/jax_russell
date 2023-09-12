"""Tests for tree models."""
from typing import Any

import pytest
from jax import numpy as jnp

from jax_russell import base, trees

tree_classes = [trees.CRRBinomialTree, trees.RendlemanBartterBinomialTree]
option_types = ["european", "american"]
mixin_classes = [
    base.StockOptionMixin,
    base.FuturesOptionMixin,
    base.AsayMargineduturesOptionMixin,
    base.StockOptionContinuousDividendMixin,
]


haug_start_price = jnp.array([100.0])
haug_volatility = jnp.array([0.3])
haug_time_to_expiration = jnp.array([0.5])
haug_is_call = jnp.array([0.0])
haug_strike = jnp.array([95.0])
haug_risk_free_rate = jnp.array([0.08])
haug_cost_of_carry = jnp.array([0.08])
haug_inputs = (
    haug_start_price,
    haug_volatility,
    haug_time_to_expiration,
    haug_is_call,
    haug_strike,
    haug_risk_free_rate,
    haug_cost_of_carry,
)


mixin_call_args = [
    (haug_start_price, haug_volatility, haug_time_to_expiration, haug_is_call, haug_strike, haug_risk_free_rate),
    (haug_start_price, haug_volatility, haug_time_to_expiration, haug_is_call, haug_strike, haug_risk_free_rate),
    (haug_start_price, haug_volatility, haug_time_to_expiration, haug_is_call, haug_strike),
    (
        haug_start_price,
        haug_volatility,
        haug_time_to_expiration,
        haug_is_call,
        haug_strike,
        haug_risk_free_rate,
        haug_risk_free_rate + 0.02,
    ),
]

time_steps = trees.calc_time_steps(
    1e-3,
    trees.CRRBinomialTree,
    ("american",),
    (
        jnp.array([100.5]),
        jnp.array([0.3]),
        jnp.array([0.5]),
        jnp.array([0.0]),
        jnp.array([95.0]),
        jnp.array([0.08]),
        jnp.array([0.08]),
    ),
)


num_samples = 4
num_assets = 2


starts = 5
triu_indices = jnp.triu_indices(
    starts + 1,
    starts + 1,
)

haug_crr_full_sample = jnp.array(
    [
        [
            100.00000000,
            109.95141602,
            120.89313507,
            132.92370605,
            146.15148926,
            160.69563293,
        ],
        [
            0.00000000,
            90.94926453,
            100.00000000,
            109.95141602,
            120.89312744,
            132.92370605,
        ],
        [
            0.00000000,
            0.00000000,
            82.71768188,
            90.94926453,
            100.00000000,
            109.95140839,
        ],
        [
            0.00000000,
            0.00000000,
            0.00000000,
            75.23111725,
            82.71768188,
            90.94925690,
        ],
        [
            0.00000000,
            0.00000000,
            0.00000000,
            0.00000000,
            68.42214966,
            75.23112488,
        ],
        [
            0.00000000,
            0.00000000,
            0.00000000,
            0.00000000,
            0.00000000,
            62.22943497,
        ],
    ]
)

haug_crr_full_exc = jnp.array(
    [
        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 4.05073547, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 12.28231812, 4.05073547, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000, 19.76888275, 12.28231812, 4.05074310],
        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 26.57785034, 19.76887512],
        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 32.77056503],
    ]
)

haug_crr_full_values = jnp.array(
    [
        [4.91921711, 2.01902986, 0.44127661, 0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 8.12521267, 3.75218487, 0.92395788, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 12.97115803, 6.86119699, 1.93461001, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000, 19.76888275, 12.28231812, 4.05074310],
        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 26.57785034, 19.76887512],
        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 32.77056503],
    ]
)


def test_haug():
    """Test American tree against textbook example."""
    actual = trees.CRRBinomialTree(5, "american")(*haug_inputs)
    jnp.allclose(actual, haug_crr_full_values[0, 0])


@pytest.mark.parametrize("tree_class", tree_classes)
@pytest.mark.parametrize("option_type", option_types)
def test_call(tree_class: Any, option_type: str):
    """Test instantiation and call for all tree classes and option types.

    Args:
        tree_class (Any): _description_
        option_type (str): _description_
    """
    tree_class(5, option_type)(*haug_inputs)


@pytest.mark.parametrize("tree_class", tree_classes)
@pytest.mark.parametrize("option_type", option_types)
@pytest.mark.parametrize("mixin_class,mixin_call_args", zip(mixin_classes, mixin_call_args))
def test_mixins_call(
    tree_class,
    option_type,
    mixin_class,
    mixin_call_args,
):
    """Test instantiation and call for all tree classes, option types and securuity mixins.

    Args:
        tree_class (trees.CRRBinomialTree): A CRRBinomialTree or child
        option_type (str): one of 'american' or 'european'
        mixin_class (Callable): a mixin class that implements __call__() for the tree
        mixin_call_args (Tuple[Any]): args to pass tree.__call__()
    """

    class UnderTest(mixin_class, tree_class):
        pass

    UnderTest(5, option_type)(*mixin_call_args)


@pytest.mark.parametrize("tree_class", tree_classes)
@pytest.mark.parametrize("option_type", option_types)
@pytest.mark.parametrize("mixin_class,mixin_call_args", zip(mixin_classes, mixin_call_args))
def test_mixins_first_order(
    tree_class,
    option_type,
    mixin_class,
    mixin_call_args,
):
    """Test instantiation and first_order() for all tree classes, option types and securuity mixins.

    Args:
        tree_class (trees.CRRBinomialTree): A CRRBinomialTree or child
        option_type (str): one of 'american' or 'european'
        mixin_class (Callable): a mixin class that implements __call__() for the tree
        mixin_call_args (Tuple[Any]): args to pass tree.__call__()
    """

    class UnderTest(mixin_class, tree_class):
        pass

    UnderTest(5, option_type).first_order(*mixin_call_args)


@pytest.mark.parametrize("tree_class", tree_classes)
@pytest.mark.parametrize("option_type", option_types)
@pytest.mark.parametrize("mixin_class,mixin_call_args", zip(mixin_classes, mixin_call_args))
def test_mixins_second_order(
    tree_class,
    option_type,
    mixin_class,
    mixin_call_args,
):
    """Test instantiation and call for all tree classes, option types and securuity mixins.

    Args:
        tree_class (trees.CRRBinomialTree): A CRRBinomialTree or child
        option_type (str): one of 'american' or 'european'
        mixin_class (Callable): a mixin class that implements __call__() for the tree
        mixin_call_args (Tuple[Any]): args to pass tree.__call__()
    """

    class UnderTest(mixin_class, tree_class):
        pass

    UnderTest(5, option_type).second_order(*mixin_call_args)


@pytest.mark.parametrize("tree_class", tree_classes)
@pytest.mark.parametrize("option_type", option_types)
@pytest.mark.parametrize("mixin_class,mixin_call_args", zip(mixin_classes, mixin_call_args))
@pytest.mark.parametrize(
    "valuer_class,valuer_args",
    [
        (trees.SoftplusValuer, (2.5e-2,)),
    ],
)
def test_mixins_valuers_second_order(
    tree_class: trees.CRRBinomialTree,
    option_type: str,
    mixin_class,
    mixin_call_args,
    valuer_class,
    valuer_args,
):
    """Using non-default valuer, test instantiation and call for all tree classes, option types and security mixins.

    Args:
        tree_class (trees.CRRBinomialTree): _description_
        option_type (str): _description_
        mixin_class (Callable): _description_
        mixin_call_args (Tuple[Any]): _description_
        valuer_class (_type_): _description_
        valuer_args (_type_): _description_
    """

    class UnderTest(mixin_class, tree_class):  # type: ignore
        pass

    steps = 5
    UnderTest(
        steps,
        option_type,
        trees.AmericanDiscounter(steps, valuer_class(*valuer_args)) if option_type == "american" else None,
    ).second_order(*mixin_call_args)
