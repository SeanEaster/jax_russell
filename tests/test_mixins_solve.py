"""Test all valuation classes with all mixins."""

import inspect

import pytest
from jax import numpy as jnp

from jax_russell.base import greeks
from jax_russell.bsm import GeneralizedBlackScholesMerten
from tests import base, trees

implied_args = [
    "volatility",
    "time_to_expiration",
    "risk_free_rate",
    "cost_of_carry",
    "strike",
]

ABSOLUTE_TOLERANCES = {
    "strike": 1e-2,
    "risk_free_rate": 1e-4,
    "cost_of_carry": 1e-4,
    "time_to_expiration": 1.0 / 365.0,
    "volatility": 1e-4,
}


@pytest.mark.parametrize("tree_class", trees.forward_tree_classes)
@pytest.mark.parametrize("option_type", base.option_types)
@pytest.mark.parametrize(
    "decorator,mixin_call_args",
    zip(base.class_decorators, base.mixin_call_args),
)
@pytest.mark.parametrize("implied_arg", implied_args)
def test_mixins_solve(
    tree_class,
    option_type,
    decorator,
    mixin_call_args,
    implied_arg,
):
    """Test solve_implied for all tree classes, option types, securuity mixins and arguments.

    Args:
        tree_class (trees.CRRBinomialTree): A CRRBinomialTree or child
        option_type (str): one of 'american' or 'european'
        mixin_class (Callable): a mixin class that implements __call__() for the tree
        mixin_call_args (Tuple[Any]): args to pass tree.__call__()
        implied_arg: argument to solve for
    """

    class UnderTest(tree_class):
        pass

    UnderTest = greeks(decorator(UnderTest))

    under_test = UnderTest(5, option_type)
    signature = inspect.signature(under_test)
    option_values = under_test(*mixin_call_args)

    arg_names = list(signature.parameters.keys())
    if implied_arg not in arg_names:
        pytest.skip(f"arg {implied_arg} not in call signature for mixed classes {tree_class} and {decorator}")
    call_args = list(mixin_call_args)
    i = arg_names.index(implied_arg)
    expected = call_args.pop(i)
    arg_names.pop(i)
    guess = expected * 1.15
    params, _ = under_test.solve_implied(
        option_values,
        {implied_arg: guess},
        **dict(zip(arg_names, call_args)),
    )

    assert jnp.allclose(
        params[implied_arg],
        expected,
        atol=ABSOLUTE_TOLERANCES.get(implied_arg, 1e-6),
    )


@pytest.mark.parametrize("decorator,mixin_call_args", zip(base.class_decorators, base.mixin_call_args))
@pytest.mark.parametrize("implied_arg", implied_args)
def test_mixins_solve_bsm(
    decorator,
    mixin_call_args,
    implied_arg,
):
    """Test instantiation, call and solve for all tree classes, option types and securuity mixins.

    Args:
        option_type (str): one of 'american' or 'european'
        mixin_class (Callable): a mixin class that implements __call__() for the tree
        mixin_call_args (Tuple[Any]): args to pass tree.__call__()
        implied_arg: argument to solve for
    """

    @greeks
    @decorator
    class UnderTest(GeneralizedBlackScholesMerten):
        pass

    under_test = UnderTest()
    signature = inspect.signature(under_test)
    option_values = under_test(*mixin_call_args)
    arg_names = list(signature.parameters.keys())
    if implied_arg not in arg_names:
        pytest.skip(
            f"arg {implied_arg} not in call signature for mixed classes GeneralizedBlackScholesMerten and {decorator}"
        )

    call_args = list(mixin_call_args)
    i = arg_names.index(implied_arg)
    expected = call_args.pop(i)
    arg_names.pop(i)
    guess = expected * 1.15
    params, _ = under_test.solve_implied(
        option_values,
        {implied_arg: guess},
        **dict(zip(arg_names, call_args)),
    )

    assert jnp.allclose(
        params[implied_arg],
        expected,
        atol=ABSOLUTE_TOLERANCES.get(implied_arg, 1e-6),
        # rtol=1e-5,
    )
