"""Test all valuation classes with all mixins."""

import inspect

import jax
import pytest
from jax import numpy as jnp

import tests.trees.test_forecast as test_forecast
import tests.trees.test_values as test_values
from jax_russell.base import AllArgs, greeks
from jax_russell.bsm import GeneralizedBlackScholesMerten
from jax_russell.trees import RubinsteinImpliedBinomialTree
from tests import base, trees

jax.config.update("jax_enable_x64", True)

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
        decorator (Callable): a decorator  that alters __call__() for the model
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
        decorator (Callable): a decorator  that alters __call__() for the model
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


def test_solve_implied_tree():
    """Test that implied values from calculated price are equivalent."""
    tree = RubinsteinImpliedBinomialTree(
        test_forecast.END_VALUES.shape[0] - 1,
        "american",
    )
    val = tree(
        start_price := jnp.ones(
            1,
        ),
        end_probabilities := test_forecast.END_PROBABILITIES,
        end_underlying_returns := test_forecast.END_VALUES,
        tte := jnp.ones(1),
        rfr := test_values.rb_risk_free_rate,
        is_call := jnp.ones(1),
        strike := jnp.ones(1),
    )
    implied = tree.solve_implied(
        val,
        {AllArgs.end_probabilities.value: end_probabilities},
        **{
            AllArgs.start_price.value: start_price,
            AllArgs.end_underlying_returns.value: end_underlying_returns,
            AllArgs.time_to_expiration.value: tte,
            AllArgs.risk_free_rate.value: rfr,
            AllArgs.time_to_expiration.value: tte,
            "is_call": is_call,
            AllArgs.strike.value: strike,
        },
    )

    assert jnp.allclose(
        val,
        tree(
            start_price,
            implied,
            end_underlying_returns,
            tte,
            rfr,
            is_call,
            strike,
        ),
        atol=1e-2,
    )


def _expand_for_broadcasting(*args):
    return tuple(
        jnp.expand_dims(
            arr,
            list(range(idx)) + list(range(-len(args) + 1 + idx, 0)),
        )
        for idx, arr in enumerate(args)
    )


def test_feasible_values(qqq_fitted_values, qqq_bid_ask):
    """Test that an implied tree can derive one set of implied probabilities from a set of options."""
    assert jnp.all(qqq_bid_ask[0] < qqq_fitted_values) and jnp.all(qqq_fitted_values <= qqq_bid_ask[1])


def test_feasible_rate(qqq_returns_fitted_probs, qqq_risk_free_rate):
    """Test that returned probability distribution is implied-rate-feasible."""
    assert jnp.allclose(jnp.power(jnp.dot(*qqq_returns_fitted_probs), 12), jnp.exp(qqq_risk_free_rate), atol=1e-5)
