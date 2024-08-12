import jax
import pytest
from jax import numpy as jnp

import tests.trees.test_values as test_values
from jax_russell import StockOptionCRRTree, StockOptionRBTree
from jax_russell.base import AllArgs
from jax_russell.trees import RubinsteinImpliedBinomialTree
from tests.test_mixins_solve import expand_for_broadcasting

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def qqq_volatility():
    volatility = jnp.array(
        [
            0.4571,
            0.4367,
            0.4177,
            0.4037,
            0.3859,
            0.3691,
            0.3546,
            0.3409,
            0.3269,
            0.3120,
            0.3004,
            0.2882,
            0.2777,
        ]
    )
    return jnp.log(volatility + 1.0)


@pytest.fixture
def qqq_solve_implied_kwargs(
    qqq_start_price,
    qqq_returns_fitted_probs,
    qqq_time_to_expiration,
    qqq_risk_free_rate,
    qqq_is_call_expanded,
    qqq_strike_expanded,
):

    returns, _ = qqq_returns_fitted_probs
    return {
        AllArgs.start_price.value: qqq_start_price,
        AllArgs.end_underlying_returns.value: returns,
        AllArgs.time_to_expiration.value: qqq_time_to_expiration,
        AllArgs.risk_free_rate.value: qqq_risk_free_rate,
        AllArgs.is_call.value: qqq_is_call_expanded,
        AllArgs.strike.value: qqq_strike_expanded,
        AllArgs.cost_of_carry.value: qqq_risk_free_rate,
    }


@pytest.fixture
def qqq_start_price():
    start_price = jnp.array([440.0])
    return start_price


@pytest.fixture
def rb_end_nodes():
    steps = 100
    rb_tree = StockOptionRBTree(steps, "american")
    return rb_tree._calc_end_nodes(
        test_values.rb_volatility,
        test_values.rb_time_to_expiration,
        test_values.rb_risk_free_rate,
    )


@pytest.fixture
def rb_end_nodes_expanded(rb_end_nodes):
    return jax.tree.map(lambda x: jnp.expand_dims(x, [1, 2]), rb_end_nodes)


@pytest.fixture
def rb_risk_free_rate_expanded():
    return jnp.exp(test_values.rb_risk_free_rate) * jnp.ones([1, 1, 1])


@pytest.fixture
def qqq_fitted_values(
    qqq_start_price,
    qqq_time_to_expiration,
    qqq_risk_free_rate,
    qqq_implied_tree,
    qqq_is_call_expanded,
    qqq_strike_expanded,
    qqq_returns_fitted_probs,
):

    returns, fitted_probs = qqq_returns_fitted_probs

    fitted_values = qqq_implied_tree(
        qqq_start_price,
        fitted_probs,
        returns,
        qqq_time_to_expiration,
        qqq_risk_free_rate,
        qqq_is_call_expanded,
        qqq_strike_expanded,
    )

    return fitted_values


@pytest.fixture
def qqq_returns_fitted_probs(
    qqq_fitted_volatility,
    qqq_start_price,
    qqq_values,
    qqq_time_to_expiration,
    qqq_risk_free_rate,
    qqq_base_tree,
    qqq_implied_tree,
    qqq_is_call_expanded,
    qqq_strike_expanded,
):
    init_probs, returns = qqq_base_tree._calc_end_nodes(
        qqq_fitted_volatility,
        qqq_time_to_expiration,
        qqq_risk_free_rate,
    )

    returns = returns[:, 0]
    init_probs = init_probs[:, 0]
    fitted_probs = qqq_implied_tree._solve_implied_probabilities(
        qqq_values,
        init_probs,
        barrier_const=1,
        probability_threshold=1e-16,
        time_to_expiration=qqq_time_to_expiration,
        is_call=qqq_is_call_expanded,
        risk_free_rate=qqq_risk_free_rate,
        strike=qqq_strike_expanded,
        start_price=qqq_start_price,
        end_underlying_returns=returns,
    )

    return returns, fitted_probs


@pytest.fixture
def qqq_implied_probs(
    qqq_fitted_volatility,
    qqq_start_price,
    qqq_values,
    qqq_time_to_expiration,
    qqq_risk_free_rate,
    qqq_base_tree,
    qqq_implied_tree,
    qqq_is_call_expanded,
    qqq_strike_expanded,
):
    init_probs, returns = qqq_base_tree._calc_end_nodes(
        qqq_fitted_volatility,
        qqq_time_to_expiration,
        qqq_risk_free_rate,
    )

    returns = returns[:, 0]
    init_probs = init_probs[:, 0]
    fitted_probs = qqq_implied_tree.thing(
        qqq_values,
        {AllArgs.end_probabilities.value: init_probs},
        time_to_expiration=qqq_time_to_expiration,
        is_call=qqq_is_call_expanded,
        risk_free_rate=qqq_risk_free_rate,
        strike=qqq_strike_expanded,
        start_price=qqq_start_price,
        end_underlying_returns=returns,
    )

    return returns, fitted_probs


@pytest.fixture
def qqq_returns(qqq_returns_fitted_probs):
    return qqq_returns_fitted_probs[0]


@pytest.fixture
def qqq_fitted_probs(qqq_returns_fitted_probs):
    return qqq_returns_fitted_probs[1]


@pytest.fixture
def qqq_is_call_expanded(qqq_is_call_strike_expanded):
    return qqq_is_call_strike_expanded[0]


@pytest.fixture
def qqq_strike_expanded(qqq_is_call_strike_expanded):
    return qqq_is_call_strike_expanded[1]


@pytest.fixture
def qqq_is_call_strike_expanded(qqq_strike, qqq_is_call):
    is_call, strike = expand_for_broadcasting(
        qqq_is_call,
        qqq_strike,
    )

    return is_call, strike


@pytest.fixture
def qqq_fitted_volatility(
    qqq_volatility,
    qqq_start_price,
    qqq_strike,
    qqq_bid_ask,
    qqq_time_to_expiration,
    qqq_risk_free_rate,
    qqq_is_call,
    qqq_base_tree,
):
    volatility = qqq_base_tree.solve_implied(
        qqq_bid_ask.mean(0),
        {AllArgs.volatility.value: qqq_volatility},
        time_to_expiration=qqq_time_to_expiration,
        is_call=qqq_is_call,
        risk_free_rate=qqq_risk_free_rate,
        strike=qqq_strike,
        start_price=qqq_start_price,
    ).params[AllArgs.volatility.value]

    return volatility


@pytest.fixture
def qqq_steps():
    return 251


@pytest.fixture
def qqq_tree_type():
    return "american"


@pytest.fixture
def qqq_base_tree(qqq_steps, qqq_tree_type):

    return StockOptionCRRTree(qqq_steps, qqq_tree_type)


@pytest.fixture
def qqq_implied_tree(qqq_steps, qqq_tree_type):
    return RubinsteinImpliedBinomialTree(qqq_steps, qqq_tree_type)


@pytest.fixture
def qqq_is_call():
    is_call = jnp.array(1.0)
    return is_call


@pytest.fixture
def qqq_risk_free_rate():
    risk_free_rate = jnp.log(jnp.array(1.05))
    return risk_free_rate


@pytest.fixture
def qqq_time_to_expiration():
    time_to_expiration = jnp.array(1.0 / 12.0)
    return time_to_expiration


@pytest.fixture
def qqq_values(qqq_bid_ask):
    values = jnp.expand_dims(qqq_bid_ask, 1)

    return values


@pytest.fixture
def qqq_bid_ask():
    return jnp.array(
        [
            [46.07, 46.40],
            [41.76, 41.99],
            [37.50, 37.71],
            [33.52, 33.74],
            [29.53, 29.72],
            [25.70, 25.88],
            [22.16, 22.31],
            [18.84, 18.97],
            [15.69, 15.82],
            [12.77, 12.86],
            [10.25, 10.33],
            [7.94, 8.05],
            [6.06, 6.14],
        ]
    ).T


@pytest.fixture
def qqq_strike():
    strike = jnp.array(
        [
            400.0,
            405.0,
            410.0,
            415.0,
            420.0,
            425.0,
            430.0,
            435.0,
            440.0,
            445.0,
            450.0,
            455.0,
            460.0,
        ]
    )

    return strike
