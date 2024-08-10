"""Test all valuation classes with all mixins."""

import inspect

import jax
import jaxopt
import pytest
from jax import numpy as jnp

import tests.trees.test_forecast as test_forecast
import tests.trees.test_values as test_values
from jax_russell import StockOptionCRRTree, StockOptionRBTree
from jax_russell.base import AllArgs, greeks
from jax_russell.bsm import GeneralizedBlackScholesMerten
from jax_russell.trees import (
    CRRBinomialTree,
    RubinsteinImpliedBinomialTree,
    _transpose_args_and_return,
    projection_end_probabilities,
    vmap_repeated,
)
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


def test_projection(
    rb_end_nodes_expanded,
    rb_risk_free_rate_expanded,
):
    rb_probs, rb_returns = rb_end_nodes_expanded
    projection = vmap_repeated(
        projection_end_probabilities,
        rb_probs.shape,
        (0, (0, 0, 0)),
        0,
    )
    projection = _transpose_args_and_return(projection)
    actual = projection(
        rb_probs,
        (
            rb_returns,
            rb_risk_free_rate_expanded,
            rb_probs,
        ),
    )
    assert jnp.allclose(rb_probs, actual)


def test_solve_implied_tree():
    tree = RubinsteinImpliedBinomialTree(
        test_forecast.END_VALUES.shape[0] - 1,
        "american",
    )
    val = tree(
        start_price := jnp.ones((1,)),
        end_probabilities := jnp.expand_dims(test_forecast.END_PROBABILITIES, -1),
        end_underlying_returns := jnp.expand_dims(test_forecast.END_VALUES, -1),
        tte := jnp.ones((1,)),
        rfr := test_values.rb_risk_free_rate,
        coc := test_values.rb_risk_free_rate,
        is_call := jnp.ones((1,)),
        strike := jnp.ones((1,)),
    )
    implied = tree.solve_implied(
        val,
        {AllArgs.end_probabilities.value: end_probabilities},
        # {AllArgs.end_probabilities.value: jnp.ones_like(end_probabilities) / end_probabilities.shape[0]},
        **{
            AllArgs.start_price.value: start_price,
            AllArgs.end_underlying_returns.value: end_underlying_returns,
            AllArgs.time_to_expiration.value: tte,
            AllArgs.risk_free_rate.value: rfr,
            AllArgs.time_to_expiration.value: tte,
            AllArgs.cost_of_carry.value: coc,
            "is_call": is_call,
            AllArgs.strike.value: strike,
        },
    )

    assert jnp.allclose(
        val,
        tree(
            start_price,
            implied.params,
            end_underlying_returns,
            tte,
            rfr,
            coc,
            is_call,
            strike,
        ),
        atol=1e-2,
    )


def expand_for_broadcasting(*args):
    return tuple(
        jnp.expand_dims(
            arr,
            list(range(idx)) + list(range(-len(args) + 1 + idx, 0)),
        )
        for idx, arr in enumerate(args)
    )


def test_feasible(qqq_fitted_values, qqq_bid_ask):
    """Test that an implied tree can derive one set of implied probabilities from a set of options."""

    assert jnp.all(qqq_bid_ask[0] < qqq_fitted_values) and jnp.all(qqq_fitted_values <= qqq_bid_ask[1])


@pytest.fixture
def qqq_fitted_values(
    qqq_volatility,
    qqq_start_price,
    qqq_strike,
    qqq_values,
    qqq_bid_ask,
    qqq_time_to_expiration,
    qqq_risk_free_rate,
    qqq_is_call,
):
    steps = 251

    tree_type = "american"
    base_tree = StockOptionCRRTree(steps, tree_type)
    implied_tree = RubinsteinImpliedBinomialTree(steps, tree_type)

    volatility = base_tree.solve_implied(
        qqq_bid_ask.mean(0),
        {AllArgs.volatility.value: qqq_volatility},
        time_to_expiration=qqq_time_to_expiration,
        is_call=qqq_is_call,
        risk_free_rate=qqq_risk_free_rate,
        strike=qqq_strike,
        start_price=qqq_start_price,
    ).params[AllArgs.volatility.value]

    is_call, strike = expand_for_broadcasting(
        qqq_is_call,
        qqq_strike,
    )
    init_probs, returns = base_tree._calc_end_nodes(
        volatility,
        qqq_time_to_expiration,
        qqq_risk_free_rate,
    )

    returns = returns[:, 0]
    init_probs = init_probs[:, 0]
    fitted_probs = implied_tree.feasible_init(
        init_probs,
        qqq_values,
        barrier_const=1,
        time_to_expiration=qqq_time_to_expiration,
        is_call=qqq_is_call,
        risk_free_rate=qqq_risk_free_rate,
        strike=strike,
        start_price=qqq_start_price,
        end_underlying_returns=returns,
        cost_of_carry=qqq_risk_free_rate,
    )

    fitted_values = implied_tree(
        qqq_start_price,
        fitted_probs,
        returns,
        qqq_time_to_expiration,
        qqq_risk_free_rate,
        qqq_risk_free_rate,
        is_call,
        strike,
    )

    return fitted_values


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


@pytest.fixture
def qqq_start_price():
    start_price = jnp.array([440.0])
    return start_price


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


def test_smile():
    """Test that an implied tree can derive one set of implied probabilities from a set of options."""
    steps = 251

    tree_type = "american"
    base_tree = StockOptionCRRTree(steps, tree_type)
    implied_tree = RubinsteinImpliedBinomialTree(steps, tree_type)

    start_price = jnp.array([440.0])
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
    volatility = jnp.log(volatility + 1.0)

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
    values = jnp.expand_dims(
        bid_ask := jnp.array(
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
        ).T,
        1,
    )  # [[46.2239191  41.87999252 37.69999599 33.51999947 29.63356442 25.89370753 22.15385063 18.84000379 15.83015718 12.82031056 10.17905607  8.11952796 6.05999985]]
    time_to_expiration = jnp.array(1.0 / 12.0)
    risk_free_rate = jnp.log(jnp.array(1.05))
    # is_call = jnp.arange(2).astype(jnp.float64)
    is_call = jnp.array(1.0)
    print(volatility)
    volatility = base_tree.solve_implied(
        bid_ask.mean(0),
        {AllArgs.volatility.value: volatility},
        time_to_expiration=time_to_expiration,
        is_call=is_call,
        risk_free_rate=risk_free_rate,
        strike=strike,
        start_price=start_price,
    ).params[AllArgs.volatility.value]

    print("vol, strike:", volatility, "\n", strike)

    is_call, strike = expand_for_broadcasting(
        is_call,
        strike,
    )
    init_probs, returns = base_tree._calc_end_nodes(
        volatility,
        time_to_expiration,
        risk_free_rate,
    )
    # init_probs = init_probs * jnp.array([10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 10])
    # print("RETURNS ", returns)
    shape = init_probs.shape
    init_probs, returns = init_probs.ravel(), returns.ravel()
    init_probs, returns = init_probs[indices := jnp.argsort(returns, descending=True)], returns[indices]
    init_probs, returns = init_probs.reshape(shape), returns.reshape(shape)
    # print("marginals", returns * start_price)

    # init_probs = init_probs.at[47].multiply(3)

    returns = (returns * init_probs).sum(-1) / init_probs.sum(-1)
    init_probs = init_probs.sum(-1)
    init_probs = init_probs / init_probs.sum()
    print("marginals", returns * start_price, init_probs)
    # assert False
    bid, ask = bid_ask[0, :], bid_ask[1, :]
    print(bid, ask)
    assert False

    def init_feasible_obj(log_probs):
        values_hat = implied_tree(
            start_price,
            jax.nn.softmax(log_probs),
            # implied_probabilities,
            returns,
            time_to_expiration,
            risk_free_rate,
            risk_free_rate,
            is_call,
            strike,
        )
        barrier = jnp.exp((jax.nn.relu(values_hat - ask) + jax.nn.relu(bid - values_hat))).mean()

        return barrier + jnp.power(bid_ask - values_hat, 2).mean()

    init_log_probs_fitted = jaxopt.LBFGS(init_feasible_obj).run(jnp.log(init_probs))
    # print((init_probs_fitted))
    print(
        "init feasible vals",
        init_vals := implied_tree(
            start_price,
            init_probs := jax.nn.softmax(init_log_probs_fitted.params),
            # implied_probabilities,
            returns,
            time_to_expiration,
            risk_free_rate,
            risk_free_rate,
            is_call,
            strike,
        ),
    )
    print("init feasible: ", returns * start_price, init_probs)
    assert False

    # print(
    #     "initial values from atm ",
    #     init_vals := implied_tree(
    #         start_price,
    #         init_probs[..., 8],
    #         # implied_probabilities,
    #         returns[..., 8],
    #         time_to_expiration,
    #         risk_free_rate,
    #         risk_free_rate,
    #         is_call,
    #         strike,
    #     ),
    # )
    # init_probs = jnp.stack(
    #     tuple(
    #         jnp.interp(
    #             returns[..., -1],
    #             jnp.flipud(returns[..., _]),
    #             jnp.flipud(init_probs[..., _]),
    #         )
    #         for _ in range(returns.shape[-1])
    #     )
    # ).T

    # print(init_probs.shape, init_probs)
    # _ = 1
    # print(
    #     jnp.flipud(returns[..., -1]),
    #     jnp.flipud(jnp.flipud(returns[..., _])),
    #     jnp.flipud(jnp.flipud(init_probs[..., _])),
    #     jnp.interp(
    #         jnp.flipud(returns[..., 0]),
    #         jnp.flipud(jnp.flipud(returns[..., _])),
    #         jnp.flipud(jnp.flipud(init_probs[..., _])),
    #     ),
    # )

    # init_probs = init_probs.mean(-1)  # [..., 8]
    # returns = returns[..., -1]
    volatility = jnp.broadcast_to(volatility, strike.shape)
    # values = base_tree(
    #     start_price,
    #     volatility,
    #     time_to_expiration,
    #     risk_free_rate,
    #     is_call,
    #     strike,
    # )
    # print(values.shape)
    # print(values)

    # print(
    #     "initial values from probs",
    #     init_vals := implied_tree(
    #         start_price,
    #         init_probs,
    #         # implied_probabilities,
    #         returns,
    #         time_to_expiration,
    #         risk_free_rate,
    #         risk_free_rate,
    #         is_call,
    #         strike,
    #     ),
    # )
    # assert False

    # perc_spread = 0.0064
    # values = jnp.stack((bid := values * (1 - perc_spread), ask := values * (1 + perc_spread)))
    print(
        "init err ",
        jnp.sqrt(jnp.square(values - init_vals).mean()),
        values.shape,
        init_vals.shape,
    )
    # print(values)
    # print(
    #     "SHAPES ",
    #     start_price.shape,
    #     init_probs.shape,
    #     returns.shape,
    #     time_to_expiration.shape,
    #     risk_free_rate.shape,
    #     is_call.shape,
    #     strike.shape,
    # )
    start = 1
    end = start + 1
    # print("val and strike ", values[:, :, start:end], strike[:, start:end])
    implied_probabilities = implied_tree.solve_implied(
        values,
        {AllArgs.end_probabilities.value: init_probs},
        start_price=start_price,
        end_underlying_returns=returns,
        time_to_expiration=time_to_expiration,
        risk_free_rate=risk_free_rate,
        cost_of_carry=risk_free_rate,
        is_call=is_call,
        strike=strike,
        stepsize=1e-6,
        # tol=1e-12,
    )
    # print(init_probs)
    # print(implied_probabilities)
    # print((returns * start_price), init_probs, implied_probabilities)
    print(
        "VALUES: ",
        values,
        "\n\nSOLVED: ",
        solved_vals := implied_tree(
            start_price,
            # jnp.expand_dims(implied_probabilities, list(range(-2, 0))),
            # jax.nn.softmax(implied_probabilities),
            implied_probabilities,
            returns,
            time_to_expiration,
            risk_free_rate,
            risk_free_rate,
            is_call,
            strike,
        ),
    )
    print("end err ", jnp.sqrt(jnp.square(values - solved_vals).mean()), values.shape, solved_vals.shape)
    bid, ask = values[0, 0, :], values[1, 0, :]
    print(bid <= ask)
    print(bid <= solved_vals)
    print(solved_vals <= ask)
    assert False
