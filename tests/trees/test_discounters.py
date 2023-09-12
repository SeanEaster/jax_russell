"""Test Discounter classes for shapes and values."""
from jax import numpy as jnp

from jax_russell.trees import AmericanDiscounter, EuropeanDiscounter

RB_FOUR_STEP_FINAL = jnp.array([190.61, 137.89, 99.75, 72.16, 52.20])
RB_FOUR_STEP_EXPECTED = jnp.array(14.41)

RB_PRICE = jnp.array([100.0])
RB_TTE = jnp.array([1.0])
RB_RISK_FREE_RATE = jnp.array([0.05])
RB_IS_CALL = jnp.array([1.0])
RB_END_PROBABILITIES = jnp.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def test_european_discounter():
    """Test EuropeanDiscounter against Rendleman Bartter (1979) example."""
    actual = EuropeanDiscounter()(
        RB_FOUR_STEP_FINAL,
        RB_PRICE,
        RB_TTE,
        RB_RISK_FREE_RATE,
        RB_IS_CALL,
        RB_END_PROBABILITIES,
    )
    assert jnp.allclose(
        actual,
        RB_FOUR_STEP_EXPECTED,
        atol=1e-2,
        rtol=1e-2,
    )


def test_european_discounter_expanded():
    """Test same example against expanded input."""
    args = expand_args()

    actual = EuropeanDiscounter()(
        RB_FOUR_STEP_FINAL,
        *args,
        RB_END_PROBABILITIES,
    )
    assert jnp.allclose(
        actual,
        RB_FOUR_STEP_EXPECTED,
        atol=1e-2,
        rtol=1e-2,
    )
    assert actual.shape == RB_FOUR_STEP_EXPECTED.shape + (1,)


def expand_args():
    """Expand dimensions of Rendleman Bartter inputs."""
    args = tuple(
        [
            jnp.expand_dims(_, -1)
            for _ in [
                RB_PRICE,
                RB_TTE,
                RB_RISK_FREE_RATE,
                RB_IS_CALL,
            ]
        ]
    )
    args = jnp.broadcast_arrays(*args)
    return args


def test_american_discounter_expanded():
    """Test EuropeanDiscounter against Rendleman Bartter (1979) example."""
    args = expand_args()

    actual = AmericanDiscounter(steps=4)(
        RB_FOUR_STEP_FINAL,
        *args,
        jnp.power(jnp.array([0.5]), 4),
        jnp.array([1.175]),
    )
    assert actual.shape == RB_FOUR_STEP_EXPECTED.shape + (1,)


def test_american_discounter():
    """Test EuropeanDiscounter against Rendleman Bartter (1979) example."""
    actual = AmericanDiscounter(steps=4)(
        RB_FOUR_STEP_FINAL,
        RB_PRICE,
        RB_TTE,
        RB_RISK_FREE_RATE,
        RB_IS_CALL,
        jnp.power(jnp.array([0.5]), 4),
        jnp.array([1.175]),
    )
    assert actual.shape == RB_FOUR_STEP_EXPECTED.shape


def test_shapes_match():
    """Test that American, European discounters return same shapes for same inputs."""
    american_val = AmericanDiscounter(steps=4)(
        RB_FOUR_STEP_FINAL,
        RB_PRICE,
        RB_TTE,
        RB_RISK_FREE_RATE,
        RB_IS_CALL,
        jnp.power(jnp.array([0.5]), 4),
        jnp.array([1.175]),
    )

    european_val = EuropeanDiscounter()(
        RB_FOUR_STEP_FINAL,
        RB_PRICE,
        RB_TTE,
        RB_RISK_FREE_RATE,
        RB_IS_CALL,
        RB_END_PROBABILITIES,
    )

    assert american_val.shape == european_val.shape
