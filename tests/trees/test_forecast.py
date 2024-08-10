"""Test forecast functions for implied trees."""

import jax
from jax import numpy as jnp

from jax_russell.trees import RubinsteinImpliedBinomialTree

jax.config.update("jax_enable_x64", True)
# dtype = jnp.float32

END_VALUES = jnp.array(
    [
        1.2776,
        1.0851,
        0.9216,
        0.7827,
    ],
)

END_PROBABILITIES = jnp.array(
    [
        0.2,
        0.3,
        0.4,
        0.1,
    ],
)

EXPECTED_RETURN_VALUES = jnp.array(
    [
        [1.0, 1.0961, 1.2023, 1.2776],
        [0.0, 0.9100, 0.9826, 1.0851],
        [0.0, 0.0, 0.8542, 0.9216],
        [0.0, 0.0, 0.0, 0.7827],
    ]
).T

EXPECTED_NODE_PROBABILITIES = jnp.array(
    [
        [1.0, 0.533, 0.3, 0.2],
        [0.0, 0.467, 0.467, 0.3],
        [0.0, 0.0, 0.233, 0.4],
        [0.0, 0.0, 0.0, 0.1],
    ]
).T


def test_forecast():
    """Test full implied tree against published results in Rubinstein (1994)."""
    tree = RubinsteinImpliedBinomialTree(3, "american")
    probabilities, forecasted_returns = tree.forecast_returns(END_PROBABILITIES, END_VALUES)
    assert jnp.allclose(EXPECTED_NODE_PROBABILITIES, probabilities, atol=1e-3)
    assert jnp.allclose(EXPECTED_RETURN_VALUES, forecasted_returns, atol=1e-3)
