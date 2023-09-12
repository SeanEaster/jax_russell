"""Test option value functions against example cases."""
import pytest
from jax import numpy as jnp

from jax_russell import base, trees
from tests.base import option_types

rb_steps = [12, 52, 100]
rb_expected = {
    "american": {
        12: jnp.array([2.26, 10.65, 27.42]),
        52: jnp.array([2.15, 10.75, 27.40]),
        100: jnp.array([2.16, 10.77, 27.38]),
    },
    "european": {
        12: jnp.array([2.19, 10.0, 25.46]),
        52: jnp.array([2.09, 10.2, 25.52]),
        100: jnp.array([2.10, 10.23, 25.50]),
    },
}


@pytest.mark.parametrize("steps", rb_steps)
@pytest.mark.parametrize("option_type", option_types)
def test_rb(steps, option_type):
    """Test tree values against values given in Rendleman Bartter (1979)."""
    rb_start = jnp.array([100.0])
    rb_volatility = jnp.array([0.324])
    rb_time_to_expiration = jnp.array([1.0])
    rb_risk_free_rate = jnp.exp(jnp.array([0.05])) - 1
    rb_is_call = jnp.array([0.0])
    rb_strike = jnp.expand_dims(jnp.array([75.0, 100.0, 125.0]), -1)

    class UnderTest(base.StockOptionMixin, trees.RendlemanBartterBinomialTree):
        pass

    test_class = UnderTest(steps, option_type)
    actual = test_class(
        rb_start,
        rb_volatility,
        rb_time_to_expiration,
        rb_risk_free_rate,
        rb_is_call,
        rb_strike,
    )
    expected = jnp.expand_dims(rb_expected[option_type][steps], -1)
    print(actual, expected)
    assert jnp.allclose(actual, expected, atol=3e-2, rtol=3e-2)
