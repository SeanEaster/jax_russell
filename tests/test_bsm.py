"""Value tests for BSM models."""

import jax.numpy as jnp

from jax_russell import StockOptionBSM

haug_bsm_start_price = jnp.array([60.0])
haug_bsm_volatility = jnp.array([0.3])
haug_bsm_time_to_expiration = jnp.array([0.25])
haug_bsm_is_call = jnp.array([1.0])
haug_bsm_strike = jnp.array([65.0])
haug_bsm_risk_free_rate = jnp.array([0.08])
haug_bsm_cost_of_carry = jnp.array([0.08])
haug_bsm_inputs = (
    haug_bsm_start_price,
    haug_bsm_volatility,
    haug_bsm_time_to_expiration,
    haug_bsm_risk_free_rate,
    haug_bsm_cost_of_carry,
    haug_bsm_is_call,
    haug_bsm_strike,
)

haug_bsm_expected = jnp.array([2.1334])


def test_bsm():
    """Test against example given in Haug, second edition page 3."""
    actual = StockOptionBSM()(
        haug_bsm_start_price,
        haug_bsm_volatility,
        haug_bsm_time_to_expiration,
        haug_bsm_risk_free_rate,
        haug_bsm_is_call,
        haug_bsm_strike,
    )
    assert jnp.allclose(actual, haug_bsm_expected, atol=1e-4)
