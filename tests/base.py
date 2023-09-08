"""Components shared across tests."""
from jax import numpy as jnp

import jax_russell.base

option_types = ["european", "american"]

haug_volatility = jnp.array([0.3])
haug_time_to_expiration = jnp.array([0.5])
haug_is_call = jnp.array([0.0])
haug_strike = jnp.array([95.0])
haug_risk_free_rate = jnp.array([0.08])
haug_cost_of_carry = jnp.array([0.08])
haug_start_price = jnp.array([100.0])
haug_inputs = (
    haug_start_price,
    haug_volatility,
    haug_time_to_expiration,
    haug_risk_free_rate,
    haug_cost_of_carry,
    haug_is_call,
    haug_strike,
)
mixin_call_args = [
    (haug_start_price, haug_volatility, haug_time_to_expiration, haug_risk_free_rate, haug_is_call, haug_strike),
    (haug_start_price, haug_volatility, haug_time_to_expiration, haug_risk_free_rate, haug_is_call, haug_strike),
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
mixin_classes = [
    jax_russell.base.StockOptionMixin,
    jax_russell.base.FuturesOptionMixin,
    jax_russell.base.AsayMargineduturesOptionMixin,
    jax_russell.base.StockOptionContinuousDividendMixin,
]
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
