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


def expand_args_for_broadcasting(
    expand_start_price,
    expand_volatility,
    expand_time_to_expiration,
    expand_risk_free_rate,
    expand_cost_of_carry,
    expand_is_call,
    expand_strike,
    min_total_dims,
):
    """Expand Haug case inputs to test broadcasting behavior.

    Returns:
        tuple: arguments with added dimenstion, expected return shape
    """
    expanded_inputs = []
    whether_to_expand = [
        expand_start_price,
        expand_volatility,
        expand_time_to_expiration,
        expand_risk_free_rate,
        expand_cost_of_carry,
        expand_is_call,
        expand_strike,
    ]

    total_dims = sum(whether_to_expand)
    if total_dims < min_total_dims:
        total_dims = min_total_dims

    adjusted_idx = 0
    expected_shape = []
    for idx, (expand, haug_input) in enumerate(zip(whether_to_expand, haug_inputs)):
        if not expand:
            expanded_inputs.append(haug_input)
            continue
        if idx == 5:
            expanded_dim_len = 2
            expanded_shape = [2 if adjusted_idx == i else 1 for i in range(total_dims)]
            expanded_input = jnp.arange(2, dtype=jnp.float32)
        else:
            expanded_dim_len = idx + 2
            expanded_shape = [expanded_dim_len if adjusted_idx == i else 1 for i in range(total_dims)]
            expanded_input = haug_input + jnp.linspace(0, 5e-1, expanded_dim_len)
        expanded_input = expanded_input.reshape(expanded_shape)
        expanded_inputs.append(expanded_input)

        expected_shape.append(expanded_dim_len)
        adjusted_idx += 1
    if len(expected_shape) < min_total_dims:
        expected_shape.append(1)
    return expanded_inputs, expected_shape


BOOL_LIST = [False, True]
