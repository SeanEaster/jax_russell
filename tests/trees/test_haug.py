"""Test against Haug example."""
import pytest
from jax_russell import trees
from tests.base import haug_inputs
from tests.base import haug_crr_full_values


from jax import numpy as jnp


def test_haug():
    """Test American tree against textbook example."""
    actual = trees.CRRBinomialTree(5, "american")(*haug_inputs)
    assert jnp.allclose(actual, haug_crr_full_values[0, 0])


bool_list = [False, True]


@pytest.mark.parametrize("expand_start_price", bool_list)
@pytest.mark.parametrize("expand_volatility", bool_list)
@pytest.mark.parametrize("expand_time_to_expiration", bool_list)
@pytest.mark.parametrize("expand_risk_free_rate", bool_list)
@pytest.mark.parametrize("expand_cost_of_carry", bool_list)
@pytest.mark.parametrize("expand_is_call", bool_list)
@pytest.mark.parametrize("expand_strike", bool_list)
@pytest.mark.parametrize("min_total_dims", [1, 2])
def test_haug_broadcasted(
    expand_start_price,
    expand_volatility,
    expand_time_to_expiration,
    expand_risk_free_rate,
    expand_cost_of_carry,
    expand_is_call,
    expand_strike,
    min_total_dims,
):
    """Test broadcasting against single known example."""
    expanded_inputs, expected_shape = expand_args_for_broadcasting(
        expand_start_price,
        expand_volatility,
        expand_time_to_expiration,
        expand_risk_free_rate,
        expand_cost_of_carry,
        expand_is_call,
        expand_strike,
        min_total_dims,
    )

    actual = trees.CRRBinomialTree(5, "american")(*expanded_inputs)
    assert actual.shape == tuple(expected_shape)


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
            expanded_input = jnp.arange(2)
        else:
            expanded_dim_len = idx + 2
            expanded_shape = [expanded_dim_len if adjusted_idx == i else 1 for i in range(total_dims)]
            expanded_input = haug_input + jnp.linspace(0, 5e-1, expanded_dim_len)
        expanded_input = expanded_input.reshape(expanded_shape)
        expanded_inputs.append(expanded_input)

        expected_shape.append(expanded_dim_len)
        adjusted_idx += 1
    if len(expected_shape) == 0:
        expected_shape.append(1)
    return expanded_inputs, expected_shape
