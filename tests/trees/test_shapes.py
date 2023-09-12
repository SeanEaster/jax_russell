"""Test against Haug example."""
import pytest
from jax import numpy as jnp

from tests.base import BOOL_LIST, expand_args_for_broadcasting, option_types
from tests.trees import tree_classes


@pytest.mark.parametrize("expand_start_price", BOOL_LIST)
@pytest.mark.parametrize("expand_volatility", BOOL_LIST)
@pytest.mark.parametrize("expand_time_to_expiration", BOOL_LIST)
@pytest.mark.parametrize("expand_risk_free_rate", BOOL_LIST)
@pytest.mark.parametrize("expand_cost_of_carry", BOOL_LIST)
@pytest.mark.parametrize("expand_is_call", BOOL_LIST)
@pytest.mark.parametrize("expand_strike", BOOL_LIST)
@pytest.mark.parametrize("min_total_dims", [1, 2])
@pytest.mark.parametrize("tree_class", tree_classes)
@pytest.mark.parametrize("option_type", option_types)
def test_expanded(
    expand_start_price,
    expand_volatility,
    expand_time_to_expiration,
    expand_risk_free_rate,
    expand_cost_of_carry,
    expand_is_call,
    expand_strike,
    min_total_dims,
    tree_class,
    option_type,
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

    actual = tree_class(5, option_type)(*expanded_inputs)
    assert actual.shape == tuple(expected_shape)


@pytest.mark.parametrize("expand_start_price", BOOL_LIST)
@pytest.mark.parametrize("expand_volatility", BOOL_LIST)
@pytest.mark.parametrize("expand_time_to_expiration", BOOL_LIST)
@pytest.mark.parametrize("expand_risk_free_rate", BOOL_LIST)
@pytest.mark.parametrize("expand_cost_of_carry", BOOL_LIST)
@pytest.mark.parametrize("expand_is_call", BOOL_LIST)
@pytest.mark.parametrize("expand_strike", BOOL_LIST)
@pytest.mark.parametrize("min_total_dims", [1, 2])
@pytest.mark.parametrize("tree_class", tree_classes)
def test_haug_broadcasted(
    expand_start_price,
    expand_volatility,
    expand_time_to_expiration,
    expand_risk_free_rate,
    expand_cost_of_carry,
    expand_is_call,
    expand_strike,
    min_total_dims,
    tree_class,
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

    print(expected_shape)
    actual = tree_class(5, "american")(*jnp.broadcast_arrays(*expanded_inputs))
    assert actual.shape == tuple(expected_shape)
