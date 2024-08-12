"""Test __call__() for all tree classes."""

from typing import Any

import jax
import pytest
from jax import numpy as jnp

from tests.base import haug_inputs, option_types
from tests.trees import forward_tree_classes

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("tree_class", forward_tree_classes)
@pytest.mark.parametrize("option_type", option_types)
def test_call(tree_class: Any, option_type: str):
    """Test instantiation and call for all tree classes and option types.

    Args:
        tree_class (Any): tree class to test
        option_type (str): 'american' or 'european'
    """
    assert jnp.greater(tree_class(5, option_type)(*haug_inputs), 0.0)
