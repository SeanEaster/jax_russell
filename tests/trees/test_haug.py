"""Test against Haug example."""
from jax_russell import trees
from tests.base import haug_inputs
from tests.base import haug_crr_full_values


from jax import numpy as jnp


def test_haug():
    """Test American tree against textbook example."""
    actual = trees.CRRBinomialTree(5, "american")(*haug_inputs)
    assert jnp.allclose(actual, haug_crr_full_values[0, 0])
