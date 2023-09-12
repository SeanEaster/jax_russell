"""Test combinatorial function."""
from jax import numpy as jnp

from jax_russell import trees


def test_comb():
    """Test combinatorial function."""
    assert jnp.allclose(
        trees.comb(5, 3),
        (5 * 4 * 3 * 2) / ((3 * 2) * 2),
    )
