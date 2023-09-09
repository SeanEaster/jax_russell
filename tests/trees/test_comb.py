"""Test combinatorial function."""
from jax_russell import trees


from jax import numpy as jnp


def test_comb():
    """Test combinatorial function."""
    assert jnp.allclose(
        trees.comb(5, 3),
        (5 * 4 * 3 * 2) / ((3 * 2) * 2),
    )
