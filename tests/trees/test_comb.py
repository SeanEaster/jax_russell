"""Test combinatorial function."""

import jax
from jax import numpy as jnp

from jax_russell import trees

jax.config.update("jax_enable_x64", True)


def test_comb():
    """Test combinatorial function."""
    assert jnp.allclose(
        trees.comb(5, 3),
        (5 * 4 * 3 * 2) / ((3 * 2) * 2),
    )
