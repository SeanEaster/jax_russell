"""Test time step calculator."""
from jax import numpy as jnp

from jax_russell import trees


def test_calc_time_steps():
    """Test calc_time_steps."""
    trees.calc_time_steps(
        1e-3,
        trees.CRRBinomialTree,
        ("american",),
        (
            jnp.array([100.5]),
            jnp.array([0.3]),
            jnp.array([0.5]),
            jnp.array([0.0]),
            jnp.array([95.0]),
            jnp.array([0.08]),
            jnp.array([0.08]),
        ),
    )
