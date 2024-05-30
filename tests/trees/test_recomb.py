"""Test recombining functions for implied trees."""

import jax
from jax import numpy as jnp

from jax_russell.trees import back_update_tree, calc_recombining_tree

jax.config.update("jax_enable_x64", True)
dtype = jnp.float32

END_VALUES = jnp.array(
    [
        1.2776,
        1.0851,
        0.9216,
        0.7827,
    ],
    dtype=dtype,
)

END_PROBABILITIES = jnp.array(
    [
        0.2,
        0.3,
        0.4,
        0.1,
    ],
    dtype=dtype,
)

EXPECTED_VALUES = jnp.array(
    [
        [1.0, 1.0961, 1.2023, 1.2776],
        [0.0, 9100, 0.9826, 1.0851],
        [0.0, 0.0, 0.8542, 0.9216],
        [0.0, 0.0, 0.0, 0.7827],
    ]
)

# EXPECTED_NODE_PROBABILITIES = jnp.array(
#     [
#         [],
#         [],
#         [],
#         [],
#     ]
# )


def test_back_update_tree():
    dim = END_VALUES.shape[0]
    in_values = (
        jnp.zeros(
            (dim, dim),
            dtype=dtype,
        )
        .at[..., -1]
        .set(END_VALUES)
    )
    in_probs = (
        jnp.zeros(
            (dim, dim),
            dtype=dtype,
        )
        .at[..., -1]
        .set(END_PROBABILITIES)
    )
    print(
        jnp.array([1.21343333, 0.99167143, 0.86207143])
        / jnp.array(
            [
                1.2023,
                0.9826,
                0.8542,
            ]
        )
    )
    print(in_values, in_probs)
    print(
        jnp.dot(END_PROBABILITIES, END_VALUES),
        jnp.power(jnp.dot(END_PROBABILITIES, END_VALUES), 1.0 / 3.0),
        1.017 / 1.008,
    )
    back_update_tree(
        dim - 1,
        in_probs,
        in_values,
        jnp.power(jnp.dot(END_PROBABILITIES, END_VALUES), 1.0 / 3.0),
    )


def test_calc_recombining_tree():
    actual = calc_recombining_tree(
        END_PROBABILITIES,
        END_VALUES,
        jnp.power(jnp.dot(END_PROBABILITIES, END_VALUES), 1.0 / 3.0),
    )
    for _ in actual:
        print(_)
    assert False
