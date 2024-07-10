"""Test recombining functions for implied trees."""

import jax
from jax import numpy as jnp

from jax_russell.trees import (
    AmericanDiscounter,
    CRRBinomialTree,
    RendlemanBartterBinomialTree,
    back_combine,
    back_update_tree,
    calc_recombining_tree,
)

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

EXPECTED_RETURN_VALUES = jnp.array(
    [
        [1.0, 1.0961, 1.2023, 1.2776],
        [0.0, 0.9100, 0.9826, 1.0851],
        [0.0, 0.0, 0.8542, 0.9216],
        [0.0, 0.0, 0.0, 0.7827],
    ]
)

EXPECTED_NODE_PROBABILITIES = jnp.array(
    [
        [1.0, 0.533, 0.3, 0.2],
        [0.0, 0.467, 0.467, 0.3],
        [0.0, 0.0, 0.233, 0.4],
        [0.0, 0.0, 0.0, 0.1],
    ]
)


# def test_back_update_tree():
#     dim = END_VALUES.shape[0]
#     in_values = (
#         jnp.zeros(
#             (dim, dim),
#             dtype=dtype,
#         )
#         .at[..., -1]
#         .set(END_VALUES)
#     )
#     in_probs = (
#         jnp.zeros(
#             (dim, dim),
#             dtype=dtype,
#         )
#         .at[..., -1]
#         .set(END_PROBABILITIES)
#     )
#     print(
#         jnp.array([1.21343333, 0.99167143, 0.86207143])
#         / jnp.array(
#             [
#                 1.2023,
#                 0.9826,
#                 0.8542,
#             ]
#         )
#     )
#     print(in_values, in_probs)
#     print(
#         jnp.dot(END_PROBABILITIES, END_VALUES),
#         jnp.power(jnp.dot(END_PROBABILITIES, END_VALUES), 1.0 / 3.0),
#         1.017 / 1.008,
#     )
#     back_update_tree(
#         dim - 1,
#         in_probs,
#         in_values,
#         jnp.power(jnp.dot(END_PROBABILITIES, END_VALUES), 1.0 / 3.0),
#     )


# def test_calc_recombining_tree():
#     actual = calc_recombining_tree(
#         END_PROBABILITIES,
#         END_VALUES,
#         jnp.power(jnp.dot(END_PROBABILITIES, END_VALUES), 1.0 / 3.0),
#     )
#     for _ in actual:
#         print(_)
#     print(
#         jnp.power(jnp.dot(END_PROBABILITIES, END_VALUES), 1.0 / 3.0),
#     )
#     assert False


# def test_rubinstein_value():

#     pass


# def test_discounters_equiv():

#     tree = CRRBinomialTree(3, "european")
#     start_price = jnp.array([1.0])
#     u, d = tree._calc_factors(
#         jnp.array([0.1]),
#         tte := jnp.array([1.0]),
#     )
#     end_values = tree._calc_end_values(start_price, u, d)
#     end_probs = tree._calc_end_probabilities(
#         u,
#         d,
#         tte,
#         cost_of_carry := jnp.array([0.1]),
#     )
#     print("shapes: ", end_values.shape, end_probs.shape)
#     print(
#         end_values[0],
#         end_probs[0],
#         jnp.power(jnp.dot(end_probs[0], end_values[0]), 1.0 / 3.0),
#         jnp.log(jnp.dot(end_probs[0], end_values[0])),
#         calc_recombining_tree(
#             end_probs[0],
#             end_values[0],
#             jnp.power(jnp.dot(end_probs[0], end_values[0]), 1.0 / 3.0),
#         ),
#     )
#     discounter = AmericanDiscounter(3)
#     a = discounter.build_next_value_body_function(
#         strike := jnp.array([1.0]),
#         tte := jnp.array([1.0]),
#         cost_of_carry,
#         is_call := jnp.array([1.0]),
#         jnp.array([0.5]),
#         u,
#     )
#     b = discounter.build_next_value_body_function_exp_rubinstein(
#         start_price,
#         end_values,
#         strike,
#         tte,
#         None,
#         is_call,
#         end_probs,
#     )
#     print(
#         "init vals: ",
#         init_vals := discounter.exercise_valuer(
#             end_values,
#             strike,
#             is_call,
#         ),
#         "end: ",
#         end_values,
#         "end_probs: ",
#         end_probs,
#     )
#     print(
#         "a: ",
#         a(
#             0,
#             (
#                 discounter.exercise_valuer(
#                     end_values,
#                     strike,
#                     is_call,
#                 ),
#                 end_values,
#             ),
#         ),
#         "\nb:",
#         b(0, (init_vals, end_values, end_probs)),
#     )

#     assert False


# def test_build_fns():
def test_back_combine():
    actual_probabilities, actual_return_values = EXPECTED_NODE_PROBABILITIES[..., -1], EXPECTED_RETURN_VALUES[..., -1]

    for i in range(EXPECTED_NODE_PROBABILITIES.shape[-1] - 1):
        # for i in range(1):
        num_nodes_start = EXPECTED_NODE_PROBABILITIES.shape[-1] - i
        actual_probabilities, actual_return_values = back_combine(
            i,
            actual_probabilities,
            actual_return_values,
        )
        assert jnp.allclose(
            actual_probabilities,
            EXPECTED_NODE_PROBABILITIES[..., num_nodes_start - 2],
            atol=1e-3,
        )
        assert jnp.allclose(
            actual_return_values,
            EXPECTED_RETURN_VALUES[..., num_nodes_start - 2],
            atol=1e-3,
        )


def test_calc_recombining_tree_exp():
    actual_probabilities, actual_return_values = calc_recombining_tree(END_PROBABILITIES, END_VALUES)
    assert jnp.allclose(
        EXPECTED_NODE_PROBABILITIES,
        actual_probabilities,
        atol=1e-3,
    )
    assert jnp.allclose(
        EXPECTED_RETURN_VALUES,
        actual_return_values,
        atol=1e-3,
    )
