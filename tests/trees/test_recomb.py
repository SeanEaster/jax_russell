# """Test recombining functions for implied trees."""

# import jax
# from jax import numpy as jnp

# from jax_russell.trees import (
#     AmericanDiscounter,
#     CRRBinomialTree,
#     RendlemanBartterBinomialTree,
#     back_combine,
#     back_combine_path_probabilities,
#     back_combine_paths_body,
#     back_combine_paths_scan,
#     calc_recombining_tree,
#     comb,
# )

# # jax.config.update("jax_enable_x64", True)
# dtype = jnp.float32

# END_VALUES = jnp.array(
#     [
#         1.2776,
#         1.0851,
#         0.9216,
#         0.7827,
#     ],
#     dtype=dtype,
# )

# END_PROBABILITIES = jnp.array(
#     [
#         0.2,
#         0.3,
#         0.4,
#         0.1,
#     ],
#     dtype=dtype,
# )

# EXPECTED_RETURN_VALUES = jnp.array(
#     [
#         [1.0, 1.0961, 1.2023, 1.2776],
#         [0.0, 0.9100, 0.9826, 1.0851],
#         [0.0, 0.0, 0.8542, 0.9216],
#         [0.0, 0.0, 0.0, 0.7827],
#     ]
# )

# EXPECTED_NODE_PROBABILITIES = jnp.array(
#     [
#         [1.0, 0.533, 0.3, 0.2],
#         [0.0, 0.467, 0.467, 0.3],
#         [0.0, 0.0, 0.233, 0.4],
#         [0.0, 0.0, 0.0, 0.1],
#     ]
# )


# # def test_back_update_tree():
# #     dim = END_VALUES.shape[0]
# #     in_values = (
# #         jnp.zeros(
# #             (dim, dim),
# #             dtype=dtype,
# #         )
# #         .at[..., -1]
# #         .set(END_VALUES)
# #     )
# #     in_probs = (
# #         jnp.zeros(
# #             (dim, dim),
# #             dtype=dtype,
# #         )
# #         .at[..., -1]
# #         .set(END_PROBABILITIES)
# #     )
# #     print(
# #         jnp.array([1.21343333, 0.99167143, 0.86207143])
# #         / jnp.array(
# #             [
# #                 1.2023,
# #                 0.9826,
# #                 0.8542,
# #             ]
# #         )
# #     )
# #     print(in_values, in_probs)
# #     print(
# #         jnp.dot(END_PROBABILITIES, END_VALUES),
# #         jnp.power(jnp.dot(END_PROBABILITIES, END_VALUES), 1.0 / 3.0),
# #         1.017 / 1.008,
# #     )
# #     back_update_tree(
# #         dim - 1,
# #         in_probs,
# #         in_values,
# #         jnp.power(jnp.dot(END_PROBABILITIES, END_VALUES), 1.0 / 3.0),
# #     )


# # def test_calc_recombining_tree():
# #     actual = calc_recombining_tree(
# #         END_PROBABILITIES,
# #         END_VALUES,
# #         jnp.power(jnp.dot(END_PROBABILITIES, END_VALUES), 1.0 / 3.0),
# #     )
# #     for _ in actual:
# #         print(_)
# #     print(
# #         jnp.power(jnp.dot(END_PROBABILITIES, END_VALUES), 1.0 / 3.0),
# #     )
# #     assert False


# # def test_rubinstein_value():

# #     pass


# # def test_discounters_equiv():
# #     tree = CRRBinomialTree(3, "european")
# #     start_price = jnp.array([1.0])
# #     u, d = tree._calc_factors(
# #         vol := jnp.array([0.1]),
# #         tte := jnp.array([0.25]),
# #     )
# #     end_values = tree._calc_end_values(start_price, u, d)
# #     risk_free_rate = jnp.array([0.05])
# #     end_probs = tree._calc_end_probabilities(
# #         u,
# #         d,
# #         tte,
# #         cost_of_carry := risk_free_rate,  # jnp.array([0.025]),
# #     )
# #     print("shapes: ", end_values.shape, end_probs.shape)
# #     print(end_values, end_probs)
# #     print(tree._forecast(start_price, u, d)[0])
# #     print(
# #         calc_recombining_tree(
# #             end_probs[0],
# #             end_values[0],
# #         )[0]
# #     )
# #     print(
# #         AmericanDiscounter(3)(
# #             end_values,
# #             strike := start_price,
# #             tte,
# #             risk_free_rate,
# #             is_call := jnp.array([1.0]),
# #             tree._calc_transition_up_probabilities(
# #                 u,
# #                 d,
# #                 tte,
# #                 cost_of_carry,
# #             ),
# #             u,
# #         )
# #     )

# # print(
# #     end_values[0],
# #     end_probs[0],
# #     jnp.power(jnp.dot(end_probs[0], end_values[0]), 1.0 / 3.0),
# #     jnp.log(jnp.dot(end_probs[0], end_values[0])),
# #     calc_recombining_tree(
# #         end_probs[0],
# #         end_values[0],
# #     ),
# # )
# # print(
# #     CRRBinomialTree(3, "american")(
# #         start_price,
# #         vol,
# #         tte,
# #         risk_free_rate,
# #         cost_of_carry,
# #         # jnp.array([0.0]),
# #         is_call := jnp.array([1.0]),
# #         strike,
# #     ),
# # )
# # print(
# #     AmericanDiscounterExp()(
# #         start_price,
# #         end_values,
# #         strike,
# #         tte,
# #         risk_free_rate,
# #         is_call,
# #         end_probs,
# #     )
# # )
# # assert False
# # print()

# #     discounter = AmericanDiscounter(3)
# #     a = discounter.build_next_value_body_function(
# #         strike := jnp.array([1.0]),
# #         tte := jnp.array([1.0]),
# #         cost_of_carry,
# #         is_call := jnp.array([1.0]),
# #         jnp.array([0.5]),
# #         u,
# #     )
# #     b = discounter.build_next_value_body_function_exp_rubinstein(
# #         start_price,
# #         end_values,
# #         strike,
# #         tte,
# #         None,
# #         is_call,
# #         end_probs,
# #     )
# #     print(
# #         "init vals: ",
# #         init_vals := discounter.exercise_valuer(
# #             end_values,
# #             strike,
# #             is_call,
# #         ),
# #         "end: ",
# #         end_values,
# #         "end_probs: ",
# #         end_probs,
# #     )
# #     print(
# #         "a: ",
# #         a(
# #             0,
# #             (
# #                 discounter.exercise_valuer(
# #                     end_values,
# #                     strike,
# #                     is_call,
# #                 ),
# #                 end_values,
# #             ),
# #         ),
# #         "\nb:",
# #         b(0, (init_vals, end_values, end_probs)),
# #     )

# # assert False


# # def test_build_fns():
# # def test_back_combine():
# #     actual_probabilities, actual_return_values = EXPECTED_NODE_PROBABILITIES[..., -1], EXPECTED_RETURN_VALUES[..., -1]

# #     for i in range(EXPECTED_NODE_PROBABILITIES.shape[-1] - 1):
# #         # for i in range(1):
# #         num_nodes_start = EXPECTED_NODE_PROBABILITIES.shape[-1] - i
# #         actual_probabilities, actual_return_values, up_transition_probs = back_combine(
# #             i,
# #             actual_probabilities,
# #             actual_return_values,
# #         )
# #         assert jnp.allclose(
# #             actual_probabilities,
# #             EXPECTED_NODE_PROBABILITIES[..., num_nodes_start - 2],
# #             atol=1e-3,
# #         )
# #         assert jnp.allclose(
# #             actual_return_values,
# #             EXPECTED_RETURN_VALUES[..., num_nodes_start - 2],
# #             atol=1e-3,
# #         )
# #         print(up_transition_probs)
# #     assert False


# # def test_calc_recombining_tree():
# #     actual_probabilities, actual_return_values = calc_recombining_tree(END_PROBABILITIES, END_VALUES)
# #     assert jnp.allclose(
# #         EXPECTED_NODE_PROBABILITIES,
# #         actual_probabilities,
# #         atol=1e-3,
# #     )
# #     assert jnp.allclose(
# #         EXPECTED_RETURN_VALUES,
# #         actual_return_values,
# #         atol=1e-3,
# #     )


# # def test_non_return_inputs():
# #     print(calc_recombining_tree(END_PROBABILITIES, END_VALUES)[1] * 100.0)
# #     assert False


# # def test_discounter():
# #     print(jnp.power(jnp.dot(END_VALUES, END_PROBABILITIES), 1.0 / 3.0))
# #     print(
# #         AmericanDiscounterExp()(
# #             jnp.array([1.0]),
# #             END_VALUES,
# #             jnp.array([0.8]),
# #             jnp.array([1.0]),
# #             jnp.log(jnp.power(jnp.dot(END_VALUES, END_PROBABILITIES), 1.0 / 3.0)),
# #             jnp.array([1.0]),
# #             END_PROBABILITIES,
# #         )
# #     )
# #     assert False


# def test_scan():
#     v = back_combine_path_probabilities(
#         paths := END_PROBABILITIES / comb(3, jnp.arange(4)),
#         END_VALUES,
#         rate := jnp.power(jnp.dot(END_PROBABILITIES, END_VALUES), 1.0 / 3.0),
#     )
#     print(v)
#     exp_paths = (
#         jnp.zeros(
#             (
#                 paths.shape[0],
#                 paths.shape[0],
#             )
#         )
#         .at[:, -1]
#         .set(paths)
#     )

#     exp_probs = (
#         jnp.zeros(
#             (
#                 END_VALUES.shape[0],
#                 END_VALUES.shape[0],
#             )
#         )
#         .at[:, -1]
#         .set(END_VALUES)
#     )
#     # print(exp_paths, exp_paths)

#     print(
#         "FOR: ",
#         jax.lax.fori_loop(
#             0,
#             3,
#             back_combine_paths_body,
#             (
#                 paths,
#                 END_VALUES,
#                 rate,
#             ),
#         ),
#     )
#     for _ in jax.lax.scan(
#         back_combine_paths_scan,
#         (
#             paths,
#             END_VALUES,
#             rate,
#         ),
#         None,
#         length=3,
#         # reverse=True,
#     ):
#         for _ in _:
#             print(_)

#     assert False
