"""Tree models."""

import abc
from functools import partial
from typing import Any, Callable, Tuple, Union

import jax
import jaxtyping
import typeguard
from jax import numpy as jnp
from jax.scipy.special import gammaln

from jax_russell.base import ValuationModel


# binomial as suggested here https://github.com/google/jax/discussions/7044
def comb(
    N: Union[int, float, jaxtyping.Float[jaxtyping.Array, "*"]],
    k: Union[int, float, jaxtyping.Float[jaxtyping.Array, "*"]],
) -> Union[float, jaxtyping.Float[jaxtyping.Array, "*"]]:
    """Jax-friendly implementation of the binomial coefficient.

    Returns:
        jax.array: number of unique combinations when drawing k from N items
    """
    return jnp.exp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1))


def calc_path_probabilities(node_probabilities, steps):
    coefs = comb(
        steps,
        jnp.arange(node_probabilities.shape[-1]),
    )
    # print(coefs, jnp.where(coefs > 0.0, node_probabilities, 0.0) / jnp.where(coefs > 0.0, coefs, 1.0))

    return jnp.where(coefs > 0.0, node_probabilities, 0.0) / jnp.where(coefs > 0.0, coefs, 1.0)


def calc_time_steps(
    change_tolerance: float,
    tree_class: Callable,
    tree_class_args: Tuple,
    tree_call_args: Tuple,
) -> int:
    """Calculate the number of time steps that finer grained trees are within `change_tolerance`.

    Args:
        change_tolerance (float): Maximum allowable change between tree with `steps - 1` and `steps` time steps.

    Returns:
        int: minimum number of steps
    """
    time_steps = 1
    price_change = jnp.array(jnp.inf)
    tree = tree_class(time_steps, *tree_class_args)

    while jnp.abs(price_change) > change_tolerance:
        time_steps += 1
        price_change = jnp.abs(
            tree(*tree_call_args) - (tree := tree_class(time_steps, *tree_class_args))(*tree_call_args)
        )
    return time_steps


class ExerciseValuer(abc.ABC):
    """Abstract class for Callables that implement, or approximate, the max(exercise value, 0) operation.

    This is applied in the intermediate steps of a binomial tree.
    """

    @typeguard.typechecked
    def __call__(
        self,
        underlying_values: jaxtyping.Float[jaxtyping.Array, "*#contracts n"],
        strike: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "*#contracts"]:
        """Calculate or approximate the value of exercising an option.

        Args:
            underlying_values (jaxtyping.Float[jaxtyping.Array, "#contracts n"]): value of the underlying asset
            strike (jaxtyping.Float[jaxtyping.Array, "*#contracts"]): option strike prices
            is_call (jaxtyping.Float[jaxtyping.Array, "*#contracts"]): whether each option is a call (1.0) or put (0.0)

        Returns:
            jaxtyping.Float[jaxtyping.Array, "*#contracts"]: Exercise values.
        """
        return self.adjust(
            self._calc_unadjusted_value(
                underlying_values,
                strike,
                is_call,
            )
        )

    @typeguard.typechecked
    def _calc_unadjusted_value(
        self,
        underlying_values: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "*#contracts"]:
        return (underlying_values - strike) * (2 * is_call - 1)

    @abc.abstractmethod
    def adjust(
        self,
        unadjusted_values: jaxtyping.Float[jax.Array, "*"],
    ) -> jaxtyping.Float[jaxtyping.Array, "*"]:
        """Adjust value difference to calculate an intermediate exercise value.

        This method should transform the difference between strike and underlying, i.e. `underlying - strike` for calls, `strike - underlying` for puts, to an exercise value.
        For example, a standard binomial tree uses max(unadjusted_values, 0.0).
        """  # noqa


class MaxValuer(ExerciseValuer):
    """Implements the standard maximum operation found in intermediate steps in binomial trees."""

    def adjust(
        self,
        unadjusted_values: jaxtyping.Float[jax.Array, "*"],
    ) -> jaxtyping.Float[jaxtyping.Array, "*"]:
        """Adjust signed strike-underlying differences by applying the max op.

        Args:
            unadjusted_values (jaxtyping.Float[jax.Array, "*"]): `underlying - strike` for calls, `strike - underlying` for puts

        Returns:
            jaxtyping.Float[jaxtyping.Array, "*"]: element-wise max(unadjusted_values, 0.0)
        """  # noqa
        return jnp.maximum(unadjusted_values, 0.0)


class SoftplusValuer(ExerciseValuer):
    """Approximate the maximum operation using a softplus function.

    This Callable will return `log(1 + exp(kx)) / k` where k is the sharpness parameter.
    """

    def __init__(self, sharpness: float = 1.0) -> None:
        """

        Args:
            sharpness (float): sharpness parameter k
        """  # noqa
        super().__init__()
        self.sharpness = sharpness

    def adjust(
        self,
        unadjusted_values: jaxtyping.Float[jax.Array, "*"],
        sharpness: Union[None, float] = None,
    ) -> jaxtyping.Float[jaxtyping.Array, "*"]:
        """Adjust using the softplus function.

        Args:
            unadjusted_values: jaxtyping.Float[jax.Array, "*"]): `underlying - strike` for calls, `strike - underlying` for puts
            sharpness: If None, uses `self.sharpness`

        Returns:
            jaxtyping.Float[jaxtyping.Array, "*"]: element-wise softplus
        """  # noqa
        return jnp.logaddexp((self.sharpness if sharpness is None else sharpness) * unadjusted_values, 0.0) / (
            self.sharpness if sharpness is None else sharpness
        )


class Discounter(abc.ABC):
    """Abstract class for Callable objects that discount final values of a tree."""

    def __init__(
        self,
        exercise_valuer: Callable = MaxValuer(),
    ) -> None:
        """

        Args:
            exercise_valuer (Callable, optional): Callable that takes `unadjusted_values` and returns exercise values. Defaults to MaxValuer().
        """  # noqa
        self.exercise_valuer = exercise_valuer

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:  # noqa
        """Must implement discounting and associated logic."""


class EuropeanDiscounter(Discounter):
    """Disounts final exercise values of binomial tree."""

    def __init__(
        self,
        exercise_valuer: Callable = MaxValuer(),
    ) -> None:
        """

        Args:
            exercise_valuer (Callable, optional): Callable that takes `unadjusted_values` and returns exercise values. Defaults to MaxValuer().
        """  # noqa
        super().__init__(exercise_valuer)

    @typeguard.typechecked
    def __call__(
        self,
        end_underlying_values: jaxtyping.Float[jaxtyping.Array, "*#contracts n"],
        strike: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_probabilities: jaxtyping.Float[jaxtyping.Array, "*#contracts n"],
    ) -> jaxtyping.Float[jaxtyping.Array, "*#contracts"]:
        """Calculate discounted expected value at expiration.

        Args:
            end_underlying_values (jaxtyping.Float[jaxtyping.Array, ): possible value for the underlying asset
            strike (jaxtyping.Float[jaxtyping.Array, ): contract strikes
            time_to_expiration (jaxtyping.Float[jaxtyping.Array, ): contract times to expiration
            risk_free_rate (jaxtyping.Float[jaxtyping.Array, ): risk free interest
            is_call (jaxtyping.Float[jaxtyping.Array, ): floats denoting whether each option is a call (1.0) or put (0.0)
            end_probabilities (jaxtyping.Float[jaxtyping.Array, ): probability that the underlying take the corresponding value in end_underlying_values

        Returns:
            jnp.array: discounted expected value of each contract at expiration
        """  # noqa

        return (
            jnp.exp(-risk_free_rate * time_to_expiration)
            * end_probabilities
            * self.exercise_valuer(
                end_underlying_values,
                strike,
                is_call,
            )
        ).sum(-1)


class AmericanDiscounter(Discounter):
    """Discount from end values, determining optimality of exercise at each time step."""

    def __init__(
        self,
        steps: int,
        exercise_valuer: Callable = MaxValuer(),
    ) -> None:
        """

        Args:
            steps (int): number of steps used in the tree
            exercise_valuer (Callable, optional): Callable that takes `unadjusted_values` and returns exercise values. Defaults to MaxValuer().
        """  # noqa
        super().__init__(exercise_valuer)
        self.steps = steps

    @typeguard.typechecked
    def __call__(
        self,
        end_underlying_values: jaxtyping.Float[jaxtyping.Array, "*contracts n"],
        strike: jaxtyping.Float[jaxtyping.Array, "*contracts 1"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*contracts 1"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "*contracts 1"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*contracts 1"],
        p_up: jaxtyping.Float[jaxtyping.Array, "*contracts 1"],
        up_factor: jaxtyping.Float[jaxtyping.Array, "*contracts 1"],
    ) -> jaxtyping.Float[jaxtyping.Array, "*contracts"]:  # noqa
        """Calculate discounted value of an American option.

        Args:
            end_underlying_values (jaxtyping.Float[jaxtyping.Array, ): possible value for the underlying asset
            strike (jaxtyping.Float[jaxtyping.Array, ): contract strikes
            time_to_expiration (jaxtyping.Float[jaxtyping.Array, ): contract times to expiration
            risk_free_rate (jaxtyping.Float[jaxtyping.Array, ): risk free interest
            is_call (jaxtyping.Float[jaxtyping.Array, ): floats denoting whether each option is a call (1.0) or put (0.0)
            p_up (jaxtyping.Float[jaxtyping.Array, ): probability of an upward move in the underlying at each time step
            up_factor (jaxtyping.Float[jaxtyping.Array, ): factor applied for an upward move

        Returns:
            jnp.array: discounted option values
        """  # noqa
        underlying_values = end_underlying_values
        delta_t = time_to_expiration / self.steps
        values = self.exercise_valuer(
            underlying_values,
            strike,
            is_call,
        )

        def next_value(_, values_tuple):
            values, underlying_values = values_tuple
            discounted_value = jnp.exp(-risk_free_rate * delta_t) * (
                (1 - p_up) * values[..., :-1] + p_up * values[..., 1:]
            )
            underlying_values = underlying_values * up_factor
            values = values.at[..., :-1].set(
                self.exercise_valuer(
                    underlying_values[..., :-1],
                    strike,
                    is_call,
                )
            )
            values = values.at[..., :-1].set(
                jnp.maximum(
                    discounted_value,
                    values[..., :-1],
                )
            )
            return values, underlying_values

        values, underlying_values = jax.lax.fori_loop(0, self.steps, next_value, (values, underlying_values))

        return values[..., 0] if len(values.shape) != 0 else jnp.expand_dims(values, -1)

    def build_next_value_body_function(
        self,
        strike,
        time_to_expiration,
        risk_free_rate,
        is_call,
        p_up,
        up_factor,
    ):

        delta_t = time_to_expiration / self.steps

        def next_value(_, values_tuple):
            values, underlying_values = values_tuple
            discounted_value = jnp.exp(-risk_free_rate * delta_t) * (
                (1 - p_up) * values[..., :-1] + p_up * values[..., 1:]
            )
            underlying_values = underlying_values / up_factor
            values = values.at[..., :-1].set(
                self.exercise_valuer(
                    underlying_values[..., :-1],
                    strike,
                    is_call,
                )
            )
            values = values.at[..., :-1].set(
                jnp.maximum(
                    discounted_value,
                    values[..., :-1],
                )
            )
            return values, underlying_values

        return next_value

    def build_next_value_body_function_exp_rubinstein(
        self,
        start_price,
        end_underlying_values,
        strike,
        time_to_expiration,
        risk_free_rate,
        is_call,
        end_probabilities,
    ):
        # this should
        tree_implied_risk_free_rate = jnp.power(
            jnp.squeeze(
                jnp.dot(
                    end_underlying_values,
                    jnp.expand_dims(end_probabilities, -1),
                ),
                -1,
            )
            / start_price,
            1.0 / (end_underlying_values.shape[-1] - 1),
        )
        delta_t = time_to_expiration / self.steps

        if risk_free_rate is None:
            risk_free_rate = tree_implied_risk_free_rate

        def next_value(idx, values_tuple):
            values, underlying_values, node_probabilities = values_tuple

            # this derives the path probabilities P+ and P- for Rubinstein's "step one"
            path_probabilities = node_probabilities / comb(
                update_from := node_probabilities.shape[-1] - 1 - idx,
                jnp.arange(
                    node_probabilities.shape[-1],
                ),
            )
            upward_path_probabilities = path_probabilities[..., :-1]
            downward_path_probabilities = path_probabilities[..., 1:]

            # add them as in Rubinstein "step one," multiply by combinatorial factor to calculate node probabilities
            node_probabilities = (upward_path_probabilities + downward_path_probabilities) * comb(
                update_from - 1,
                jnp.arange(upward_path_probabilities.shape[-1]),
            )
            # "step two"
            p_up = upward_path_probabilities / (upward_path_probabilities + downward_path_probabilities)
            discounted_value = jnp.exp(-risk_free_rate * delta_t) * (
                (1 - p_up) * values[..., :-1] + p_up * values[..., 1:]
            )
            underlying_values = (
                p_up * underlying_values[..., :-1] + (1.0 - p_up) * underlying_values[..., 1:]
            ) / tree_implied_risk_free_rate

            values = jnp.maximum(
                self.exercise_valuer(underlying_values, strike, is_call),
                discounted_value,
            )

            return values, underlying_values, node_probabilities

        return next_value


class BinomialTree(ValuationModel):
    """Base abstract class for binomial trees."""

    def __init__(
        self,
        steps: int,
        option_type: str,
        discounter: Union[AmericanDiscounter, EuropeanDiscounter, None] = None,
    ) -> None:
        """

        Args:
            steps (int): The number of time steps in the binomial tree.
        """  # noqa
        assert option_type in [
            "european",
            "american",
        ], f"option_type must be one of `european` or `american` got {option_type}"
        assert (
            discounter is None
            or getattr(discounter, "steps", None) is None
            or getattr(discounter, "steps", None) == steps
        )
        self.steps = steps
        self.option_type = option_type
        self.discounter = (
            discounter
            if discounter is not None
            else AmericanDiscounter(steps) if option_type == 'american' else EuropeanDiscounter()
        )

    def _calc_end_values(
        self,
        start_price: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        up_factors: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        down_factors: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "*#contracts n"]:
        """Return the possible end values for the underlying.

        Returns:
            jnp.array: array with possible values of each contract in the last dimension
        """
        up_steps = jnp.flip(jnp.arange(self.steps + 1))

        return jnp.exp(
            jnp.log(jnp.expand_dims(start_price, -1))
            + up_steps * jnp.log(jnp.expand_dims(up_factors, -1))
            + (self.steps - up_steps) * jnp.log(jnp.expand_dims(down_factors, -1))
        )

    @typeguard.typechecked
    def _calc_transition_up_probabilities(
        self,
        up_factors: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        down_factors: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        cost_of_carry: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "*#contracts"]:
        """Calculate the probability of an upward move at any step in the tree.

        Args:
            up_factors (jaxtyping.Float[jaxtyping.Array, ): factor for upward movement
            down_factors (jaxtyping.Float[jaxtyping.Array, ): factor for downward movement
            time_to_expiration (jaxtyping.Float[jaxtyping.Array, ): Contract times to expiration in years
            cost_of_carry (jaxtyping.Float[jaxtyping.Array, ): Contract costs of carry

        Returns:
            jnp.array: probability of an upward transition
        """
        p_up = (jnp.exp(cost_of_carry * (time_to_expiration / self.steps)) - down_factors) / (up_factors - down_factors)
        return p_up

    def _transform_args_for_discounter(
        self,
        time_to_expiration,
        risk_free_rate,
        cost_of_carry,
        is_call,
        strike,
        up_factors,
        down_factors,
        end_probabilities,
        end_underlying_values,
    ):
        args_to_expand = [
            strike,
            time_to_expiration,
            risk_free_rate,
            is_call,
        ] + (
            [
                self._calc_transition_up_probabilities(
                    up_factors,
                    down_factors,
                    time_to_expiration,
                    cost_of_carry,
                ),
                up_factors,
            ]
            if self.option_type == "american"
            else []
        )
        args = (
            [end_underlying_values]
            + [jnp.expand_dims(_, -1) for _ in args_to_expand]
            + ([end_probabilities] if self.option_type == "european" else [])
        )

        return args


class CRRBinomialTree(BinomialTree):
    """Cox Ross Rubinstein binomial tree.

    `__call__()` is tested against example in Haug.
    """  # noqa

    @partial(jax.jit, static_argnums=0)
    @typeguard.typechecked
    def value(
        self,
        start_price: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        volatility: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        cost_of_carry: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "*#contracts"]:
        """Calculate values for option contracts.

        Returns:
            jnp.array: contract values
        """
        up_factors, down_factors = self._calc_factors(
            volatility,
            time_to_expiration,
        )
        end_probabilities = self._calc_end_probabilities(
            up_factors,
            down_factors,
            time_to_expiration,
            cost_of_carry,
        )

        end_underlying_values = self._calc_end_values(
            start_price,
            up_factors,
            down_factors,
        )

        args = self._transform_args_for_discounter(
            time_to_expiration,
            risk_free_rate,
            cost_of_carry,
            is_call,
            strike,
            up_factors,
            down_factors,
            end_probabilities,
            end_underlying_values,
        )
        return self.discounter(*args)

    def _calc_factors(
        self,
        volatility: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> Tuple[jaxtyping.Float[jaxtyping.Array, "*#contracts"], jaxtyping.Float[jaxtyping.Array, "*#contracts"]]:
        """Calculates the factor by which an asset price is multiplied for upward, downward movement at a step.

        Returns:
            jnp.array, jnp.array: factors on upward move, factors on downward move
        """
        scaled_volatility = volatility * jnp.sqrt(time_to_expiration / self.steps)
        return jnp.exp(scaled_volatility), jnp.exp(-scaled_volatility)

    @partial(jax.jit, static_argnums=0)
    @typeguard.typechecked
    def _calc_end_probabilities(
        self,
        up_factors: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        down_factors: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        cost_of_carry: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "*#contracts n"]:  # noqa
        """Calculate the probability of arriving at every end node in the tree.

        Returns:
            jnp.Array: Array with probabiliities in the last dimension, size `self.steps + 1`
        """
        p_up = self._calc_transition_up_probabilities(
            up_factors,
            down_factors,
            time_to_expiration,
            cost_of_carry,
        )
        up_steps = jnp.arange(self.steps + 1)
        end_probabilities = jnp.power(jnp.expand_dims(p_up, -1), up_steps) * jnp.power(
            1 - jnp.expand_dims(p_up, -1), self.steps - up_steps
        )
        if self.option_type == "european":
            end_probabilities *= comb(self.steps, up_steps)

        return end_probabilities


class RendlemanBartterBinomialTree(BinomialTree):
    """Rendleman Bartter tree method (equal probability of upward and downward movement).

    `__call__()` is tested to within 3e-2 (absolute and relative tolerance) of published results.
    """

    def _calc_end_probabilities(
        self,
        broadcast_to,
    ) -> jaxtyping.Float[jaxtyping.Array, "contracts"]:  # noqa
        """Calculate the probability of arriving at every end node in the tree.

        In the Rendleman Bartter tree, the p(up) = p(down) = 0.5.

        Returns:
            jnp.Array: Array with probabiliities in the last dimension, size `self.steps + 1`
        """
        p_up = jnp.broadcast_to(jnp.array([0.5]), broadcast_to.shape)
        p_up = jnp.expand_dims(p_up, -1)
        up_steps = jnp.flip(jnp.arange(self.steps + 1))

        end_probabilities = jnp.power(p_up, up_steps) * jnp.power(
            1 - p_up,
            self.steps - up_steps,
        )
        if self.option_type == "european":
            end_probabilities *= comb(self.steps, up_steps)
        return end_probabilities

    def _calc_factors(
        self,
        volatility: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        cost_of_carry: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> Tuple[jaxtyping.Float[jaxtyping.Array, "*#contracts"], jaxtyping.Float[jaxtyping.Array, "*#contracts"]]:
        """Calculates the factor by which an asset price is multiplied for upward, downward movement at a step.

        Returns:
            jnp.array, jnp.array: factors on upward move, factors on downward move
        """
        scaled_volatility = volatility * jnp.sqrt(delta_t := time_to_expiration / self.steps)
        const = (cost_of_carry - jnp.power(volatility, 2.0) / 2.0) * delta_t
        return jnp.exp(const + scaled_volatility), jnp.exp(const - scaled_volatility)

    @partial(jax.jit, static_argnums=0)
    def value(
        self,
        start_price: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        volatility: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        cost_of_carry: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "*#contracts"]:
        """Calculate values for option contracts.

        Returns:
            jnp.array: contract values
        """
        up_factors, down_factors = self._calc_factors(
            volatility,
            time_to_expiration,
            cost_of_carry,
        )
        end_probabilities = self._calc_end_probabilities(up_factors)
        end_underlying_values = self._calc_end_values(
            start_price,
            up_factors,
            down_factors,
        )

        args = self._transform_args_for_discounter(
            time_to_expiration,
            risk_free_rate,
            cost_of_carry,
            is_call,
            strike,
            up_factors,
            down_factors,
            end_probabilities,
            end_underlying_values,
        )
        return self.discounter(*args)


class RubinsteinImpliedBinomialTree(BinomialTree):
    """Value options using implied trees over a single maturity as described in Rubinstein 1994."""

    @partial(jax.jit, static_argnums=0)
    def value(
        self,
        start_price: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_probabilities: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_underlying_values: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        cost_of_carry: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ):
        pass


class RubinsteinAmericanDiscounter:

    def __call__(
        self,
        end_probabilities: jaxtyping.Float[
            jaxtyping.Array, "*#contracts n"
        ],  # todo: update shapes against AmericanDiscounter as example
        end_underlying_values: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "*contracts 1"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*contracts 1"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "*contracts 1"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*contracts 1"],
        underlying_return,  # todo: figure out shape
    ):
        underlying_values = end_underlying_values
        delta_t = time_to_expiration / self.steps
        values = self.exercise_valuer(
            underlying_values,
            strike,
            is_call,
        )

        def next_value(index_from_right, values_tuple):
            values, underlying_values, node_probabilities = values_tuple
            update_from_index = node_probabilities.shape[-1] - index_from_right
            path_probabilities = node_probabilities / comb(
                update_from_index,
                jnp.arange(node_probabilities.shape[-1]),
            )
            upward = path_probabilities[..., :-1]
            downward = path_probabilities[..., 1:]
            p_up = upward / (upward + downward)
            discounted_value = jnp.exp(-risk_free_rate * delta_t) * (
                (1 - p_up) * values[..., 1:] + p_up * values[..., :-1]
            )

            values = values.at[..., :-1].set(
                jnp.maximum(
                    self.exercise_valuer(
                        underlying_values[..., -1],
                        strike,
                        is_call,
                    ),
                    discounted_value,
                )
            )

            node_probabilities = node_probabilities.at[..., :-1].set(
                (upward + downward)
                * comb(
                    update_from_index - 1,
                    jnp.arange(upward.shape[-1]),
                )
            )
            underlying_values = (
                underlying_values.at[..., :-1].set(
                    (1.0 - p_up) * underlying_values[..., 1:] + p_up * underlying_values[..., :-1]
                )
                / underlying_return
            )

            return values, underlying_values, node_probabilities

        values, _, _ = jax.lax.fori_loop(
            0, end_underlying_values.shape[-1] - 1, next_value, (values, underlying_values, end_probabilities)
        )

        return values[..., 0] if len(values.shape) != 0 else jnp.expand_dims(values, -1)


@partial(jax.jit, static_argnums=0)
def back_update_tree(
    update_from_index: int,
    node_probabilities: jaxtyping.Float[jaxtyping.Array, "*batch steps+1 steps+1"],
    node_values: jaxtyping.Float[jaxtyping.Array, "*batch steps+1 steps+1"],
    stepwise_cost_of_carry,
):

    node_probabilities_vec = node_probabilities[..., update_from_index]

    path_probabilities = node_probabilities_vec / comb(
        update_from_index,
        jnp.arange(node_probabilities_vec.shape[-1]),
    )
    upward = path_probabilities[..., :-1]
    downward = path_probabilities[..., 1:]
    updated_probabilities_vec = (upward + downward) * comb(
        update_from_index - 1,
        jnp.arange(upward.shape[-1]),
    )

    up_transition_probability = upward / (upward + downward)
    node_values_vec = node_values[..., update_from_index]
    updated_values_vec = (1.0 - up_transition_probability) * node_values_vec[
        1:
    ] + up_transition_probability * node_values_vec[:-1]
    updated_values_vec = updated_values_vec / stepwise_cost_of_carry

    return (
        node_probabilities.at[..., :-1, update_from_index - 1].set(updated_probabilities_vec),
        node_values.at[..., :-1, update_from_index - 1].set(updated_values_vec),
        stepwise_cost_of_carry,
    )


@partial(jax.jit, static_argnums=0)
def back_update_body_fn(index_from_right, values_tuple):

    node_probabilities, node_values, stepwise_cost_of_carry = values_tuple
    update_from_index = node_probabilities.shape[-1] - index_from_right
    return back_update_tree(
        update_from_index,
        node_probabilities,
        node_values,
        stepwise_cost_of_carry,
    )


def calc_recombining_tree(
    end_probabilities: jaxtyping.Float[jaxtyping.Array, "*batch steps+1"],
    end_values: jaxtyping.Float[jaxtyping.Array, "*batch steps+1"],
    stepwise_cost_of_carry,
):
    steps = end_probabilities.shape[-1] - 1
    end_probabilities /= end_probabilities.sum(
        -1,
        keepdims=True,
    )
    node_probabilities = (
        jnp.zeros(end_probabilities.shape + (end_probabilities.shape[-1],)).at[..., :, -1].set(end_probabilities)
    )
    node_values = jnp.zeros(end_values.shape + (end_values.shape[-1],)).at[..., :, -1].set(end_values)
    node_probabilities, node_values, _ = jax.lax.fori_loop(
        1,
        steps + 1,
        back_update_body_fn,
        (node_probabilities, node_values, stepwise_cost_of_carry),
    )
    return jnp.triu(node_probabilities), jnp.triu(node_values)


def back_combine(
    iter_from_end_time: int,
    node_probabilities,
    node_return_values,
):  # todo: type hints and shapes

    # todo: refactor to avoid different sizes

    num_nodes_start = node_probabilities.shape[-1] - iter_from_end_time

    path_probabilities = calc_path_probabilities(
        node_probabilities,
        steps := num_nodes_start - 1,
    )

    upward_path_probabilities, downward_path_probabilities = (
        path_probabilities[..., :-1],
        path_probabilities[..., 1:],
    )
    up_transition_probability = upward_path_probabilities / (
        combined_path_probabilities := upward_path_probabilities + downward_path_probabilities
    )

    return (
        jnp.zeros(node_probabilities.shape)
        .at[..., :-1]
        .set(
            new_node_probabilities := combined_path_probabilities
            * comb(
                steps - 1,
                jnp.arange(upward_path_probabilities.shape[-1]),
            )
        ),
        jnp.zeros(node_return_values.shape)
        .at[..., :-1]
        .set(
            jnp.where(
                new_node_probabilities > 0.0,
                (
                    up_transition_probability * node_return_values[..., :-1]
                    + (1 - up_transition_probability) * node_return_values[..., 1:]
                )
                / jnp.power(
                    jnp.squeeze(
                        jnp.dot(
                            node_probabilities[..., :],
                            jnp.expand_dims(
                                node_return_values[..., :],
                                -1,
                            ),
                        ),
                        -1,
                    ),
                    1.0 / (num_nodes_start - 1),
                ),
                0.0,
            )
        ),
    )


def calc_recombining_tree_exp(
    end_probabilities: jaxtyping.Float[jaxtyping.Array, "*batch steps+1"],
    end_values: jaxtyping.Float[jaxtyping.Array, "*batch steps+1"],
):
    end_probabilities = (
        jnp.zeros(end_probabilities.shape + (end_probabilities.shape[-1],)).at[..., :, -1].set(end_probabilities)
    )
    end_values = jnp.zeros(end_values.shape + (end_values.shape[-1],)).at[..., :, -1].set(end_values)
    num_nodes = end_probabilities.shape[-1]

    def tree_combine(i, values_tuple):
        end_probabilities, end_values = values_tuple
        next_end_probabilities, next_end_values = back_combine(
            i,
            end_probabilities[..., :, source_idx := num_nodes - 1 - i],
            end_values[..., :, source_idx],
        )

        return end_probabilities.at[..., :, update_idx := source_idx - 1].set(next_end_probabilities), end_values.at[
            ..., :, update_idx
        ].set(next_end_values)

    return jax.lax.fori_loop(
        0,
        end_probabilities.shape[-1] - 1,
        tree_combine,
        (
            end_probabilities,
            end_values,
        ),
    )
