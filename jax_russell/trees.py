"""Tree models."""


import abc
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
    price_change = jnp.inf
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
            _type_: _description_
        """  # noqa
        underlying_values = end_underlying_values
        delta_t = time_to_expiration / self.steps
        values = self.exercise_valuer(
            underlying_values,
            strike,
            is_call,
        )
        while values.shape[-1] > 1:
            discounted_value = jnp.exp(-risk_free_rate * delta_t) * (
                (1 - p_up) * values[..., :-1] + p_up * values[..., 1:]
            )

            underlying_values = underlying_values[..., :-1] * up_factor
            values = self.exercise_valuer(
                underlying_values,
                strike,
                is_call,
            )

            values = jnp.maximum(discounted_value, values)

        return values[..., 0] if len(values.shape) != 0 else jnp.expand_dims(values, -1)


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
            else AmericanDiscounter(steps)
            if option_type == 'american'
            else EuropeanDiscounter()
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
        up_steps = jnp.arange(self.steps + 1)

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
    """Base class for binomial trees.

    ``jax_russell.BinomialTree`` houses operations common across various flavors of pricing models that employ binomial trees. For example, both European and American stock options TK

    """  # noqa

    @typeguard.typechecked
    def value(
        self,
        start_price: jaxtyping.Float[
            jaxtyping.Array,
            "*#contracts",
        ],
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
        if self.option_type == "european":
            end_probabilities *= comb(self.steps + 1, jnp.arange(self.steps + 1))

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
    """Rendleman Bartter tree method (equal probability of upward and downward movement)."""

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
        up_steps = jnp.arange(self.steps + 1)

        end_probabilities = jnp.power(p_up, up_steps) * jnp.power(1 - p_up, self.steps - up_steps)
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
