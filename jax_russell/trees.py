"""Tree models."""

import abc
import inspect
from functools import partial
from typing import Any, Callable, Tuple, Union

import jax
import jaxopt
import jaxopt._src
import jaxopt._src.implicit_diff
import jaxtyping
import typeguard
from jax import numpy as jnp
from jax.scipy.special import gammaln

from jax_russell.base import AllArgs, ValuationModel, broadcast_args


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


def calc_path_probabilities(
    node_probabilities: jaxtyping.Float[jaxtyping.Array, "nodes *#contracts"],
    steps: int,
) -> jaxtyping.Float[jaxtyping.Array, "nodes *#contracts"]:
    """Calculate path probabilities from node probabilities.

    Returns:
        jnp.array: probability for each unique path that can arrive at each node
    """
    coefs = comb(
        steps,
        jnp.arange(node_probabilities.shape[0]),
    )
    coefs = jnp.expand_dims(
        coefs,
        range(1, len(node_probabilities.shape)),
    )

    return jnp.where(
        coefs > 0.0,
        node_probabilities,
        0.0,
    ) / jnp.where(
        coefs > 0.0,
        coefs,
        1.0,
    )


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
    def __call__(
        self,
        start_price: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_probabilities: jaxtyping.Float[jaxtyping.Array, " num_nodes *#contracts"],
        end_underlying_returns: jaxtyping.Float[jaxtyping.Array, " num_nodes *contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "*#contracts"]:  # noqa F821
        """Must implement discounting and associated logic."""


class EuropeanDiscounter(Discounter):
    """Disounts final exercise values of binomial tree."""

    @typeguard.typechecked
    def __call__(
        self,
        start_price: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_probabilities: jaxtyping.Float[jaxtyping.Array, " num_nodes *#contracts"],
        end_underlying_returns: jaxtyping.Float[jaxtyping.Array, " num_nodes *contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
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

        weighted_discounted_exercise_values = (
            jnp.exp(-risk_free_rate * time_to_expiration)
            * end_probabilities
            * self.exercise_valuer(
                start_price * end_underlying_returns,
                strike,
                is_call,
            )
        )

        return weighted_discounted_exercise_values.sum(0)


class AmericanDiscounter(Discounter):
    """A class for calculating the discounted value of an option that can be exercised any time.

    This class starts at the end nodes of binomial tree, and backward calculates the value.
    It takes the maximum of a node's discounted expected value and its exercise value at each step.

    Attributes:
        discounter (Callable): A function that returns the value of exercising an option.
    """

    def __call__(
        self,
        start_price: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_probabilities: jaxtyping.Float[jaxtyping.Array, " num_nodes *#contracts"],
        end_underlying_returns: jaxtyping.Float[jaxtyping.Array, " num_nodes *#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> Any:
        """Discount from expiration, taking max of exercise and option value at each step.

        Args:
            start_price (jaxtyping.Float[jaxtyping.Array, ): _description_
            end_probabilities (jaxtyping.Float[jaxtyping.Array, &quot;num_nodes): _description_
            end_underlying_returns (jaxtyping.Float[jaxtyping.Array, &quot;num_nodes): _description_
            strike (jaxtyping.Float[jaxtyping.Array, ): _description_
            time_to_expiration (jaxtyping.Float[jaxtyping.Array, ): _description_
            risk_free_rate (jaxtyping.Float[jaxtyping.Array, ): _description_
            is_call (jaxtyping.Float[jaxtyping.Array, ): _description_

        Returns:
            Any: _description_
        """
        underlying_values = end_underlying_returns * start_price
        delta_t = time_to_expiration / (end_underlying_returns.shape[0] - 1)
        values = self.exercise_valuer(
            underlying_values,
            strike,
            is_call,
        )
        implied_risk_free_rate = calc_implied_risk_free_rate(end_probabilities, end_underlying_returns)
        path_probabilities = calc_path_probabilities(
            end_probabilities,
            end_probabilities.shape[0] - 1,
        )

        def body_fn(_, values_tuple):
            values, path_probabilities, node_underlying_returns = values_tuple
            (
                new_path_probabilities,
                new_node_underlying_returns,
                up_transition_probability,
            ) = _back_combine_path_probabilities(path_probabilities, node_underlying_returns, implied_risk_free_rate)
            downward_values = jnp.roll(values.at[0, ...].set(0.0), -1, 0)
            upward_values = jnp.where(new_path_probabilities > 0.0, values, 0.0)
            discounted_value = jnp.exp(-risk_free_rate * delta_t) * (
                up_transition_probability * upward_values + (1 - up_transition_probability) * downward_values
            )
            return (
                jnp.maximum(
                    self.exercise_valuer(
                        new_node_underlying_returns * start_price,
                        strike,
                        is_call,
                    ),
                    discounted_value,
                ),
                new_path_probabilities,
                new_node_underlying_returns,
            )

        values, *_ = jax.lax.fori_loop(
            0,
            end_probabilities.shape[0] - 1,
            body_fn,
            (values, path_probabilities, end_underlying_returns),
        )
        return values[0, ...] if len(values.shape) != 0 else jnp.expand_dims(values, -1)


def calc_implied_risk_free_rate(
    end_probabilities: jaxtyping.Float[jaxtyping.Array, " num_nodes *contracts"],
    end_underlying_returns: jaxtyping.Float[jaxtyping.Array, " num_nodes *contracts"],
):
    """Calculate the implied risk-free rate of an end-node forecast.

    Args:
        end_probabilities: probabilities of each node
        end_underlying_returns: underlying returns of each node

    Returns:
        jnp.array: the implied discrete risk-free rates
    """
    return jnp.power(
        jnp.multiply(end_probabilities, end_underlying_returns).sum(
            0,
            keepdims=True,
        ),
        1.0 / (end_probabilities.shape[0] - 1),
    )


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
            else AmericanDiscounter()
            if option_type == 'american'
            else EuropeanDiscounter()
        )

    def _calc_end_returns(
        self,
        up_factors: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        down_factors: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "num_end_nodes *#contracts"]:
        """Return the possible end values for the underlying.

        Returns:
            jnp.array: array with possible values of each contract in the last dimension
        """
        up_steps = jnp.expand_dims(
            jnp.flip(jnp.arange(self.steps + 1)),
            range(1, len(up_factors.shape) + 1),
        )

        return jnp.exp(up_steps * jnp.log(up_factors) + (self.steps - up_steps) * jnp.log(down_factors))

    def _right_expand_step_values(self, broadcastable_to):
        return jnp.expand_dims(
            jnp.flip(jnp.arange(self.steps + 1)),
            range(1, len(broadcastable_to.shape) + 1),
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

    def _forecast(self, end_probabilities, end_underlying_returns):
        _, (probabilities, forecasted_returns) = jax.lax.scan(
            _back_combine_paths_scan,
            (
                calc_path_probabilities(end_probabilities, self.steps),
                end_underlying_returns,
                calc_implied_risk_free_rate(end_probabilities, end_underlying_returns),
            ),
            None,
            length=self.steps,
        )
        probabilities = jnp.flipud(probabilities)
        num_paths = comb(
            jnp.arange(probabilities.shape[0]).reshape((-1, 1)),
            jnp.arange(probabilities.shape[-1]).reshape((1, -1)),
        )

        probabilities = probabilities * num_paths
        probabilities, forecasted_returns = jnp.concatenate(
            (
                probabilities,
                jnp.expand_dims(end_probabilities, 0),
            )
        ), jnp.concatenate(
            (jnp.flipud(forecasted_returns), jnp.expand_dims(end_underlying_returns, 0)),
        )

        return probabilities, forecasted_returns


class ForwardForecastTree(BinomialTree):
    """Class for trees that create a forward forecast based on assumed volatility.

    This groups methods shard by e.g. Cox Ross Rubinstein and Rendleman Bartter trees.
    """

    @typeguard.typechecked
    def forecast_returns(
        self,
        volatility: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        cost_of_carry: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> Tuple:
        """Forecast of underlying asset's assumed return distribution at expiration.

        Args:
            volatility (jaxtyping.Float[jaxtyping.Array, ): Assumed volatility
            time_to_expiration (jaxtyping.Float[jaxtyping.Array, ): Time to expiration of option and forecast
            cost_of_carry (jaxtyping.Float[jaxtyping.Array, ): Cost of carry

        Returns:
            Tuple: Probabilities and corresponding return ratios
        """
        end_probabilities, end_underlying_returns = self._calc_end_nodes(volatility, time_to_expiration, cost_of_carry)
        probabilities, forecasted_returns = self._forecast(end_probabilities, end_underlying_returns)
        return probabilities, forecasted_returns

    @typeguard.typechecked
    def forecast_values(
        self,
        start_price: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        volatility: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        cost_of_carry: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ):
        """Forecast of underlying asset's assumed price distribution at expiration.

        Args:
            start_price (jaxtyping.Float[jaxtyping.Array, ): Underlying asset start price
            volatility (jaxtyping.Float[jaxtyping.Array, ): Assumed volatility
            time_to_expiration (jaxtyping.Float[jaxtyping.Array, ): Time to expiration of option and forecast
            cost_of_carry (jaxtyping.Float[jaxtyping.Array, ): Cost of carry

        Returns:
            Tuple: Probabilities and corresponding return ratios
        """
        probabilities, returns = self.forecast_returns(volatility, time_to_expiration, cost_of_carry)
        return probabilities, returns * start_price

    @partial(jax.jit, static_argnums=0)
    @broadcast_args
    def __call__(
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
        end_probabilities, end_underlying_returns = self._calc_end_nodes(volatility, time_to_expiration, cost_of_carry)

        args = (
            start_price,
            end_probabilities,
            end_underlying_returns,
            strike,
            time_to_expiration,
            risk_free_rate,
            is_call,
        )
        return self.discounter(*args)

    @abc.abstractmethod
    def _calc_end_nodes(
        self,
        volatility,
        time_to_expiration,
        cost_of_carry,
    ):  # todo: docstring
        pass


class CRRBinomialTree(ForwardForecastTree):
    """Cox Ross Rubinstein binomial tree.

    `__call__()` is tested against example in Haug.
    """  # noqa

    @partial(jax.jit, static_argnums=0)
    @typeguard.typechecked
    @broadcast_args
    def __call__(
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
        end_probabilities, end_underlying_returns = self._calc_end_nodes(volatility, time_to_expiration, cost_of_carry)

        args = (
            start_price,
            end_probabilities,
            end_underlying_returns,
            strike,
            time_to_expiration,
            risk_free_rate,
            is_call,
        )
        return self.discounter(*args)

    def _calc_end_nodes(self, volatility, time_to_expiration, cost_of_carry):
        up_factors, down_factors = self._calc_factors(
            volatility,
            time_to_expiration,
        )
        end_probabilities, end_underlying_returns = self._calc_end_probabilities(
            up_factors,
            down_factors,
            time_to_expiration,
            cost_of_carry,
        ), self._calc_end_returns(
            up_factors,
            down_factors,
        )

        return end_probabilities, end_underlying_returns

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
    ) -> jaxtyping.Float[jaxtyping.Array, "num_end_nodes *#contracts"]:  # noqa
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
        up_steps = self._right_expand_step_values(up_factors)
        end_probabilities = (
            jnp.power(jnp.expand_dims(p_up, 0), up_steps)
            * jnp.power(1 - jnp.expand_dims(p_up, 0), self.steps - up_steps)
            * comb(self.steps, up_steps)
        )

        return end_probabilities


class RendlemanBartterBinomialTree(ForwardForecastTree):
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
        p_up = jnp.broadcast_to(jnp.array(0.5), broadcast_to.shape)
        p_up = jnp.expand_dims(p_up, 0)
        up_steps = self._right_expand_step_values(broadcast_to)
        end_probabilities = (
            jnp.power(p_up, up_steps)
            * jnp.power(
                1 - p_up,
                self.steps - up_steps,
            )
            * comb(self.steps, up_steps)
        )
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

    def _calc_end_nodes(self, volatility, time_to_expiration, cost_of_carry):
        up_factors, down_factors = self._calc_factors(
            volatility,
            time_to_expiration,
            cost_of_carry,
        )
        end_probabilities, end_underlying_returns = self._calc_end_probabilities(up_factors), self._calc_end_returns(
            up_factors,
            down_factors,
        )

        return end_probabilities, end_underlying_returns


class RubinsteinImpliedBinomialTree(BinomialTree):
    """Value options using implied trees over a single maturity as described in Rubinstein 1994."""

    @partial(jax.jit, static_argnums=0)
    def __call__(
        self,
        start_price: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_probabilities: jaxtyping.Float[jaxtyping.Array, "num_end_nodes *#contracts"],
        end_underlying_returns: jaxtyping.Float[jaxtyping.Array, " num_end_nodes *#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ):
        """Calculate option values from return distribution."""
        shape = jnp.broadcast_shapes(
            *[
                _.shape
                for _ in (
                    start_price,
                    strike,
                    time_to_expiration,
                    risk_free_rate,
                    is_call,
                )
            ]
        )
        (start_price, strike, time_to_expiration, risk_free_rate, is_call) = tuple(
            jnp.broadcast_arrays(start_price, strike, time_to_expiration, risk_free_rate, is_call)
        )
        end_probabilities = jnp.expand_dims(end_probabilities, list(range(-len(shape), 0)))
        end_underlying_returns = jnp.expand_dims(end_underlying_returns, list(range(-len(shape), 0)))

        return self.discounter(
            *jnp.broadcast_arrays(
                start_price,
                end_probabilities,
                end_underlying_returns,
                strike,
                time_to_expiration,
                risk_free_rate,
                # cost_of_carry,
                is_call,
            ),
        )

    def forecast_returns(
        self,
        end_probabilities: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_underlying_returns: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ):
        """Calculate full returns forecast from end return distribution."""
        return self._forecast(end_probabilities, end_underlying_returns)

    def forecast_values(
        self,
        start_price: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_probabilities: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_underlying_returns: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ):
        """Calculate full price forecast from end return distribution."""
        probabilities, returns = self.forecast_returns(end_probabilities, end_underlying_returns)
        return probabilities, returns * start_price

    def solve_implied(
        self,
        expected_option_values,
        init_params,
        bid_ask_dim=0,
        barrier_const=1,
        **kwargs,
    ):
        """Solve implied variable(s)."""
        assert "end_underlying_returns" not in init_params, "solving for `end_underlying_returns` is not supported"
        if AllArgs.end_probabilities.value in init_params:
            if len(init_params) > 1:
                return self._solve_implied(expected_option_values, init_params, **kwargs)

            return self._solve_implied_probabilities(
                expected_option_values,
                init_params[AllArgs.end_probabilities.value],
                bid_ask_dim=bid_ask_dim,
                barrier_const=barrier_const,
                **kwargs,
            )

        return super().solve_implied(
            expected_option_values,
            init_params,
            **kwargs,
        )

    def _solve_implied(
        self,
        expected_option_values,
        init_params,
        bid_ask_dim,
        barrier_const,
        **kwargs,
    ):
        signature = inspect.signature(self.__call__)
        # inspect signature using bind to make sure all args have been passed
        signature.bind(**{**init_params, **kwargs})

        @jax.jit
        def objective(params, expected, kwargs):
            bound_arguments = signature.bind(
                **{
                    **params,
                    **kwargs,
                    # becomes an inner loop optimmizing probabilities
                    AllArgs.end_probabilities.value: self._solve_implied_probabilities(
                        expected_option_values,
                        params.get(AllArgs.end_probabilities.value),
                        bid_ask_dim=bid_ask_dim,
                        barrier_const=barrier_const,
                        **kwargs,
                    ),
                }
            )
            residuals = expected - self(*bound_arguments.args, **bound_arguments.kwargs)
            return jnp.mean(residuals**2)

        solver = jaxopt.LBFGSB(
            objective,
        )
        res = solver.run(
            init_params,
            expected=expected_option_values,
            kwargs=kwargs,
            bounds=(
                {k: jnp.zeros_like(v) for k, v in init_params.items()},
                {k: jnp.ones_like(v) * jnp.inf for k, v in init_params.items()},
            ),
        )
        return res

    def _solve_implied_probabilities(
        self,
        expected_option_values,
        init_probabilities,
        bid_ask_dim=0,
        barrier_const=1,
        **kwargs,
    ):
        signature = inspect.signature(self.__call__)
        signature.bind(**{**kwargs, AllArgs.end_probabilities.value: init_probabilities})

        ind = jnp.zeros(tuple(1 for _ in expected_option_values.shape), dtype=jnp.int32)
        # todo: add option for assumed relative spread,
        # (default to None and require one of bid_ask_dim or
        # relative_spread to be set?)
        bid, ask = jnp.take_along_axis(expected_option_values, ind, bid_ask_dim), jnp.take_along_axis(
            expected_option_values, ind + 1, bid_ask_dim
        )

        @jax.jit
        def init_feasible_obj(nat_params):
            log_probs = jnp.concat((jnp.zeros(1), nat_params))
            bound_arguments = signature.bind(
                **{
                    **kwargs,
                    AllArgs.end_probabilities.value: (probs := jax.nn.softmax(log_probs)),
                }
            )

            preds = self(*bound_arguments.args, **bound_arguments.kwargs)
            residuals = expected_option_values - self(*bound_arguments.args, **bound_arguments.kwargs)

            predictive_error = jnp.mean(residuals**2)

            prices_barrier = jnp.exp(
                barrier_const * (jax.nn.relu(preds - ask) + barrier_const * jax.nn.relu(bid - preds))
            ).mean()

            implied_rate = jnp.dot(probs, kwargs.get(AllArgs.end_underlying_returns.value))
            rate_difference = (
                jnp.exp(kwargs.get(AllArgs.risk_free_rate.value) * kwargs.get(AllArgs.time_to_expiration.value))
                - implied_rate
            )
            rate_barrier = jnp.exp(jnp.abs(rate_difference)).sum()

            return prices_barrier + rate_barrier + predictive_error

        init_log_probs = jnp.log(init_probabilities)
        nat_params = (init_log_probs - init_log_probs[0])[1:]
        return jax.nn.softmax(
            jnp.concat(
                (
                    jnp.zeros(1),
                    jaxopt.LBFGS(init_feasible_obj).run(nat_params).params,
                )
            )
        )


def _back_combine_path_probabilities(
    path_probabilities,
    node_return_values,
    risk_free_rate,
):
    downward_path_probabilities = jnp.roll(path_probabilities.at[0, ...].set(0.0), -1, 0)
    upward_path_probabilities = jnp.where(downward_path_probabilities != 0.0, path_probabilities, 0.0)
    up_transition_probability = jnp.where(
        (combined_path_probabilities := upward_path_probabilities + downward_path_probabilities) > 0.0,
        upward_path_probabilities / combined_path_probabilities,
        0.0,
    )
    node_return_values = jnp.where(
        up_transition_probability > 0.0,
        (
            up_transition_probability * node_return_values
            + (1 - up_transition_probability) * jnp.roll(node_return_values, -1, 0)
        )
        / risk_free_rate,
        0.0,
    )
    return combined_path_probabilities, node_return_values, up_transition_probability


def _back_combine_paths_scan(
    carry_tuple,
    _,
):
    probs, vals, risk_free_rate = carry_tuple

    ys = _back_combine_path_probabilities(
        probs,
        vals,
        risk_free_rate,
    )[:-1]
    return (*ys, risk_free_rate), ys
