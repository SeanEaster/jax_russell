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
        start_price: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_probabilities: jaxtyping.Float[jaxtyping.Array, "*#contracts n"],
        end_underlying_returns: jaxtyping.Float[jaxtyping.Array, "*contracts n"],
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

        return (
            jnp.exp(-risk_free_rate * time_to_expiration)
            * end_probabilities
            * self.exercise_valuer(
                start_price * end_underlying_returns,
                strike,
                is_call,
            )
        ).sum(0)


class AmericanDiscounter(Discounter):
    """A class for calculating the discounted value of an option that can be exercised any time.

    This class starts at the end nodes of binomial tree, and backward calculates the value.
    It takes the maximum of a node's discounted expected value and its exercise value at each step.

    Attributes:
        discounter (Callable): A function that returns the value of exercising an option.
    """

    def __init__(
        self,
        exercise_valuer: Callable = MaxValuer(),
    ) -> None:
        """

        Args:
            steps (int): number of steps used in the tree
            exercise_valuer (Callable, optional): Callable that takes `unadjusted_values` and returns exercise values. Defaults to MaxValuer().
        """  # noqa
        super().__init__(exercise_valuer)

    def __call__(
        self,
        start_price: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_probabilities: jaxtyping.Float[jaxtyping.Array, "num_nodes *#contracts"],
        end_underlying_returns: jaxtyping.Float[jaxtyping.Array, "num_nodes *#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> Any:
        """_summary_

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
            ) = back_combine_path_probabilities(path_probabilities, node_underlying_returns, implied_risk_free_rate)
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
    end_probabilities: jaxtyping.Float[jaxtyping.Array, "num_nodes *contracts"],
    end_underlying_returns: jaxtyping.Float[jaxtyping.Array, "num_nodes *contracts"],
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
            else AmericanDiscounter() if option_type == 'american' else EuropeanDiscounter()
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
            back_combine_paths_scan,
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
    @typeguard.typechecked
    def forecast_returns(
        self,
        volatility: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        cost_of_carry: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ) -> Tuple:
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
    ) -> jaxtyping.Float[jaxtyping.Array, "steps steps *#contracts"]:
        probabilities, returns = self.forecast_returns(volatility, time_to_expiration, cost_of_carry)
        return probabilities, returns * start_price

    @partial(jax.jit, static_argnums=0)
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
    def value(
        self,
        start_price: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_probabilities: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_underlying_returns: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        cost_of_carry: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ):
        return self.discounter(
            start_price,
            end_probabilities,
            end_underlying_returns,
            time_to_expiration,
            risk_free_rate,
            cost_of_carry,
            is_call,
            strike,
        )

    def forecast_returns(
        self,
        end_probabilities: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_underlying_returns: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ):
        return self._forecast(end_probabilities, end_underlying_returns)

    def forecast_values(
        self,
        start_price: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_probabilities: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
        end_underlying_returns: jaxtyping.Float[jaxtyping.Array, "*#contracts"],
    ):
        probabilities, returns = self.forecast_returns(end_probabilities, end_underlying_returns)
        return probabilities, returns * start_price


def back_combine_path_probabilities(
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


def back_combine_paths_scan(
    carry_tuple,
    _,
):
    probs, vals, risk_free_rate = carry_tuple

    ys = back_combine_path_probabilities(
        probs,
        vals,
        risk_free_rate,
    )[:-1]
    return (*ys, risk_free_rate), ys


def back_combine_paths_body(_, values_tuple):
    return *back_combine_path_probabilities(*values_tuple)[:-1], values_tuple[-1]
