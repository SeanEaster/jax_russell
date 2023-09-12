"""Module for shared components and utilities."""


import abc
import inspect
from functools import partial
from typing import Any, Protocol

import jax
import jaxtyping
from jax import numpy as jnp


class ImplementsValueProtocol(Protocol):
    """Protocol used to tell `mypy` mixins rely on another class to implement `value()`."""

    def value(
        self,
        start_price: jaxtyping.Float[
            jaxtyping.Array,
            "#contracts",
        ],
        volatility: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        cost_of_carry: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "#contracts"]:
        """Should be implemented by another mixed in class.

        Returns:
            jnp.array: option contract values
        """


class ValuationModel(abc.ABC):
    """Abstract class for valuation methods."""

    argnums = list(range(4))

    @abc.abstractmethod
    def value(
        self,
        start_price: jaxtyping.Float[
            jaxtyping.Array,
            "#contracts",
        ],
        volatility: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        cost_of_carry: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "#contracts"]:
        """Calculate the value of an option.

        This method is used internally by `__call__()`, and should return the value of options.
        By default, `__call__()` is a pass through to `value()`, but many available mixins overwrite this behavior to pass arguments to `value()`.
        In these cases, this allows the single, general method `value()` to implement valuations, while leveraging `__call__()` for security-specific argument logic and meaningful autodifferentiation.
        """  # noqa

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> jaxtyping.Float[jaxtyping.Array, "#contracts"]:
        """Value arrays of options.

        By default, `__call__` checks its arguments against `value()` and passes them through.

        Returns:
            jnp.array: option values
        """
        inspect.signature(self.value).bind(*args, **kwargs)
        return self.value(*jnp.broadcast_arrays(*args), **kwargs)

    @partial(jax.jit, static_argnums=0)
    def first_order(self, *args, **kwargs):
        """Automatically calculate first-order greeks.

        Returns:
            _type_: _description_
        """
        inspect.signature(self).bind(*args, **kwargs)
        return jnp.hstack(
            jax.jacfwd(
                self,
                self.argnums,
            )(*args, **kwargs)
        )

    @partial(jax.jit, static_argnums=0)
    def second_order(self, *args, **kwargs):
        """Automatically calculate second-order greeks.

        Returns:
            _type_: _description_
        """
        inspect.signature(self).bind(*args, **kwargs)
        return jax.jacfwd(
            self.first_order,
            self.argnums,
        )(*args, **kwargs)


class AsayMargineduturesOptionMixin:
    """Assumes zero interest and zero cost of carry."""

    def __call__(
        self: ImplementsValueProtocol,
        start_price: jaxtyping.Float[
            jaxtyping.Array,
            "#contracts",
        ],
        volatility: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "#contracts"]:
        """Calculate values for option contracts.

        Assumes zero interest and zero cost of carry.

        Returns:
            jnp.array: contract values
        """
        return self.value(
            start_price,
            volatility,
            time_to_expiration,
            is_call,
            strike,
            jnp.zeros(1),
            jnp.zeros(1),
        )


class FuturesOptionMixin:
    """Assume zero cost of carry."""

    def __call__(
        self: ImplementsValueProtocol,
        start_price: jaxtyping.Float[
            jaxtyping.Array,
            "#contracts",
        ],
        volatility: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "#contracts"]:
        """Calculate values for option contracts.

        Assumes zero cost of carry.

        Returns:
            jnp.array: contract values
        """
        return self.value(
            start_price,
            volatility,
            time_to_expiration,
            is_call,
            strike,
            risk_free_rate,
            jnp.zeros(risk_free_rate.shape),
        )


class StockOptionContinuousDividendMixin:
    """Adjust a stock option by a continuous dividend."""

    def __call__(
        self: ImplementsValueProtocol,
        start_price: jaxtyping.Float[
            jaxtyping.Array,
            "#contracts",
        ],
        volatility: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        continuous_dividend: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "#contracts"]:
        """Calculate values for option contracts.

        Adjusts the risk-free rate by (subtracting) the continuous dividend to calculate cost of carry.

        Returns:
            jnp.array: contract values
        """
        return self.value(
            start_price,
            volatility,
            time_to_expiration,
            is_call,
            strike,
            risk_free_rate,
            risk_free_rate - continuous_dividend,
        )


class StockOptionMixin:
    """Uses `risk_free_rate` for both the risk free rate and cost of carry.

    This gives the correct rho, and is the cost of carry defined in Haug.
    """

    def __call__(
        self: ImplementsValueProtocol,
        start_price: jaxtyping.Float[
            jaxtyping.Array,
            "#contracts",
        ],
        volatility: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        time_to_expiration: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        risk_free_rate: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        is_call: jaxtyping.Float[jaxtyping.Array, "#contracts"],
        strike: jaxtyping.Float[jaxtyping.Array, "#contracts"],
    ) -> jaxtyping.Float[jaxtyping.Array, "#contracts"]:
        """Calculate values for option contracts.

        Uses the risk-free rate for both risk-free rate and cost of carry, ensuring accurate greeks.

        Returns:
            jnp.array: contract values
        """
        (
            start_price,
            volatility,
            time_to_expiration,
            risk_free_rate,
            is_call,
            strike,
        ) = jnp.broadcast_arrays(
            start_price,
            volatility,
            time_to_expiration,
            risk_free_rate,
            is_call,
            strike,
        )
        return self.value(
            start_price,
            volatility,
            time_to_expiration,
            risk_free_rate,
            risk_free_rate,
            is_call,
            strike,
        )
