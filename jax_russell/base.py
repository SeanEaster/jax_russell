"""Module for shared components and utilities."""

import abc
import inspect
from enum import Enum
from functools import partial, wraps
from typing import Callable, Protocol

import jax
import jaxopt
import jaxtyping
from jax import numpy as jnp


class AllArgs(Enum):

    start_price = "start_price"
    volatility = "volatility"
    time_to_expiration = "time_to_expiration"
    is_call = "is_call"
    strike = "strike"
    risk_free_rate = "risk_free_rate"
    cost_of_carry = "cost_of_carry"
    continuous_dividend = "continuous_dividend"


def first_order_greeks(value_fn: Callable) -> Callable:
    """Decorate a value function to instead return first-order greeks.

    Args:
        value_fn (Callable): _description_

    Returns:
        _type_: _description_
    """

    @partial(jax.jit, static_argnums=0)
    def first_order(self, *args, argnums=None, **kwargs):

        return jnp.hstack(
            jax.jacfwd(
                value_fn,
                range(len(args)) if argnums is None else argnums,
            )(*args, **kwargs)
        )

    return first_order


def second_order_greeks(first_order: Callable):

    @partial(jax.jit, static_argnums=0)
    def second_order(*args, argnums=None, **kwargs):

        return jnp.concatenate(
            jax.jacfwd(
                first_order,
                range(len(args)) if argnums is None else argnums,
            )(*args, **kwargs),
            axis=-1,
        )

    return second_order


def greeks(cls: Callable) -> Callable:

    cls.first_order = first_order_greeks(cls.__call__)
    cls.second_order = second_order_greeks(cls.first_order)
    return cls


def zero_named_args(arg_names):

    if type(arg_names) is not list:
        arg_names = [arg_names]

    def decorate(value_fn):
        parent_signature, child_signature = signatures(value_fn, arg_names)

        @wraps(value_fn)
        def updated_value_fn(*args, **kwargs):
            child_arguments = child_signature.bind(*args)
            shared_params = {k: v for k, v in child_arguments.arguments.items() if k in parent_signature.parameters}
            static_args = {arg_name: jnp.zeros(1) for arg_name in arg_names}
            parent_arguments = parent_signature.bind(
                **{
                    **shared_params,
                    **static_args,
                }
            )
            return value_fn(*parent_arguments.args)

        return updated_value_fn

    return decorate


def asay_margined(cls):
    cls.__call__ = zero_named_args(
        [
            AllArgs.cost_of_carry.value,
            AllArgs.risk_free_rate.value,
        ]
    )(cls.__call__)
    return cls


def futures_option(cls):
    cls.__call__ = zero_named_args(AllArgs.cost_of_carry.value)(cls.__call__)
    return cls


def stock_option_continuous_dividend(value_fn):

    parent_signature = inspect.signature(value_fn)
    parameters = [
        (param.replace(name=AllArgs.continuous_dividend.value) if par_name == AllArgs.cost_of_carry.value else param)
        for par_name, param in parent_signature.parameters.items()
    ]

    child_signature = parent_signature.replace(parameters=parameters)

    @wraps(value_fn)
    def updated_value_fn(*args):
        child_arguments = child_signature.bind(*args)
        shared_params = {k: v for k, v in child_arguments.arguments.items() if k in parent_signature.parameters}
        parent_arguments = parent_signature.bind_partial(**shared_params)
        parent_arguments.arguments[AllArgs.cost_of_carry.value] = (
            child_arguments.arguments[AllArgs.risk_free_rate.value]
            - child_arguments.arguments[AllArgs.continuous_dividend.value]
        )

        return value_fn(*parent_arguments.args)

    return updated_value_fn


def stock_option_continuous_dividend_cls(cls):

    cls.__call__ = stock_option_continuous_dividend(cls.__call__)
    return cls


def stock_option(value_fn):

    parent_signature, child_signature = signatures(
        value_fn,
        arg_names=AllArgs.cost_of_carry.value,
    )

    @wraps(value_fn)
    def updated_value_fn(*args):
        child_arguments = child_signature.bind(*args)
        shared_params = {k: v for k, v in child_arguments.arguments.items() if k in parent_signature.parameters}
        parent_arguments = parent_signature.bind(
            **{
                **shared_params,
                **{AllArgs.cost_of_carry.value: child_arguments.arguments[AllArgs.risk_free_rate.value]},
            }
        )
        return value_fn(*parent_arguments.args)

    return updated_value_fn


def signatures(value_fn, arg_names):
    parent_signature = inspect.signature(value_fn)
    parameters = [param for par_name, param in parent_signature.parameters.items() if par_name not in arg_names]

    child_signature = parent_signature.replace(parameters=parameters)
    return parent_signature, child_signature


def stock_option_cls(cls):
    cls.__call__ = stock_option(cls.__call__)
    return cls


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


@greeks
class ValuationModel(abc.ABC):
    """Abstract class for valuation methods."""

    def solve_implied(
        self,
        expected_option_values,
        init_params,
        **kwargs,
    ):
        """Solve for an implied value, usually volatility.

        This method allows the flexibility to solve for any combination of values used in the valuation method's `__call__()` signature.
        For example, passing `{"risk_free_rate": jnp.array([0.05]),"volatility":jnp.array([.5])}` will solve for the implied values of both volatility and the risk free rate.

        Args:
            expected_option_values jnp.array: option values, typically observed market prices
            init_params dict[jnp.array]: initial guesses to begin solve optimization

        Returns:
            params, state: the parameters and state returned by a `jaxopt` optimizer `run()`
        """  # noqa: E501
        signature = inspect.signature(self.__call__)  # todo: refactor into decorator?
        # inspect signature using bind to make sure all args have been passed
        signature.bind(**{**init_params, **kwargs})

        # todo: if end_probabilities is in init_params, take log here...

        @jax.jit
        def objective(params, expected, kwargs):
            # todo ...and softmax here
            bound_arguments = signature.bind(**{**params, **kwargs})
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
