"""Module for shared components and utilities."""

import abc
import inspect
from enum import Enum
from functools import partial, wraps
from typing import Callable, List, Tuple

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


def broadcast_args(meth: Callable) -> Callable:
    """Wrap a function to broadcast its inputs (arrays) before call.

    Args:
        meth (Callable): A callable whose inputs are arrays.

    Returns:
        Callable: A callable that calls `jnp.broadcast_arrays` before passing to the original function
    """

    @wraps(meth)
    def broadcasted(self, *args):
        return meth(self, *jnp.broadcast_arrays(*args))

    return broadcasted


def first_order_greeks(value_fn: Callable) -> Callable:
    """Decorate a value function to instead return first-order greeks.

    The returned function will accept the same arguments, in the same order, as the orginal, and return derivatives in corresponding order.
    E.g., if `start_price` is the first argument, delta will be the first value in the returned derivatives.
    The returned function will have addition keyword argument `argnums`.
    This can be used to select a subset of greeks by passing a list of their **1-indexed** indices and 0, which corresponds to `self`.

    Args:
        value_fn (Callable): A function or other callable that returns option values.

    Returns:
        Callable: A callable that returns first derivatives of option values w.r.t. its inputs.
    """

    @partial(jax.jit, static_argnums=0)
    def first_order(*args, argnums=None, **kwargs):

        return jnp.hstack(
            jax.jacfwd(
                value_fn,
                range(1, len(args)) if argnums is None else argnums,
            )(*args, **kwargs)
        )

    return first_order


def second_order_greeks(first_order: Callable) -> Callable:
    """Decorate a value function to instead return second-order greeks.

    The returned function will accept the same arguments, in the same order, as the orginal, and return derivatives in corresponding order.
    E.g., if `start_price` is the first argument, gamma will be the first value in the returned derivatives.
    The returned function will have addition keyword argument `argnums`.
    This can be used to select a subset of greeks by passing a list of their **1-indexed** indices and 0, which corresponds to `self`.

    Args:
        value_fn (Callable): A function or other callable that returns option values.

    Returns:
        Callable: A callable that returns second derivatives of option values w.r.t. its inputs.
    """

    @partial(jax.jit, static_argnums=0)
    def second_order(*args, argnums=None, **kwargs):

        return jnp.concatenate(
            jax.jacfwd(
                first_order,
                range(1, len(args)) if argnums is None else argnums,
            )(*args, **kwargs),
            axis=-1,
        )

    return second_order


def greeks(cls):
    """Decorate a class to support first- and second-order greeks.

    Args:
        cls (Callable): a class whose `__call__` method calculates option values

    Returns:
        cls: the passed class, decorated to add `first_order()` and `second_order()` methods
    """

    cls.first_order = first_order_greeks(cls.__call__)
    cls.second_order = second_order_greeks(cls.first_order)
    return cls


def zero_named_args(arg_names: List[str]) -> Callable:

    if type(arg_names) is not list:
        arg_names = [arg_names]

    def decorate(value_fn: Callable) -> Callable:
        parent_signature, child_signature = signatures(value_fn, arg_names)

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

        updated_value_fn.__signature__ = child_signature
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


def stock_option_continuous_dividend(value_fn: Callable) -> Callable:
    """Decorate a Callable to use `risk_free_rate` - `continuous_dividend` as `cost_of_carry`.

    The returned function will pass (`risk_free_rate` - `continuous_dividend`) as `cost_of_carry` to `value_fn` and return the result.
    The returned function's signature is modified to replace `cost_of_carry` with `continuous_dividend`.

    Args:
        value_fn (Callable): A function that includes arguments `risk_free_rate` and `cost_of_carry`

    Returns:
        Callable: A modified function that uses `(risk_free_rate - continuous_dividend)` as `cost_of_carry`
    """
    parent_signature = inspect.signature(value_fn)
    parameters = [
        (param.replace(name=AllArgs.continuous_dividend.value) if par_name == AllArgs.cost_of_carry.value else param)
        for par_name, param in parent_signature.parameters.items()
    ]

    child_signature = parent_signature.replace(parameters=parameters)

    def updated_value_fn(*args):
        child_arguments = child_signature.bind(*args)
        shared_params = {k: v for k, v in child_arguments.arguments.items() if k in parent_signature.parameters}
        parent_arguments = parent_signature.bind_partial(**shared_params)
        parent_arguments.arguments[AllArgs.cost_of_carry.value] = (
            child_arguments.arguments[AllArgs.risk_free_rate.value]
            - child_arguments.arguments[AllArgs.continuous_dividend.value]
        )

        return value_fn(*parent_arguments.args)

    updated_value_fn.__signature__ = child_signature
    return updated_value_fn


def stock_option_continuous_dividend_cls(cls):
    """Decorate class to support continous dividends.

    Returns:
        cls: `cls` with `__call__` decorated with `stock_option_continuous_dividend`
    """

    cls.__call__ = stock_option_continuous_dividend(cls.__call__)
    return cls


def stock_option(value_fn: Callable) -> Callable:
    """Decorate a Callable to use to use `risk_free_rate` as `cost_of_carry`.

    The returned function will pass `risk_free_rate` as both `risk_free_rate` and `cost_of_carry` to `value_fn` and return the result.
    The returned function's signature is modified to remove `cost_of_carry`.

    Args:
        value_fn (Callable): A function that includes arguments `risk_free_rate` and `cost_of_carry`

    Returns:
        Callable: A modified function that uses `risk_free_rate` for `cost_of_carry`
    """

    parent_signature, child_signature = signatures(
        value_fn,
        arg_names=AllArgs.cost_of_carry.value,
    )

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

    updated_value_fn.__signature__ = child_signature
    return updated_value_fn


def signatures(value_fn: Callable, arg_names: List[str]) -> Tuple[inspect.Signature, inspect.Signature]:
    """Inspect signature and remove arguments.

    Args:
        value_fn (Callable): A function whose arguments include those listed in `arg_names`
        arg_names (List[str]): A list of argument names found in `value_fn`

    Returns:
        Tuple[inspect.Signature, inspect.Signature]: (Original Signature, reduced signature that excludes `arg_names`)
    """

    parent_signature = inspect.signature(value_fn)
    parameters = [param for par_name, param in parent_signature.parameters.items() if par_name not in arg_names]

    child_signature = parent_signature.replace(parameters=parameters)
    return parent_signature, child_signature


def stock_option_cls(cls):
    """Decorate a class to use `risk_free_rate` as `cost_of_carry`."""
    cls.__call__ = stock_option(cls.__call__)
    return cls


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
