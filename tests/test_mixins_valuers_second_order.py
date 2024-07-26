"""Test all valuation classes with all mixins, valuers second order greeks."""

import pytest

from jax_russell.base import greeks
from jax_russell.trees import AmericanDiscounter, SoftplusValuer
from tests import base, trees


@pytest.mark.parametrize("tree_class", trees.forward_tree_classes)
@pytest.mark.parametrize("option_type", base.option_types)
@pytest.mark.parametrize(
    "decorator,mixin_call_args",
    zip(
        base.class_decorators,
        base.mixin_call_args,
    ),
)
@pytest.mark.parametrize(
    "valuer_class,valuer_args",
    [
        (SoftplusValuer, (2.5e-2,)),
    ],
)
def test_mixins_valuers_second_order(
    tree_class,
    option_type,
    decorator,
    mixin_call_args,
    valuer_class,
    valuer_args,
):
    """Using non-default valuer, test instantiation and call for all tree classes, option types and security mixins.

    Args:
        tree_class (trees.BinomialTree): class of tree
        option_type (str): One of 'american', 'european'
        mixin_class: a security mixin class to test
        mixin_call_args (Tuple[Any]): args to pass __call__()
        valuer_class: ExerciseValuer concrete child class to test
        valuer_args Tuple[Any]: args to pass `valuer_class.__init__()`
    """  # noqa

    @greeks
    @decorator
    class UnderTest(tree_class):  # type: ignore
        pass

    steps = 5
    UnderTest(
        steps,
        option_type,
        AmericanDiscounter(valuer_class(*valuer_args)) if option_type == "american" else None,
    ).second_order(*mixin_call_args)
