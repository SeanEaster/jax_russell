"""Test all valuation classes with all mixins."""

import jax
import pytest
from jax import numpy as jnp

from jax_russell.bsm import GeneralizedBlackScholesMerten
from tests import base, trees

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("tree_class", trees.forward_tree_classes)
@pytest.mark.parametrize("option_type", base.option_types)
@pytest.mark.parametrize(
    "decorator,mixin_call_args",
    zip(
        base.class_decorators,
        base.mixin_call_args,
    ),
)
def test_mixins_call(
    tree_class,
    option_type,
    decorator,
    mixin_call_args,
):
    """Test instantiation and call for all tree classes, option types and securuity mixins.

    Args:
        tree_class (trees.CRRBinomialTree): A CRRBinomialTree or child
        option_type (str): one of 'american' or 'european'
        mixin_class (Callable): a mixin class that implements __call__() for the tree
        mixin_call_args (Tuple[Any]): args to pass tree.__call__()
    """

    @decorator
    class UnderTest(tree_class):  # type: ignore
        pass

    undertest = UnderTest(5, option_type)
    actual = undertest(*mixin_call_args)

    assert jnp.greater(actual, 0.0)


@pytest.mark.parametrize(
    "decorator,mixin_call_args",
    zip(
        base.class_decorators,
        base.mixin_call_args,
    ),
)
def test_mixins_call_bsm(
    decorator,
    mixin_call_args,
):
    """Test instantiation and call for all tree classes, option types and securuity mixins.

    Args:
        tree_class (trees.CRRBinomialTree): A CRRBinomialTree or child
        option_type (str): one of 'american' or 'european'
        mixin_class (Callable): a mixin class that implements __call__() for the tree
        mixin_call_args (Tuple[Any]): args to pass tree.__call__()
    """

    @decorator
    class UnderTest(GeneralizedBlackScholesMerten):  # type: ignore
        pass

    undertest = UnderTest()
    actual = undertest(*mixin_call_args)

    assert jnp.greater(actual, 0.0)
