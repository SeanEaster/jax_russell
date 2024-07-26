"""Test all valuation classes with all mixins."""

import pytest
from jax import numpy as jnp

from jax_russell.bsm import GeneralizedBlackScholesMerten
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

    try:
        UnderTest = decorator(tree_class)
        undertest = UnderTest(5, option_type)
        actual = undertest(*mixin_call_args)

        assert jnp.greater(actual, 0.0)
    except Exception as e:
        raise e
    finally:
        tree_class.__call__ = tree_class.__call__.__wrapped__


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

    try:
        UnderTest = decorator(GeneralizedBlackScholesMerten)
        undertest = UnderTest()
        actual = undertest(*mixin_call_args)

        assert jnp.greater(actual, 0.0)
    except Exception as e:
        raise e
    finally:
        GeneralizedBlackScholesMerten.__call__ = GeneralizedBlackScholesMerten.__call__.__wrapped__
