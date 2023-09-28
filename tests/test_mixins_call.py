"""Test all valuation classes with all mixins."""
import pytest
from jax import numpy as jnp

from jax_russell.bsm import GeneralizedBlackScholesMerten
from tests.base import mixin_call_args, mixin_classes, option_types
from tests.trees import tree_classes


@pytest.mark.parametrize("tree_class", tree_classes)
@pytest.mark.parametrize("option_type", option_types)
@pytest.mark.parametrize("mixin_class,mixin_call_args", zip(mixin_classes, mixin_call_args))
def test_mixins_call(
    tree_class,
    option_type,
    mixin_class,
    mixin_call_args,
):
    """Test instantiation and call for all tree classes, option types and securuity mixins.

    Args:
        tree_class (trees.CRRBinomialTree): A CRRBinomialTree or child
        option_type (str): one of 'american' or 'european'
        mixin_class (Callable): a mixin class that implements __call__() for the tree
        mixin_call_args (Tuple[Any]): args to pass tree.__call__()
    """

    class UnderTest(mixin_class, tree_class):
        pass

    assert jnp.greater(UnderTest(5, option_type)(*mixin_call_args), 0.0)


@pytest.mark.parametrize("mixin_class,mixin_call_args", zip(mixin_classes, mixin_call_args))
def test_mixins_call_bsm(
    mixin_class,
    mixin_call_args,
):
    """Test instantiation and call for all tree classes, option types and securuity mixins.

    Args:
        tree_class (trees.CRRBinomialTree): A CRRBinomialTree or child
        option_type (str): one of 'american' or 'european'
        mixin_class (Callable): a mixin class that implements __call__() for the tree
        mixin_call_args (Tuple[Any]): args to pass tree.__call__()
    """

    class UnderTest(mixin_class, GeneralizedBlackScholesMerten):
        pass

    assert jnp.greater(UnderTest()(*mixin_call_args), 0.0)
