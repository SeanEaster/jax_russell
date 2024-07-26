"""Test all valuation classes with all mixins first order greeks."""

import pytest

from jax_russell.base import greeks
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
def test_mixins_first_order(
    tree_class,
    option_type,
    decorator,
    mixin_call_args,
):
    """Test instantiation and first_order() for all tree classes, option types and securuity mixins.

    Args:
        tree_class (trees.CRRBinomialTree): A CRRBinomialTree or child
        option_type (str): one of 'american' or 'european'
        mixin_class (Callable): a mixin class that implements __call__() for the tree
        mixin_call_args (Tuple[Any]): args to pass tree.__call__()
    """

    @greeks
    @decorator
    class UnderTest(tree_class):  # type: ignore
        pass

    UnderTest(5, option_type).first_order(*mixin_call_args)


@pytest.mark.parametrize("decorator,mixin_call_args", zip(base.class_decorators, base.mixin_call_args))
def test_mixins_first_order_bsm(
    decorator,
    mixin_call_args,
):
    """Test instantiation and first_order() for all tree classes, option types and securuity mixins.

    Args:
        tree_class (trees.CRRBinomialTree): A CRRBinomialTree or child
        option_type (str): one of 'american' or 'european'
        mixin_class (Callable): a mixin class that implements __call__() for the tree
        mixin_call_args (Tuple[Any]): args to pass tree.__call__()
    """

    @greeks
    @decorator
    class UnderTest(GeneralizedBlackScholesMerten):
        pass

    UnderTest().first_order(*mixin_call_args)
