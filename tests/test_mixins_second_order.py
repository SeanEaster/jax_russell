"""Test all valuation classes with all mixins second order greeks."""
import pytest

from tests.base import mixin_call_args, mixin_classes, option_types
from tests.trees import tree_classes


@pytest.mark.parametrize("tree_class", tree_classes)
@pytest.mark.parametrize("option_type", option_types)
@pytest.mark.parametrize("mixin_class,mixin_call_args", zip(mixin_classes, mixin_call_args))
def test_mixins_second_order(
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

    UnderTest(5, option_type).second_order(*mixin_call_args)
