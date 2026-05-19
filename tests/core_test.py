import numpy as np
import pytest

from dezero.core import Variable, as_array, exp, square


class TestSquare:
    @pytest.mark.parametrize(
        ("x", "expected", "assert_func"),
        [
            (np.array(2.0), np.array(4.0), np.testing.assert_equal),
        ],
    )
    def test_forward(self, x, expected, assert_func):
        x = Variable(x)
        y = square(x)
        assert_func(y.data, expected)

    @pytest.mark.parametrize(
        ("x", "expected", "assert_func"),
        [
            (np.array(3.0), np.array(6.0), np.testing.assert_equal),
        ],
    )
    def test_backward(self, x, expected, assert_func):
        x = Variable(x)
        y = square(x)
        y.backward()
        assert_func(x.grad, expected)


class TestExp:
    @pytest.mark.parametrize(
        ("x", "expected", "assert_func"),
        [
            (np.array([0.0, 1.0]), np.array([1.0, np.e]), np.testing.assert_equal),
        ],
    )
    def test_forward(self, x, expected, assert_func):
        x = as_array(x) if not isinstance(x, Variable) else x
        y = exp(x)
        assert_func(y.data, expected)

    @pytest.mark.parametrize(
        ("x", "expected", "assert_func"),
        [
            (np.array(3.0), np.array(6.0), np.testing.assert_equal),
        ],
    )
    def test_backward(self, x, expected, assert_func):
        x = Variable(x)
        y = square(x)
        y.backward()
        assert_func(x.grad, expected)
