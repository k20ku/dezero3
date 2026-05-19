import numpy as np
import pytest

from dezero.core import Variable, exp, square
from dezero.utils import gradient_check, numerical_grad


def test_gradient_check():
    def f(x: Variable) -> Variable:
        return exp(x)

    x = np.array([[1.0, 0.0], [4.0, -1]])
    assert gradient_check(f, x)


@pytest.mark.parametrize(
    ("func", "x", "desired"),
    [
        (square, np.array([-1.0, 0.0, 2.0]), np.array([-2.0, 0.0, 4.0])),
        (lambda x: exp(square(x)), np.array([1.0]), np.array([2 * np.e])),
    ],
)
def test_numerical_grad(func, x, desired):
    actual = numerical_grad(func, x, eps=1e-4)
    np.testing.assert_allclose(actual, desired)
