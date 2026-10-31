import numpy as np
import pytest

from dezero.core import Variable, add, div, exp, square, as_variable, as_variables
from dezero.util import gradient_check, numerical_grad
from dezero.utils.array import sum_to


def test_gradient_check():
    def f(x: Variable) -> Variable:
        return exp(x)

    x = np.array([[1.0, 0.0], [4.0, -1]])
    assert gradient_check(f, x)


@pytest.mark.parametrize(
    argnames=("func", "x", "args", "want"),
    argvalues=[
        (square, [-1.0, 0.0, 2.0], (), [-2.0, 0.0, 4.0]),
        (lambda x: exp(square(x)), [1.0], (),[2 * np.e]),
        (add, [2.0], (np.array(1.0),), [1.0]),
        #(div, [4.0, 5])
    ],
)
def test_numerical_grad(func, x, args, want):
    x = np.array(x).astype(np.float64)
    want = np.array(want)
    args = as_variables(*args)
    y = numerical_grad(func, x, *args)
    np.testing.assert_allclose(y, want)

@pytest.mark.parametrize(
        argnames=(
            "x", "shape", "want",
        ),
        argvalues=[
            (
                [[1, 1, 1] ,[1, 1, 1]], (3,), [2, 2, 2]
            ),
            (
                [[1,2,3],[4,5,6]], (1, 3), [[5,7,9]]
            ),
            (
                [[5, -1, 6], [8, 98, -18]], (2, 1), [[10],[88]],
            )
        ],
)
def test_sum_to(x, shape, want):
    x = np.array(x)
    want = np.array(want)
    y = sum_to(x, shape)
    np.testing.assert_array_equal(y, want)

