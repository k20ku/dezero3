import numpy as np
import pytest

from collections.abc import Callable
from dezero.core import NDArray, Variable, add, asarray, rdiv, exp, square, as_variables
from dezero.util import gradient_check
from dezero.utils.grad import numerical_grad



@pytest.mark.parametrize(
    argnames=("func", "x", "args", "want"),
    argvalues=[
        (square, [-1.0, 0.0, 2.0], (), [-2.0, 0.0, 4.0]),
        # y = exp(square(x)) -> dy/dx = exp(square(x))*2*x
        (lambda x: exp(square(x)), [1.0, 2.0, -2.4], (), lambda x: np.exp(x**2)*2*x),
        # y = x + 1 -> dy/dx = 1
        (add, [2.0], (1.0,), [1.0]),
        # y / x -> d/dx(y/x) = -y/x**2
        (rdiv, [4.0, - 5.0], ([4.0, -12.1],), lambda x, y: -y/np.array([4.0, -5.0])**2)
    ],
)
def test_numerical_grad(func, x, args, want: Callable[..., NDArray] | object):
    x = asarray(x).astype(np.float64)
    args = tuple(as_variables(*args))
    if isinstance(want, Callable):
        want = want(x, *args)
    want: NDArray = asarray(want)
    y = numerical_grad(func, x, *args)
    np.testing.assert_allclose(y, want)

