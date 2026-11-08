from dezero.core import Variable, variable
import dezero.functions as F

import numpy as np

from dezero.util import gradient_check
from dezero.utils.grad import numerical_grad


def rawtanh(x: Variable) -> Variable:
    return (F.exp(x) - F.exp(-x)) / (F.exp(x) + F.exp(-x))


def test_tanh():
    data = np.linspace(-10, 10, 1000)
    x = variable(data)
    y = F.tanh(x)
    wanty = rawtanh(x)

    np.testing.assert_allclose(y.data, wanty.data)

    y.backward()
    assert x.grad is not None
    gxd = x.grad.data

    x.cleargrad()

    wanty.backward(create_graph=True)
    assert x.grad is not None
    gwantxd = x.grad.data

    np.testing.assert_allclose(gxd, gwantxd)

    np.testing.assert_allclose(gxd, 1 - np.tanh(data) ** 2)


def test_tanh_numgrad():
    data = np.linspace(-1, 1, 100)
    x = variable(data)

    f = F.tanh
    ngrad = numerical_grad(f, data, eps=1e-3, enable5p=True)
    y = f(x)
    y.backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.data, ngrad)


def test_reshape():
    x = variable(np.random.randn(3 * 4))
    gradient_check(lambda x: x.reshape(3, 4), x)
