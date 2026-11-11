from functools import partial

import numpy as np
import pytest

import dezero.functions as F
from dezero.core import Variable, as_variables, variable
from dezero.typing import NDArray
from dezero.util import CallableV, gradient_check
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


@pytest.mark.parametrize(
    argnames=("data", "f"),
    argvalues=[
        (np.random.randn(2, 2), lambda x: x.transpose()),
        (np.random.randn(4), lambda x: x.transpose(0)),
        (np.random.randn(3, 4, 5), lambda x: x.transpose(2, 0, 1)),
    ],
)
def test_transpose(data, f: CallableV):
    x = variable(data)
    gradient_check(f, x)


@pytest.mark.parametrize(
    argnames=("data", "f"),
    argvalues=[
        (np.random.randn(2, 3), lambda x: F.sum_to(x, shape=(2, 1))),
        (np.random.randn(4, 2, 5), lambda x: F.sum_to(x, shape=(5,))),
    ],
)
def test_sum_to(data: NDArray, f: CallableV):
    x = variable(data)
    gradient_check(f, x)


@pytest.mark.parametrize(
    argnames=("data", "f"),
    argvalues=[
        (np.random.randn(5), lambda x: F.broadcast_to(x, shape=(2, 5))),
        (np.random.randn(4, 2, 5), lambda x: F.broadcast_to(x, shape=(3, 4, 2, 5))),
        (np.random.randn(4), lambda x: F.broadcast_to(x, shape=(4,))),
    ],
)
def test_broadcast_to(data: NDArray, f: CallableV):
    x = variable(data)
    gradient_check(f, x)


@pytest.mark.parametrize(
    argnames=("data", "f"),
    argvalues=[
        (np.random.randn(5), lambda x: x.sum()),
        (np.random.randn(4, 1, 6, 7), lambda x: x.sum(axis=(-1, -2))),
        (np.random.randn(5, 3), lambda x: x.sum(axis=1)),
        (np.random.randn(5, 3), lambda x: x.sum(axis=-1)),
        (np.random.randn(3, 3, 3, 3), lambda x: x.sum(axis=(2, 1))),
    ],
)
def test_sum(data: NDArray, f: CallableV):
    x = variable(data)
    gradient_check(f, x)


@pytest.mark.parametrize(
    argnames=("data0", "data1", "f"),
    argvalues=[
        (np.random.randn(1), np.random.randn(4), lambda x, y: x + y),
        (np.random.randn(2, 2), np.random.randn(2), lambda x, y: x - y),
        (np.random.randn(3), np.random.randn(), lambda x, y: x * y),
        (np.random.randn(2, 1), np.random.randn(), lambda x, y: x * y),
    ],
    ids=["add", "sub", "mul", "div"],
)
def test_broadcasting(data0: NDArray, data1: NDArray, f: CallableV):
    x0 = variable(data0)
    x1 = variable(data1)
    gradient_check(f, x1, x0)
    x0.cleargrad()
    x1.cleargrad()
    gradient_check(f, x0, x1)


def test_matmul():
    data_x = np.random.randn(3, 4)
    data_W = np.random.randn(4, 4)
    x = variable(data_x)
    W = variable(data_W)
    gradient_check(lambda _x: F.matmul(_x, W), x)
    x.cleargrad()
    W.cleargrad()
    gradient_check(lambda _W: F.matmul(x, _W), W)


@pytest.mark.parametrize(
    argnames=("data0", "data1"),
    argvalues=[
        (np.random.randn(5), np.random.rand(5)),
        (np.random.randn(2, 2), np.random.rand(2)),
        (np.random.randn(), np.random.rand()),
    ],
    ids=["normal", "broadcast", "ndim=0"],
)
def test_mean_squared_error(data0: NDArray, data1: NDArray):
    x0 = variable(data0)
    x1 = variable(data1)
    f = F.mean_squared_error
    gradient_check(f, x1, x0)
    x0.cleargrad()
    x1.cleargrad()
    gradient_check(f, x0, x1)


def cleargrads(*args: Variable | None):
    for arg in args:
        if arg:
            arg.cleargrad()


@pytest.mark.parametrize(
    "x, W, b",
    [
        (np.random.randn(1, 2), np.random.randn(2, 2), None),
        (np.random.randn(3, 2), np.random.randn(2, 2), np.random.randn(3, 2)),
        (np.random.randn(3, 2), np.random.randn(2, 2), np.random.randn(2)),
    ],
    ids=["b is None", "b is not broadcasted", "b is broadcasted"],
)
def test_linear(x, W, b):
    x, W = as_variables(x, W)
    b = None if b is None else variable(b)
    gradient_check(partial(F.linear, W=W, b=b), x)
    cleargrads(x, W, b)
    gradient_check(partial(F.linear, x, b=b), W)
    cleargrads(x, W, b)
    if b:
        gradient_check(partial(F.linear, x, W), b)
        cleargrads(x, W, b)


def sigmoid_simple(x):
    return 1 / (1 + F.exp(-x))


def test_sigmoid():
    xdata = np.random.randn(2, 3)
    x = variable(xdata)
    gradient_check(F.sigmoid, x)
    x.cleargrad()

    y = F.sigmoid(x)
    y.backward()
    gx = x.grad
    assert gx is not None
    gx_data = gx.data
    x.cleargrad()

    y_s = sigmoid_simple(x)
    y_s.backward()
    gx_s = x.grad
    assert gx_s is not None
    gx_s_data = gx_s.data
    np.testing.assert_allclose(gx_data, gx_s_data)
