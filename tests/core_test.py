from typing import Any, Callable, cast

import numpy as np
import pytest

from dezero.core import Variable, add, as_variable, as_variables, exp, square


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
        ("x0", "expected", "assert_func"),
        [
            (np.array(3.0), np.array(6.0), np.testing.assert_equal),
        ],
    )
    def test_backward(self, x0, expected, assert_func):
        x = as_variable(x0)
        y = square(x)
        y.backward()
        assert x.grad is not None
        assert_func(x.grad.data, expected)


class TestAdd:
    @pytest.mark.parametrize(
        ("x0", "x1", "expected", "assert_func"),
        [
            (np.array(2.0), np.array(3.0), np.array(5.0), np.testing.assert_equal),
        ],
    )
    def test_forward(self, x0, x1, expected, assert_func):
        x0 = as_variable(x0)
        x1 = as_variable(x1)
        y = add(x0, x1)
        assert_func(y.data, expected)

    @pytest.mark.parametrize(
        ("x0", "x1", "expected", "assert_func"),
        [
            (np.array(3.0), np.array(6.0), np.array(1.0), np.testing.assert_equal),
        ],
    )
    def test_backward(self, x0, x1, expected, assert_func):
        x0 = as_variable(x0)
        x1 = as_variable(x1)
        y = add(x0, x1)
        y.backward()
        assert x0.grad is not None
        assert x1.grad is not None
        assert_func(x0.grad.data, expected)
        assert_func(x1.grad.data, expected)


class TestExp:
    @pytest.mark.parametrize(
        ("x", "expected", "assert_func"),
        [
            (np.array([0.0, 1.0]), np.array([1.0, np.e]), np.testing.assert_equal),
        ],
    )
    def test_forward(self, x, expected, assert_func):
        x = as_variable(x)
        y = exp(x)
        assert_func(y.data, expected)

    @pytest.mark.parametrize(
        ("x", "expected", "assert_func"),
        [
            (np.array(3.0), np.exp(3.0), np.testing.assert_equal),
            (
                np.array([[1, 3.1], [- 4, 0]]),
                np.exp(np.array([[1, 3.1], [- 4, 0]])),
                np.testing.assert_equal,
            ),
        ],
    )
    def test_backward(self, x, expected, assert_func):
        x = Variable(x)
        y = exp(x)
        y.backward()
        assert x.grad is not None
        assert_func(x.grad.data, expected)


def quad_two_times(x: Variable):
    a = square(x)
    y = add(square(a), square(a))
    return y


@pytest.mark.parametrize(
    ("args", "func", "fwants", "bwants"),
    [
        (
            ([2.0, 3.0], [3.0, 4.0]),
            lambda x, y: add(square(x), square(y)),
            [13.0, 25.0],
            ([4.0, 6.0], [6.0, 8.0]),
        ),
        (
            [[2.0, 3.0], [-1.0, 0.0]],
            lambda x: quad_two_times(x),  # d/dx(2*x**4) = 8*x**3
            2 * np.array([[2.0, 3.0], [-1.0, 0.0]]) ** 4,
            8 * np.array([[2.0, 3.0], [-1.0, 0.0]]) ** 3,
        ),
        (
            ([-1.0, 23.0], [-3.0, 4.0]),
            lambda x, y: x + y,
            [-4.0, 27.0],
            ([1.0, 1.0], [1.0, 1.0]),
        ),
        (
            ([32.1, -2.0], [3.0, 111.0]),
            lambda x, y: x * y,
            [96.3, -222.0],
            ([3.0, 111.0], [32.1, -2.0]),
        ),
    ],
)
def test_for_back(args: tuple[Any], func: Callable[..., Variable], fwants, bwants):
    if not isinstance(args, tuple):
        args = (args,)
    xs = as_variables(*args)
    ys = func(*xs)

    if not isinstance(ys, tuple):
        ys = (ys,)
    if not isinstance(fwants, tuple):
        fwants = (fwants,)

    for y, fwant in zip(ys, fwants):
        assert y is not None
        np.testing.assert_allclose(y.data, fwant)

    # when backward not implimented
    if bwants is None:
        return

    if not isinstance(bwants, tuple):
        bwants = (bwants,)

    if len(args) != len(bwants):
        pytest.fail("length of args must equals to that of bwants")

    # とりあえず一変数のみで
    # 多変数の出力だとbackwardしたときにかぶったりするので保留
    if not isinstance(ys[0], Variable):
        pytest.fail("func must return an instance of Variable or that of tuple")

    y = cast(Variable, ys[0])
    y.backward()

    for x, bwant in zip(xs, bwants):
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.data, bwant)


def test_shared_variable():
    x = as_variable(3)
    y = add(x, x)
    np.testing.assert_equal(y.data, 2 * x.data)
    y.backward()
    assert x.grad is not None
    np.testing.assert_equal(x.grad.data, np.array(2))


def test_reused_variable():
    # first calculation
    x = as_variable(1)
    y = add(x, x)
    y.backward()

    # second
    x.cleargrad()
    y = add(add(x, x), x)
    y.backward()
    assert x.grad is not None
    np.testing.assert_equal(x.grad.data, 3)


"""
TODO: test

Variable - Variable
int - Variable
float - Variable
NDArray - Variable
Variable - int
Variable - float
Variable -NDArray
int * Variable
NDArray * Variable
square(x0) = x0 ** 2
sqrt(x0) = x0 ** (1/2)

numerical_diff = backward
"""
