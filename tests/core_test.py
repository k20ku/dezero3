from typing import cast

import numpy as np
import pytest

from dezero.core import Variable, add, as_array, as_variable, as_variables, exp, square


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
        assert_func(x.grad, expected)


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
        assert_func(x0.grad, expected)
        assert_func(x1.grad, expected)


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


@pytest.mark.parametrize(
    ("args", "func", "fwants", "bwants"),
    [
        (
            (np.array([2.0, 3.0]), np.array([3.0, 4.0])),
            lambda x, y: add(square(x), square(y)),
            np.array([13.0, 25.0]),
            (np.array([4.0, 6.0]), np.array([6.0, 8.0])),
        ),
    ],
)
def test_forward(args, func, fwants, bwants):
    if not isinstance(args, tuple):
        args = (args,)
    xs = as_variables(*args)
    ys = func(*xs)

    if not isinstance(ys, tuple):
        ys = (ys,)
    if not isinstance(fwants, tuple):
        fwants = (fwants,)

    for y, fwant in zip(ys, fwants):
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
        np.testing.assert_allclose(x.grad, bwant)


def test_shared_variable():
    x = as_variable(3)
    y = add(x, x)
    np.testing.assert_equal(y.data, 2 * x.data)
    y.backward()
    np.testing.assert_equal(x.grad, np.array(2))


def test_reused_variable():
    # first calculation
    x = as_variable(1)
    y = add(x, x)
    y.backward()

    # second
    x.cleargrad()
    y = add(add(x, x), x)
    y.backward()
    np.testing.assert_equal(x.grad, 3)
