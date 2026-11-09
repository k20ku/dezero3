import dezero.core as dc
import dezero.utils.array as darray
from dezero.core import Function, Variable
from dezero.typing import NDArray, Shape, ShapeLike, xp

Exp = dc.Exp
Square = dc.Square

Add = dc.Add
Sub = dc.Sub
Mul = dc.Mul
Div = dc.Div
Neg = dc.Neg
Pow = dc.Pow
exp = dc.exp
square = dc.square


class Sin(Function):
    def forward(self, x):
        y = xp.sin(x)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        return gy * cos(x)


class Cos(Function):
    def forward(self, x):
        y = xp.cos(x)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        return -gy * sin(x)


class Tanh(Function):
    def forward(self, x):
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y_weak = self.outputs[0]
        y = y_weak()
        if y is None:
            raise AssertionError(f"{self} has None output")
        return gy * (1 - y**2)


class Reshape(Function):
    def __init__(self, shape: int | tuple[int]):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = xp.reshape(x, self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


class Transpose(Function):
    def __init__(self, axes: tuple[int] | None = None):
        self.axes = axes

    def forward(self, x):
        y = xp.transpose(x, self.axes)
        return y

    def backward(self, gy):
        # transpose needs information on x.ndim. however
        if self.axes is None:
            return transpose(gy)
        axes_inv = tuple(xp.argsort(self.axes))
        gx = transpose(gy, axes=axes_inv)
        return gx


class Sum(Function):
    def __init__(self, axis: None | ShapeLike, keepdims: bool):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: NDArray):
        self.x_shape = x.shape
        y = xp.sum(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy: Variable):
        gy = darray.reshape_for_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


class BroadcastTo(Function):
    def __init__(self, shape: Shape):
        self.shape = shape

    def forward(self, x: NDArray):
        self.x_shape = x.shape
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = darray.sum_to(gy, self.x_shape)
        return gx


class SumTo(Function):
    def __init__(self, shape: Shape):
        self.shape = shape

    def forward(self, x: NDArray):
        self.x_shape = x.shape
        y = darray.sum_to(x, self.shape)
        return y

    def backward(self, gy: Variable):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sin(x: Variable) -> Variable:
    return Sin()(x)


def cos(x: Variable) -> Variable:
    return Cos()(x)


def tanh(x: Variable) -> Variable:
    return Tanh()(x)


def reshape(x: Variable, shape: int | tuple[int]) -> Variable:
    if x.shape == shape:
        return x
    return Reshape(shape)(x)


def transpose(x: Variable, axes: None | tuple[int] = None) -> Variable:
    return Transpose(axes)(x)


def sum(x: Variable, axis: None | ShapeLike = None, keepdims: bool = False) -> Variable:
    return Sum(axis, keepdims)(x)


def broadcast_to(x: Variable, shape: Shape) -> Variable:
    return BroadcastTo(shape)(x)


def sum_to(x: Variable, shape: Shape) -> Variable:
    """
    Sum of elements in x in the shape of given shape.
    sum function that can be designated returning shape
    """
    return SumTo(shape)(x)
