from collections.abc import Sequence
from typing import overload

from dezero.core import Variable, as_variable
from dezero.typing import NDArray, Shape, ShapeLike


@overload
def sum_to(x: NDArray, shape: Shape) -> NDArray: ...
@overload
def sum_to(x: Variable, shape: Shape) -> Variable: ...
def sum_to(x: NDArray | Variable, shape: Shape) -> NDArray | Variable:
    """
    original: https://github.com/chainer/chainer/blob/v7.8.1/chainer/utils/array.py
    """
    if x.shape == shape:
        return x

    if isinstance(x, Variable):
        x = x.data
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y: NDArray = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)

    if isinstance(x, Variable):
        return as_variable(y)
    return y


@overload
def reshape_for_sum_backward(
    gy: NDArray, x_shape: Shape, axis, keepdims: bool
) -> NDArray: ...
@overload
def reshape_for_sum_backward(
    gy: Variable, x_shape: Shape, axis, keepdims: bool
) -> Variable: ...
def reshape_for_sum_backward(
    gy: NDArray | Variable, x_shape: Shape, axis: None | ShapeLike, keepdims: bool
):
    """Reshape for compute gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (NDArray | Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        NDArray | Variable: reshaped gradient variable for gradient appropriately for dezero.functions.sum's backward
    """
    ndim = len(x_shape)
    if axis is None:
        sequenced_axis = None
    elif isinstance(axis, Sequence):
        sequenced_axis = axis
    else:
        sequenced_axis = (axis,)

    if not (ndim == 0 or sequenced_axis is None or keepdims):
        # ex) ndim = 4, axis = (-1, -3)
        actual_axis = [int(a) if int(a) >= 0 else int(a) + ndim for a in sequenced_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
        shape = tuple(s for s in shape)
    else:
        shape = gy.shape

    return gy.reshape(shape)
