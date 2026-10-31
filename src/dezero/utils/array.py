from dezero.core import NDArray


def sum_to(x: NDArray, shape: tuple[int,...]) -> NDArray:
    """
    original: https://github.com/chainer/chainer/blob/v7.8.1/chainer/utils/array.py
    """
    if x.shape == shape:
        return x
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y
