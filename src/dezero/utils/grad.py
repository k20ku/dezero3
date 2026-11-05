from typing import Callable, TypeVar, Any

import numpy as np

from dezero.core import NDArray, Variable, asarray, as_variable

Variable_T = TypeVar("Variable_T", bound=Variable)


def numerical_grad(
    f: Callable[..., Variable_T],
    x: NDArray | Variable,
    *args: Any,
    eps: float = 1e-5,
    enable5p=False,
    **kwargs: Any,
) -> NDArray:
    """Computes numerical gradient by finite differences.

        Args:
            f (callable): A function which gets `Variable`s and returns `Variable`s.
            x (`ndarray` or `dezero.Variable`): A target `Variable` for computing
                the gradient.
            *args: If `f` needs variables except `x`, you can specify with this
                argument.,
            eps (float): The diff
            enable5p (bool): enable five-point stencil
            **kwargs: If `f` needs keyword variables, you can specify with this
                argument.

        Returns:
            `NDarray`: Gradient.


        ## Example

        ```
        x = as_variable(x)
        x.data = x.data.astype(np.float64)
        num_grad = numerical_grad(f, x, *args, **kwargs)
        y = f(x, *args, **kwargs)
        y.backward()
        assert x.grad is not None
        bp_grad = x.grad

        assert bp_grad.shape == num_grad.shape
        res = array_allclose(num_grad, bp_grad, atol=atol, rtol=rtol)
        ```

        ## Details
        Original implementation is
    https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/b5f3cf12a9e4ed84fc97b57bf41a25610d69c9e5/dezero/utils.py#L234

    """

    x: NDArray = asarray(x)

    # `f` of `_numerical_grad_nd` is NDArray to NDArray func,
    # but this func's `f` is Variable to Variable (because we observe auto-grad!).
    # so translation is needed. And fnd is that.
    def fnd(xdata: NDArray, *args, **kwargs) -> NDArray:
        y: Variable_T = f(as_variable(xdata), *args, **kwargs)
        return y.data

    if enable5p:
        grad = _numerical_grad_nd_5pstencil(fnd, x, eps=eps, *args, **kwargs)
        return grad

    grad = _numerical_grad_nd(fnd, x, eps=eps, *args, **kwargs)
    return grad


def _numerical_grad_nd(
    f: Callable[..., NDArray],
    x: NDArray,
    *args: Any,
    eps: float = 1e-5,
    **kwargs: Any,
) -> NDArray:
    x = np.astype(x, np.float64, copy=True)
    grad = np.zeros_like(x)
    # Use ndindex to iterate over all elements in x.
    # It is very desiable to be this func implemented as
    # `simple_grad = lambda f, x: (f(x + eps) - f(x - eps))/ (2 * eps)`
    # However this does not count where x[idx0] and x[idx1] interacts,
    # such as reshape and sum.
    # Numerical grad of each element can be calculated as total differences
    # resulted from that element's aviation divided by that element's aviation.
    for idx in np.ndindex(x.shape):
        tmp_x = x[idx]

        x[idx] = tmp_x + eps  # x+e
        y1 = f(x, *args, **kwargs)  # f(x+e)

        x[idx] = tmp_x - eps
        y0 = f(x, *args, **kwargs)

        # total diffs resulted by eps diff of x[idx]
        diff = np.sum(y1 - y0)
        grad[idx] = diff / (2 * eps)

        x[idx] = tmp_x

    return grad


def _numerical_grad_nd_5pstencil(
    f: Callable[..., NDArray],
    x: NDArray,
    *args: Any,
    eps: float = 1e-5,
    **kwargs: Any,
) -> NDArray:
    """
    https://en.wikipedia.org/wiki/Five-point_stencil
    """
    x = np.astype(x, np.float64, copy=True)
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        tmp_x = x[idx]

        x[idx] = tmp_x + eps  # x+e
        y4 = f(x, *args, **kwargs)  # f(x+e)

        x[idx] = tmp_x - eps
        y3 = f(x, *args, **kwargs)

        x[idx] = tmp_x + 2 * eps  # x+e
        y2 = f(x, *args, **kwargs)  # f(x+e)

        x[idx] = tmp_x - 2 * eps
        y1 = f(x, *args, **kwargs)

        # total diffs resulted by eps diff of x[idx]
        diff = np.sum(8 * (y4 - y3) - (y2 - y1))
        grad[idx] = diff / (12 * eps)

        x[idx] = tmp_x

    return grad
