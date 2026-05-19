from typing import Callable, Concatenate, ParamSpec, TypeVar

import numpy as np
from numpy import ndarray

from dezero.core import Variable, as_variable

Param = ParamSpec("Param")

""" return value is Variable or its subclass"""
Variable_T = TypeVar("Variable_T", bound=Variable)


def gradient_check(
    f: Callable[Concatenate[Variable, Param], Variable_T],
    x: ndarray | Variable,
    rtol=1e-4,
    atol=1e-5,
    *args: Param.args,
    **kwargs: Param.kwargs,
) -> bool:
    """Test backward procedure of a given function.

    This automatically checks the backward-process of a given function. For
    checking the correctness, this function compares gradients by
    backprop and ones by numerical derivation. If the result is within a
    tolerance this function return True, otherwise False.

    Params:
        f (callable): A function which gets `Variable`(s) and returns `Variable`(s).
        x (`ndarray` or `dezero.Variable`): A traget `Variable` for computing
            the gradient.
        *args: If `f` needs variables except `x`, you can specify with this
            argument.
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.
        **kwargs: If `f` needs keyword variables, you can specify with this
            argument.

    Returns:
        bool: Return True if the result is within a tolerance, otherwise False.
    """
    # TODO:accepting multiple 'Variable's
    var_x = as_variable(x)
    var_x.data = var_x.data.astype(np.float64)
    num_grad = numerical_grad(f, x, *args, **kwargs)
    y = f(var_x, *args, **kwargs)
    y.backward()
    assert var_x.grad is not None
    bp_grad = var_x.grad

    assert bp_grad.shape == num_grad.shape
    res = array_allclose(num_grad, bp_grad, atol=atol, rtol=rtol)

    if not res:
        print("")
        print("========== FAILED (Gradient Check) ==========")
        print("Numerical Grad")
        print(" shape: {}".format(num_grad.shape))
        val = str(num_grad.flatten()[:10])
        print(" values: {} ...".format(val[1:-1]))
        print("Backprop Grad")
        print(" shape: {}".format(bp_grad.shape))
        val = str(bp_grad.flatten()[:10])
        print(" values: {} ...".format(val[1:-1]))
    return res


def numerical_grad(
    f: Callable[Concatenate[Variable, Param], Variable],
    x: ndarray | Variable,
    eps=1e-4,
    *args: Param.args,
    **kwargs: Param.kwargs,
) -> ndarray:
    """Computes numerical gradient by finite differences.

    ## Example
    ```
    def test_backward3(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        self.assertTrue(gradient_check(F.add, x, y))
    def add(x0, x1):
        x1 = as_array(
            x1,
            dezero.cuda.get_array_module(x0.data)
        )
        return Add()(x0, x1)
    ```
    Args:
        f (callable): A function which gets `Variable`s and returns `Variable`s.
        x (`ndarray` or `dezero.Variable`): A target `Variable` for computing
            the gradient.
        eps (float): The diff
        *args: If `f` needs variables except `x`, you can specify with this
            argument.,
        **kwargs: If `f` needs keyword variables, you can specify with this
            argument.

    Returns:
        `ndarray`: Gradient.
    """

    # TODO:accepting multiple 'Variable's
    # このVariableを使い倒す
    x_var = x if isinstance(x, Variable) else as_variable(x)
    # TODO: cuda extentions in future
    # xp = cuda.get_array_module(x)
    # if xp is not np:
    #     np_x = cuda.as_numpy(x)
    # else:
    #     np_x = x
    grad = np.zeros_like(x_var.data)

    # C低レイヤでndarrayを直接メモリ操作
    it = np.nditer(x_var.data, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x_var.data[idx].copy()

        x_var.data[idx] = tmp_val + eps
        y1: Variable = f(x_var, *args, **kwargs)  # f(x+h)
        _y1 = y1.data.copy()

        x_var.data[idx] = tmp_val - eps
        y2 = f(x_var, *args, **kwargs)  # f(x-h)
        _y2 = y2.data.copy()

        diff = (_y1 - _y2).sum()
        grad[idx] = diff / (2 * eps)

        x_var.data[idx] = tmp_val
        it.iternext()
    return grad


def array_allclose(
    a: ndarray | Variable, b: ndarray | Variable, rtol=1e-4, atol=1e-5
) -> bool:
    """Returns True if two arrays(or variables) are element-wise equal within a
    tolerance.

    Args:
        a, b (numpy.ndarray or cupy.ndarray or dezero.Variable): input arrays
            to compare
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.

    Returns:
        bool: True if the two arrays are equal within the given tolerance,
            False otherwise.
    """
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b
    # TODO: cuda extentions
    # a, b = cuda.as_numpy(a), cuda.as_numpy(b)
    return np.allclose(a, b, atol=atol, rtol=rtol)
