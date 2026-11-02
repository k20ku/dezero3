from typing import Callable, TypeVar, Any

import numpy as np
from numpy import ndarray

from dezero.core import Variable, as_variable, asarray
from dezero.utils.grad import numerical_grad

""" return value is Variable or its subclass"""
Variable_T = TypeVar("Variable_T", bound=Variable)


def gradient_check(
    f: Callable[..., Variable_T],
    x: ndarray | Variable,
    *args: Any,
    rtol: float =1e-4,
    atol: float =1e-5,
    **kwargs: Any,
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

    """
    x = as_variable(x)
    x.data = x.data.astype(np.float64)
    num_grad = numerical_grad(f, x, *args, **kwargs)
    y = f(x, *args, **kwargs)
    y.backward()
    assert x.grad is not None
    bp_grad = x.grad.data

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
    a = asarray(a)
    b = asarray(b)
    # TODO: cuda extentions
    # a, b = cuda.as_numpy(a), cuda.as_numpy(b)
    return np.allclose(a, b, atol=atol, rtol=rtol)
