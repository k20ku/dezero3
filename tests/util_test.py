import numpy as np
import pytest

from dezero.core import Variable, add, div, exp, square, as_variable, as_variables
from dezero.util import gradient_check
from dezero.utils.grad import numerical_grad
from dezero.utils.array import sum_to


def test_gradient_check():
    def f(x: Variable) -> Variable:
        return exp(x)

    x = np.array([[1.0, 0.0], [4.0, -1]])
    gradient_check(f, x)
