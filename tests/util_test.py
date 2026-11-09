import numpy as np
import pytest

from dezero.core import Variable, exp
from dezero.util import gradient_check


def test_gradient_check():
    def f(x: Variable) -> Variable:
        return exp(x)

    x = np.array([[1.0, 0.0], [4.0, -1]])
    gradient_check(f, x)
