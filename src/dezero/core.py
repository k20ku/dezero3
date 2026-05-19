from __future__ import annotations

from typing import Any, Protocol

import numpy as np

array_types = np.ndarray


# Helper
def as_array(x) -> array_types:
    if isinstance(x, array_types):
        return x
    if np.isscalar(x):
        return np.array(x)
    raise TypeError(f"{type(x)} is not allowed for Variable")


def to_str(obj) -> str:
    fields = []
    for key, val in vars(obj).items():
        val_repr = repr(val)
        val_type = type(val).__name__
        fields.append(f"{key}=({val_repr})({val_type})")
    fields_str = ",".join(fields)
    return f"{obj.__class__}({fields_str})"


# Variable
class Variable:
    def __init__(self, data: Any):
        self.data = as_array(data)
        self.grad: np.ndarray | None = None
        self.creator: "Function" | None = None

    def set_creater(self, func: "Function" | None):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        # graph contains only one Variable node
        if self.creator is None:
            return
        # func stack (never contain None!)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()

            x, y = f.input, f.output
            x.grad = f._backward_safely(y.grad)

            if x.creator is not None:
                # funcs never contains None!
                funcs.append(x.creator)

    def __str__(self):
        return to_str(self)


# Abstract Function
class Function(Protocol):
    input: Variable
    output: Variable

    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)

        output: Variable = Variable(y)
        output.set_creater(self)  # output memorize this as the parent

        self.input = input  # memorize input variable
        self.output = output  # momorize output variable
        return output

    def forward(self, x) -> Any: ...

    def backward(self, gy) -> Any: ...

    def _backward_safely(self, gy) -> np.ndarray:
        if gy is None:
            raise TypeError("gradient must not be None.")
        gx = self.backward(gy)
        gx = as_array(gx)

        return gx

    def __str__(self):
        return to_str(self)


def as_variable(obj: Any) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


# Concrete Functions
class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2.0 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)
