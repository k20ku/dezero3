from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator

import numpy as np

type NDArray = np.ndarray

ndarray = np.ndarray


# Helper
def as_array(x) -> NDArray:
    if isinstance(x, ndarray):
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
        self.grad: NDArray | None = None
        self.creator: "Function" | None = None

    def set_creater(self, func: "Function" | None):
        self.creator = func

    def backward(self):
        # Complexity should exist somewhere. Choose where.
        # Hence, choose here.
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        # graph contains only one Variable node
        if self.creator is None:
            return
        # func stack (never contain None!)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()

            gys = (output.grad for output in f.outputs)
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx
                if x.creator is not None:
                    # funcs never contains None!
                    funcs.append(x.creator)

    def __str__(self):
        return to_str(self)


# Abstract Function
class Function(ABC):
    inputs: tuple[Variable, ...]
    outputs: tuple[Variable, ...]

    def __call__(self, *inputs: Variable) -> Any:
        """
        Args:
            inputs: Variables

        Returns:
            output Variable if the function returns single output, else returns tuple of Variable

        Intentionally weakly typed because its return type depends on the runtime subclass implementation.
        Public APIs recover precise types with cast or wrapper functions.
        """
        xs = (input.data for input in inputs)
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = tuple(as_variable(y) for y in ys)

        for output in outputs:
            output.set_creater(self)  # output memorize this as the parent

        self.inputs = inputs  # memorize input variable
        self.outputs = outputs  # momorize output variable
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, x) -> Any: ...

    @abstractmethod
    def backward(self, gy) -> Any: ...

    def _backward_safely(self, gy) -> NDArray:
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


def as_variables(*objs: Any) -> Generator[Variable]:
    for obj in objs:
        yield as_variable(obj)


# Concrete Functions
class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2.0 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


def add(x0: Variable, x1: Variable) -> tuple[Variable, Variable]:
    return Add()(x0, x1)
