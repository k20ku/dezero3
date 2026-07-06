from __future__ import annotations

import contextlib
import heapq
import weakref
from abc import ABC, abstractmethod
from itertools import count
from typing import Any, Generator

import numpy as np

type NDArray = np.ndarray

xp = np

ndarray = xp.ndarray


# Helper
def as_array(x) -> NDArray:
    if isinstance(x, ndarray):
        return x
    if xp.isscalar(x):
        return xp.array(x)
    raise TypeError(f"{type(x)} is not allowed for Variable")


def to_str(obj) -> str:
    fields = []
    for key, val in vars(obj).items():
        val_repr = repr(val)
        val_type = type(val).__name__
        fields.append(f"{key}=({val_repr})({val_type})")
    fields_str = ",".join(fields)
    return f"{obj.__class__}({fields_str})"


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name: str, value):
    try:
        old_value = getattr(Config, name)
    except AttributeError as e:
        raise RuntimeError(f"failed to change config name {name!r}") from e
    else:
        try:
            setattr(Config, name, value)
            yield
        finally:
            setattr(Config, name, old_value)


def no_grad():
    """
    Example:
    ```
    with no_grad():
        x = Variable(2.0)
        y = square(x)
    ```
    """
    return using_config("enable_backprop", False)


# Variable
class Variable:
    def __init__(self, data: Any):
        self.data = as_array(data)
        self.grad: NDArray | None = None
        self.creator: "Function" | None = None
        self.generation: int = 0

    def set_creater(self, func: "Function" | None):
        self.creator = func
        # creator's successer variable (this) is new generation
        self.generation = func.generation + 1 if func is not None else 0

    def backward(self, retain_grad=False):
        # Complexity should exist somewhere. Choose where.
        # Hence, choose here.
        if self.grad is None:
            self.grad = xp.ones_like(self.data)
        # graph contains only one Variable node
        # creator
        if self.creator is None:
            return
        # func priority queue (never contain None!)
        funcs = []
        counter = count()  # to avoid heapq starting to compare Function and Function
        # when both two Functions are of the generation.

        def push_func(f: "Function"):
            # avoid repeated backward
            if not f.is_seen:
                # set -generation as a key and value is f
                heapq.heappush(funcs, (-f.generation, next(counter), f))
                f.is_seen = True

        def pop_func() -> "Function":
            # heappop pops smallest key element with O(log(n)) time.
            # We registered function's key as -f.generation,
            # so we can retrive a function with smallest -generation i.e. maximum generation
            _, _, f = heapq.heappop(funcs)
            return f

        push_func(self.creator)
        while funcs:
            f = pop_func()

            gys = (output().grad for output in f.outputs)
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:  # when backward not yet computed
                    x.grad = gx
                else:
                    # copy to avoid overwrite
                    # (if you give this x.grad as View to another Variable such as Reshape)
                    x.grad = x.grad + gx
                if x.creator is not None:
                    # funcs never contains None!
                    push_func(x.creator)

                # all path node drops the grad
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None

    def cleargrad(self):
        self.grad = None

    def __str__(self):
        return to_str(self)


# Abstract Function
class Function(ABC):
    inputs: tuple[Variable, ...]
    outputs: tuple[weakref.ReferenceType[Variable], ...]
    generation: int
    is_seen: bool = False

    def __call__(self, *inputs: Variable) -> Any:
        """
        Args:
            inputs: Variables

        Returns:
            Variable if the function returns single output, else returns tuple of Variable

        Intentionally weakly typed because its return type depends on the runtime subclass implementation.
        Public APIs recover precise types with cast or wrapper functions.
        """
        xs = (input.data for input in inputs)
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = tuple(as_variable(y) for y in ys)

        # processes for backprop
        if Config.enable_backprop:
            # maximum generation of inputs is the same as function's generation
            self.generation = max(x.generation for x in inputs)
            # make each output to memorize this as the parent
            for output in outputs:
                output.set_creater(self)

            self.inputs = inputs  # memorize input variable for backprop
            self.outputs = tuple(
                weakref.ref(output) for output in outputs
            )  # momorize output variable

        return outputs if len(outputs) > 1 else outputs[0]

    def __str__(self):
        return to_str(self)

    @abstractmethod
    def forward(self, x) -> Any: ...

    @abstractmethod
    def backward(self, gy) -> Any: ...


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
        y = xp.exp(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = xp.exp(x) * gy
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


def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)
