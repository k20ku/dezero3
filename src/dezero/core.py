from __future__ import annotations

import contextlib
import heapq
import weakref
from collections.abc import Generator
from abc import ABC
from itertools import count
from typing import Any, override

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
        fields.append(f"{key}={val_repr}")
    fields_str = ", ".join(fields)
    return f"{type(obj)}{{{fields_str}}}"

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
    """
    aaaaa
    """
    # sets enough high priority so that `NDArray [operator] Variable`
    # calls not the operator overriden by the NDArray but Variable.
    __array_priority__: int = 200

    def __init__(self, data: Any, name: str=''):
        self.data: NDArray = as_array(data)
        self.grad: NDArray | None = None
        self.creator: 'Function | None' = None
        self.generation: int = 0
        self.name: str = name

    def set_creater(self, func: 'Function | None'):
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
        # (generation, count, Function)
        funcs: list[tuple[int, int, "Function"]] = []
        # 二重登録を防ぐため
        seen_funcs = set["Function"]()
        counter = count()  # to avoid heapq starting to compare Function and Function
        # when both two Functions are of the generation.

        def push_func(f: "Function"):
            # avoid repeated backward
            if not f in seen_funcs:
                # set -generation as a key and value is f
                heapq.heappush(funcs, (-f.generation, next(counter), f))
                seen_funcs.add(f)

        def pop_func() -> "Function":
            # heappop pops smallest key element with O(log(n)) time.
            # We registered function's key as -f.generation,
            # so we can retrive a function with smallest -generation i.e. maximum generation
            _, _, f = heapq.heappop(funcs)
            return f

        push_func(self.creator)
        while funcs:
            f = pop_func()

            def gen_gys(f: "Function"):
                for output in f.outputs:
                    output = output()
                    if output is None:
                        raise AssertionError("Function output should not contain None")
                    yield output.grad

            gys = gen_gys(f)
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
                    for y_weak in f.outputs:
                        y = y_weak()
                        if y is None:
                            continue
                        y.grad = None

    def cleargrad(self):
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return self.data.__len__()

    @override
    def __repr__(self):
        return to_str(self)

    def __add__(self, other: Any):
        return add(self, other)
    def __radd__(self, other: Any):
        return add(self, other)
    def __mul__(self, other: Any):
        return mul(self, other)
    def __rmul__(self, other: Any):
        return mul(self, other)
    def __neg__(self):
        return neg(self)
    def __sub__(self, other: Any):
        return sub(self, other)
    def __rsub__(self, other: Any):
        return rsub(self, other)
    def __truediv__(self, other: Any):
        return div(self, other)
    def __rtruediv__(self, other: Any):
        return rdiv(self, other)
    def __pow__(self, c: Any):
        return pow(self, c)


# Abstract Function
class Function(ABC):
    """
    Base Function that represents Variables to Variables function.

    For Users of this class:
        forward: NDArrays -> NDArray
        backward: NDArrays -> NDArray
    """
    inputs: tuple[Variable, ...]
    outputs: tuple[weakref.ReferenceType[Variable], ...]
    generation: int

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
        outputs: tuple[Variable, ...] = tuple(as_variable(y) for y in ys)

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

        if len(outputs) == 0:
            raise AssertionError("Expected code to be unreachable")

        return outputs if len(outputs) > 1 else outputs[0]

    def __str__(self):
        return to_str(self)

    def forward(self, x) -> Any:...
    """
    Foward is not strictly typed for the users's flexiblity of Function subclass
    Prefer users to returns NDArray or list of NDarrays not None.
    """

    def backward(self, gy) -> Any:...


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

class Mul(Function):
    def forward(self, x0, x1) -> Any:
        y = x0 * x1
        return y

    def backward(self, gy) -> Any:
        # y = x0 * x1 => dy/dx0 = x1, dy/dx1 = x0
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, -gy

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        y0 = gy / x0
        y1 = gy * (- x0 / x1 ** 2)
        return y0, y1

class Pow(Function):
    def __init__(self, c):
        self.c = c
    def forward(self, x):
        return x ** self.c
    def backward(self, gy):
        # y = x ** c, dy/dx = c * x**(c-1)
        x = self.inputs[0].data
        c = self.c
        return c * x ** (c - 1) * gy

def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


def add(x0: Variable, x1: Any) -> Variable:
    _x1 = as_variable(x1)
    return Add()(x0, _x1)

def sub(x0: Variable, x1: Any) -> Variable:
    _x1 = as_variable(x1)
    return Sub()(x0, _x1)

def rsub(x0: Variable, x1: Any) -> Variable:
    """
    rsub(x, y) = y - x
    """
    _x1 = as_variable(x1)
    return Sub()(_x1, x0)

def mul(x0: Variable, x1: Any) -> Variable:
    _x1 = as_variable(x1)
    return Mul()(x0, _x1)

def neg(x: Variable) -> Variable:
    return Neg()(x)

def div(x0: Variable, x1: Any) -> Variable:
    _x1 = as_variable(x1)
    return Div()(x0, _x1)

def rdiv(x0: Variable, x1: Any) -> Variable:
    _x1 = as_variable(x1)
    return Div()(_x1, x0)

def pow(x: Variable, c: Any) -> Variable:
    return Pow(c)(x)


