from __future__ import annotations

import contextlib
import heapq
import weakref
from abc import ABC
from collections.abc import Generator, Sequence
from itertools import count
from typing import Any, Self, SupportsIndex, TypeVar, overload, override

import numpy as np

from dezero.typing import NDArray, Shape, ShapeLike, dtype, xp


class Config:
    enable_backprop = True
    detect_anomaly = False


# Helper
def asarray(x: Any) -> NDArray:
    """
    Args:
        x: A value tranformed into NDArray. This does not create new NDArray if x is NDArray or Variable
    Returns:
        NDArray which Variable can hold.
    Raises:
        ValueError: x contains infs or NaNs.
        TypeError: x's dtype is not subtype of xp.number
    """
    if isinstance(x, Variable):
        return x.data
    try:
        if not Config.detect_anomaly:
            x: NDArray = xp.asarray(x)
        else:
            x: NDArray = xp.asarray_chkfinite(x)
    except ValueError:
        raise ValueError("array must not contain infs or NaNs")
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError(f"{x.dtype} is not allowed. Subclass of {np.number} allowed.")
    return np.astype(x, dtype)


def to_str(obj) -> str:
    fields = []
    for key, val in vars(obj).items():
        val_repr = repr(val)
        fields.append(f"{key}={val_repr}")
    fields_str = ", ".join(fields)
    return f"{type(obj)}{{{fields_str}}}"


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

    def __init__(self, data: Any, name: str | None = None):
        self.data: NDArray = asarray(data)
        self.grad: Variable | None = None
        self.creator: "Function | None" = None
        self.generation: int = 0
        self.name: str | None = name

    def set_creater(self, func: "Function | None"):
        self.creator = func
        # creator's successer variable (this) is new generation
        self.generation = func.generation + 1 if func is not None else 0

    def backward(self, retain_grad: bool = False, create_graph: bool = False):
        # Complexity should exist somewhere. Choose where.
        # Hence, choose here.
        if self.grad is None:
            self.grad = as_variable(xp.ones_like(self.data))
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

            def gen_outputs_grad(f: "Function") -> Generator[Variable]:
                for output in f.outputs:
                    output = output()
                    if output is None:
                        raise AssertionError("Function output should not contain None")
                    if output.grad is None:
                        raise AssertionError(
                            "Function outputs should not contain None grad when backwarding"
                        )
                    yield output.grad

            gys = tuple(gen_outputs_grad(f))
            # backward and x.grad + gx create graph.
            with using_config("enable_backprop", create_graph):
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
                        # z = f(y0(x), y1(x)) -> dz/dz = df/dy0 * dy0/x + df/dy1 * dy1/x
                    if x.creator:
                        # funcs never contains None!
                        push_func(x.creator)

            # all path node drops the grad
            if not retain_grad:
                for y_weak in f.outputs:
                    y = y_weak()
                    if y is not None:
                        y.grad = None

    def cleargrad(self):
        self.grad = None

    @property
    def shape(self) -> Shape:
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
        data_repr = repr(self.data).replace("\n", "\n" + " " * 9)
        grad_repr = repr(self.grad).replace("\n", "\n" + " " * 9)
        fields = []
        for key, val in vars(self).items():
            if key in ("data", "grad"):
                continue
            val_repr = repr(val)
            fields.append(f"{key}={val_repr}")
        fields_str = ",\n    ".join(fields)
        return f"""{type(self)}{{
    data={data_repr},
    grad={grad_repr},
    {fields_str}
}}"""

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

    @overload
    def reshape(self, shape: ShapeLike | None, /) -> Variable: ...
    @overload
    def reshape(self, /, *shape: SupportsIndex) -> Variable: ...
    def reshape(self, *shape: Any):
        import dezero.functions as F

        if len(shape) == 1 and isinstance(shape, Sequence):
            shape = shape[0]

        return F.reshape(self, shape)

    @overload
    def transpose(self, axes: ShapeLike | None, /) -> Variable: ...
    @overload
    def transpose(self, /, *axes: SupportsIndex) -> Variable: ...
    def transpose(self, *axes: Any):
        import dezero.functions as F

        if len(axes) == 1 and isinstance(axes, Sequence):
            axes = axes[0]
        if axes == ():
            return F.transpose(self)
        return F.transpose(self, axes)

    @property
    def T(self):
        return self.transpose()

    def sum(self, axis: None | ShapeLike = None, keepdims: bool = False) -> Variable:
        import dezero.functions as F

        return F.sum(self, axis, keepdims)


def variable(x: Any, name=None) -> Variable:
    return Variable(x, name)


# Abstract Function
class Function(ABC):
    """
    Base Function that represents Variables to Variables function.

    For Users of this class:
        forward: NDArrays -> NDArray
        backward: Variables -> Variable
    """

    inputs: tuple[Variable, ...]
    outputs: tuple[weakref.ReferenceType[Variable], ...]
    generation: int

    def __call__(self, *inputs: Variable):
        """
        Args:
            inputs: Variables

        Returns:
            Variable if the function returns single output, else returns tuple of Variable

        Intentionally weakly typed because its return type depends on the runtime subclass implementation.
        Public APIs recover precise types with cast or wrapper functions.
        """
        xs = tuple(input.data for input in inputs)
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs: tuple[Variable, ...] = tuple(as_variables(*ys))

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

    @overload
    def forward(self, x) -> Any: ...

    """
    Foward is not strictly typed for the users's flexiblity of Function subclass
    Prefer users to returns NDArray or list of NDarrays not None.
    """

    def backward(self, gy) -> Any: ...


def as_variable(obj: Any) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_variables(*objs: Any) -> Generator[Variable]:
    """
    Cautious:
        Be careful when passing a `Generator` around using `*args`;
        since `Generator` is single-use,
        they get consumed, which can lead to unintended bugs.
        Get generators consumed by `tuple(generator)` to use with `*args`.
    """
    for obj in objs:
        yield as_variable(obj)


# Concrete Functions
class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        (x,) = self.inputs
        gx = 2.0 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = xp.exp(x)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        gx = exp(x) * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = gy
        if self.x0_shape != self.x1_shape:
            import dezero.functions as F

            gx0 = F.sum_to(gx0, self.x0_shape)
            gx1 = F.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy: Variable):
        x0, x1 = self.inputs
        # y = x0 * x1 => dy/dx0 = x1, dy/dx1 = x0
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            import dezero.functions as F

            gx0 = F.sum_to(gx0, x0.shape)
            gx1 = F.sum_to(gx1, x1.shape)

        return gx0, gx1


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0: NDArray, x1: NDArray):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        return x0 - x1

    def backward(self, gy: Variable):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:
            import dezero.functions as F

            gx0 = F.sum_to(gx0, self.x0_shape)
            gx1 = F.sum_to(gx1, self.x1_shape)

        return gx0, gx1


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        if x0.shape != x1.shape:
            import dezero.functions as F

            gx0 = F.sum_to(gx0, x0.shape)
            gx1 = F.sum_to(gx1, x1.shape)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x**self.c

    def backward(self, gy):
        # y = x ** c, dy/dx = c * x**(c-1)
        (x,) = self.inputs
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


class Parameter(Variable):
    pass
