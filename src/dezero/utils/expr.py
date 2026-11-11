import heapq
import logging
from collections.abc import Callable

import sympy as sy

import dezero.functions as F
from dezero.core import Add, Div, Exp, Function, Mul, Neg, Pow, Square, Sub, Variable

logger = logging.getLogger(__name__)


class reshape(sy.Function):
    def _latex(self, printer):
        x, shape = (printer._print(a) for a in self.args)
        return r"%s.\operatorname{reshape}\left( %s \right)" % (x, shape)


class transpose(sy.Function):
    def __new__(cls, *args, axes=None, **options):
        if axes is None:
            return super().__new__(cls, *args, **options)
        return super().__new__(cls, *args, axes, **options)

    def _latex(self, printer):
        if len(self.args) == 1:
            x = printer._print(self.args[0])
            return r"%s^{\top}" % (x,)

        x, axes = (printer._print(a) for a in self.args)
        return r"%s.\operatorname{transpose}\left( %s \right)" % (x, axes)


class broadcast(sy.Function):
    def _latex(self, printer):
        x, shape = (printer._print(a) for a in self.args)
        return r"%s.\operatorname{broadcast}\left( %s \right)" % (x, shape)


FuncTable = dict[type[Function], Callable[..., sy.Expr]]
SYMPY: FuncTable = {
    Square: lambda f, x: x**2,
    Exp: lambda f, x: sy.exp(x),
    # syecials
    Add: lambda f, x, y: x + y,
    Sub: lambda f, x, y: x - y,
    Mul: lambda f, x, y: x * y,
    Div: lambda f, x, y: x / y,
    Neg: lambda f, x: -x,
    Pow: lambda f, x: x**f.c,
    F.Sin: lambda f, x: sy.sin(x),
    F.Cos: lambda f, x: sy.cos(x),
    F.Reshape: lambda f, x: reshape(x, f.shape),
    F.Transpose: lambda f, x: transpose(x, axes=f.axes),
    F.BroadcastTo: lambda f, x: broadcast(x, f.shape),
}


def symbol_leaf(v: Variable):
    if v.name:
        name = v.name
    else:
        if v.data.ndim <= 1:
            name = f"{v.data}"
        else:
            name = sy.latex(sy.Matrix(v.data))
    return sy.Symbol(name)


def symbol_branch(f: Function, *args: sy.Expr, functable: FuncTable = {}):
    ftype = type(f)
    if ftype in functable:
        func = functable[ftype]
        if not callable(func):
            raise ValueError(f"{ftype} key has not callable {func} value")
        output = func(f, *args)
        if not isinstance(output, sy.Expr):
            raise TypeError(f"f: {func} should return {sy.Expr}")
        return output
    if ftype in SYMPY:
        func = SYMPY[ftype]
        output = func(f, *args)
        return output

    return sy.Function(ftype.__name__)(*args)


def to_expr(output: Variable, functable: FuncTable = {}):
    # collect tractable from outputs.
    # pushs greater generation first
    seen, vars = set[int](), list[Variable]()
    heap = [(-output.generation, id(output), output)]
    logger.debug("start cmp-graph traversing")
    while heap:
        _, _, v = heapq.heappop(heap)
        if id(v) in seen:
            continue
        seen.add(id(v))
        vars.append(v)
        if v.creator:
            for input in v.creator.inputs:
                heapq.heappush(heap, (-v.generation, id(input), input))

    expr: dict[int, sy.Expr] = {}
    # pops lesser generation first
    logger.debug("strat to construct symbols")
    for v in reversed(vars):
        f = v.creator
        if f is None:
            expr[id(v)] = symbol_leaf(v)
        else:
            args = tuple(expr[id(input)] for input in f.inputs)
            expr[id(v)] = symbol_branch(f, *args, functable=functable)

    outid = id(output)
    return expr[outid]


def to_latex(v: Variable, functable: FuncTable = {}) -> str:
    """
    Args:
        output : Variable object
        functable: dict of Function type to to-sympy.Expr function.
    Returns:
        latex expression string of Variable
    Raises:
        ValueError: functable contains a callable value.
        TypeError: functable's values contains a function does not return sympy.Expr
    """
    logger.debug("start to convert to expr")
    e = to_expr(v, functable)
    logger.debug("to latex")
    return sy.latex(e)
