from collections.abc import Callable
from dezero.core import Function, Variable
import dezero.core as dc
import dezero.functions as F
import sympy as sp
import heapq

FuncTable = dict[type[Function], Callable[..., sp.Expr]]
SYMPY: FuncTable = {
    F.Square: lambda f, x: x**2,
    F.Exp: lambda f, x: sp.exp(x),
    # specials
    F.Add: lambda f, x, y: x + y,
    F.Sub: lambda f, x, y: x - y,
    F.Mul: lambda f, x, y: x * y,
    F.Div: lambda f, x, y: x / y,
    F.Neg: lambda f, x: -x,
    F.Pow: lambda f, x: x**f.c,
    F.Sin: lambda f, x: sp.sin(x),
    F.Cos: lambda f, x: sp.cos(x),
}


def symbol_leaf(v: Variable):
    if v.name and v.name != "":
        name = v.name
    else:
        name = f"{v.data}"
    return sp.Symbol(name)


def symbol_branch(f: Function, *args: sp.Expr, functable: FuncTable = {}):
    ftype = type(f)
    if ftype in functable:
        func = functable[ftype]
        if not callable(func):
            raise ValueError(f"{ftype} key has not callable {func} value")
        output = func(f, *args)
        if not isinstance(output, sp.Expr):
            raise TypeError(f"f: {func} should return {sp.Expr}")
        return output
    if ftype in SYMPY:
        func = SYMPY[ftype]
        output = func(f, *args)
        return output

    return sp.Function(ftype.__name__)(*args)


def to_expr(output: Variable, functable: FuncTable = {}):
    # collect tractable from outputs.
    # pushs greater generation first
    seen, vars = set[int](), list[Variable]()
    heap = [(-output.generation, id(output), output)]
    while heap:
        _, _, v = heapq.heappop(heap)
        if id(v) in seen:
            continue
        seen.add(id(v))
        vars.append(v)
        if v.creator:
            for input in v.creator.inputs:
                heapq.heappush(heap, (-v.generation, id(input), input))

    expr: dict[int, sp.Expr] = {}
    # pops lesser generation first
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
    return sp.latex(to_expr(v, functable))
