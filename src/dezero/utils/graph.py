from collections.abc import Generator
import heapq
from itertools import count

from dezero.core import Function, NDArray, Variable

import graphviz

def add_var(dot: graphviz.Digraph, v: Variable, verbose=False):
    name = v.name if v.name is not None else ''
    if verbose and v.data is not None:
        label = f"{name}: {v.shape} {v.dtype}"
    else:
        label = name
    dot.node(str(id(v)), label, color="orange", style="filled")

def add_func(dot: graphviz.Digraph, f: Function):
    dot.node(str(id(f)), label=f.__class__.__name__, 
            color="lightblue", style="filled", shape="box")
    for x in f.inputs:
        dot.edge(str(id(x)), str(id(f)))
    for y in f.outputs:
        dot.edge(str(id(f)), str(id(y())))

def graph(output: Variable, verbose=True):
    dot = graphviz.Digraph()
    add_var(dot, output, verbose)
    if output.creator is None:
        return dot
    funcs: list[tuple[int, int, "Function"]] = []
    seen_funcs = set["Function"]()
    counter = count()  # to avoid heapq starting to compare Function and Function

    def push_func(f: "Function"):
        if not f in seen_funcs:
            heapq.heappush(funcs, (-f.generation, next(counter), f))
            seen_funcs.add(f)

    def pop_func() -> "Function":
        _, _, f = heapq.heappop(funcs)
        return f

    push_func(output.creator)
    while funcs:
        f = pop_func()
        add_func(dot, f)
        for x in f.inputs:
            add_var(dot, x, verbose=verbose)
            if x.creator is not None:
                # funcs never contains None!
                push_func(x.creator)

    return dot

