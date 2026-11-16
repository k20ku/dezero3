import heapq
import os
import tempfile
import typing
from pathlib import Path

import graphviz

from dezero.core import Function, Variable

_vcolor = "orange"
_fcolor = "lightblue"


def add_var(
    dot: graphviz.Digraph,
    v: Variable,
    vcolor: str,
    verbose=False,
):
    name = v.name if v.name is not None else ""
    if verbose and v.data is not None:
        label = f"{name}: {v.shape} {v.dtype}"
    else:
        label = name
    dot.node(str(id(v)), label, color=vcolor, style="filled")


def add_func(dot: graphviz.Digraph, f: Function, fcolor: str):
    dot.node(
        str(id(f)),
        label=f.__class__.__name__,
        color=fcolor,
        style="filled",
        shape="box",
    )
    for x in f.inputs:
        dot.edge(str(id(x)), str(id(f)))
    for y in f.outputs:
        dot.edge(str(id(f)), str(id(y())))


def create_graph(output: Variable, verbose=True, vcolor=_vcolor, fcolor=_fcolor):
    dot = graphviz.Digraph()
    add_var(dot, output, vcolor=vcolor, verbose=verbose)

    if output.creator is None:
        return dot
    seen, heap = set[Function](), list[tuple[int, int, Function]]()

    def push_func(f: Function):
        if f not in seen:
            heapq.heappush(heap, (-f.generation, id(f), f))
            seen.add(f)

    def pop_func() -> Function:
        _, _, f = heapq.heappop(heap)
        return f

    push_func(output.creator)
    while heap:
        f = pop_func()
        add_func(dot, f, fcolor=fcolor)
        for x in f.inputs:
            add_var(dot, x, verbose=verbose, vcolor=vcolor)
            if x.creator:
                push_func(x.creator)

    return dot


def save_gv(
    x: Variable,
    dir: os.PathLike | str,
    fname: os.PathLike | str,
    verbose=True,
    vcolor=_vcolor,
    fcolor=_fcolor,
) -> Path:
    dot = create_graph(x, verbose=verbose, fcolor=fcolor, vcolor=vcolor)
    filename = f"{fname}.gv"
    return Path(dot.save(filename=filename, directory=dir))

def save_svg(
    x: Variable,
    dir: os.PathLike | str,
    fname: os.PathLike | str,
    verbose=True,
    vcolor=_vcolor,
    fcolor=_fcolor,
) -> Path:
    dot = create_graph(x, verbose=verbose, fcolor=fcolor, vcolor=vcolor)
    path = dot.render(filename=fname, directory=dir, format='svg')
    return Path(path)