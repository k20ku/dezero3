import heapq
import os
import tempfile
import typing
from functools import partial
from pathlib import Path, PurePath

import graphviz

from dezero.core import Function, Variable

Vcolor = "orange"
Fcolor = "lightblue"


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


def create_graph(output: Variable, verbose=True, vcolor=Vcolor, fcolor=Fcolor):
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
    vcolor=Vcolor,
    fcolor=Fcolor,
) -> Path:
    dot = create_graph(x, verbose=verbose, fcolor=fcolor, vcolor=vcolor)
    filename = f"{fname}.gv"
    return Path(dot.save(filename=filename, directory=dir))

def render(
    x: Variable,
    dir: os.PathLike | str,
    fname: os.PathLike | str,
    format: str | None,
    cleanup=True,
    verbose=True,
    vcolor=Vcolor,
    fcolor=Fcolor,
) -> Path:
    """
    Example:
        >>> import numpy as np
        >>> import dezero
        >>> x = dezero.Variable(np.random.randn(2,3), dtype=np.float32, name='x')
        >>> y = (x + 1) ** 2
        >>> from pathlib import Path
        >>> render(y, Path.home(), "quad", "svg", cleanup=True)
        PosixPath('home/[user]/quad.svg')
        >>> render(y, Path.home()/"Downloads", "quad.svg")
        PosixPath('home/[user]/Downloads/quad.svg')
    """
    dot = create_graph(x, verbose=verbose, fcolor=fcolor, vcolor=vcolor)
    dir = Path(dir)
    fname = PurePath(fname)
    dir.mkdir(parents=True, exist_ok=True)

    dot.directory = dir.__str__()
    dot.filename = fname.with_suffix(".gv").__str__()

    if format is None:
        outfile = dir / fname
    else:
        outfile = dir / fname.with_suffix(f".{format}")
    path = dot.render(outfile=outfile, cleanup=cleanup, format=format)
    return Path(path)

save_image = render

def save_svg(
    x: Variable,
    dir: os.PathLike | str,
    fname: os.PathLike | str,
    cleanup=True,
    verbose=True,
    vcolor=Vcolor,
    fcolor=Fcolor,
) -> Path:
    path = render(x, dir, fname, "svg")
    return path
