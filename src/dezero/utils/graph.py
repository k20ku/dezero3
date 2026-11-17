import heapq
import os
from collections.abc import Mapping
from pathlib import Path, PurePath
from typing import Unpack

import graphviz

from dezero.core import Function, Variable
from dezero.views import (
    EDGE_DEFAULT,
    FUNC_DEFAULT,
    GRAPH_DEFAULT,
    NODE_DEFAULT,
    VAR_DEFAULT,
    Attrs,
    GraphAttrs,
)


def add_var(dot: graphviz.Digraph, v: Variable, attr: Attrs, verbose=False):
    name = v.name if v.name is not None else ""
    if verbose and v.data is not None:
        label = f"{name}: {v.shape} {v.dtype}"
    else:
        label = name
    dot.node(str(id(v)), label, **attr)


def add_func(dot: graphviz.Digraph, f: Function, attr: Attrs):
    dot.node(name=str(id(f)), label=f.__class__.__name__, **attr)
    for x in f.inputs:
        dot.edge(str(id(x)), str(id(f)))
    for y in f.outputs:
        if (y := y()) is not None:
            dot.edge(str(id(f)), str(id(y)))


def create_graph(
    output: Variable,
    verbose=True,
    graph_attr: Attrs | None = None,
    node_attr: Attrs | None = None,
    edge_attr: Attrs | None = None,
    var_attr: Attrs | None = None,
    func_attr: Attrs | None = None,
) -> graphviz.Digraph:
    """Build a graphviz Digraph of the computational graph ending at `output`.

    graph_attr / node_attr / edge_attr are passed to graphviz.Digraph as is.
    var_attr / func_attr apply only to Variable / Function nodes and take
    precedence over node_attr. They are merged on top of VAR_DEFAULT /
    FUNC_DEFAULT, so only the keys you give are overridden.
    """
    graph_attr = {
        **GRAPH_DEFAULT,
        **(graph_attr or {}),
    }
    node_attr = {
        **NODE_DEFAULT,
        **(node_attr or {}),
    }
    edge_attr = {
        **EDGE_DEFAULT,
        **(edge_attr or {}),
    }
    var_attr = {
        **VAR_DEFAULT,
        **(var_attr or {}),
    }
    func_attr = {
        **FUNC_DEFAULT,
        **(func_attr or {}),
    }
    dot = graphviz.Digraph(
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
    )

    add_var(dot, output, var_attr, verbose=verbose)

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
        add_func(dot, f, func_attr)
        for x in f.inputs:
            add_var(dot, x, var_attr, verbose=verbose)
            if x.creator:
                push_func(x.creator)

    return dot


def save_gv(
    x: Variable,
    dir: os.PathLike | str,
    fname: os.PathLike | str,
    verbose=True,
    **attrs: Unpack[GraphAttrs],
) -> Path:
    """Save only the DOT source as `<fname>.gv` in `dir`."""
    dot = create_graph(x, verbose=verbose, **attrs)
    filename = f"{fname}.gv"
    return Path(dot.save(filename=filename, directory=dir))


def render(
    x: Variable,
    to_file: os.PathLike | str,
    view=False,
    cleanup=True,
    verbose=True,
    **attrs: Unpack[GraphAttrs],
) -> Path:
    """Render the computational graph of `x` to a file and return its path.

    `to_file` is divided into three parts, `<dir>/<fname>.<format>`

    The DOT source is saved as `<fname>.gv` in `dir`, and the rendered file
    as `<fname>.<format>`. If `format` is None, `fname` is used as is for the
    output. The source is removed afterwards when `cleanup` is True.

    `attrs` are forwarded to `create_graph`: `graph_attr`, `node_attr`, `edge_attr`,
    `var_attr` and `func_attr`.

    Example:
        >>> import numpy as np
        >>> import dezero
        >>> x = dezero.Variable(np.random.randn(2,3), dtype=np.float32, name='x')
        >>> y = (x + 1) ** 2
        >>> from pathlib import Path
        >>> render(y, Path.home()/"quad.svg", cleanup=True)
        PosixPath('home/[user]/quad.svg')
        >>> render(y, "~/Downloads/quad.svg")
        PosixPath('home/[user]/Downloads/quad.svg')
    """
    path = Path(to_file).expanduser()
    dir = path.parent
    fname = path.stem
    format = path.suffix.removeprefix(".")
    dir = Path(dir)
    fname = PurePath(fname)
    dir.mkdir(parents=True, exist_ok=True)

    dot = create_graph(x, **attrs, verbose=verbose)

    dot.directory = str(dir)
    dot.filename = str(fname.with_suffix(".gv"))

    if format is None:
        outfile = dir / fname
    else:
        outfile = dir / fname.with_suffix(f".{format}")
    path = dot.render(outfile=outfile, view=view, cleanup=cleanup, format=format)
    return Path(path)


save_image = render


def save_svg(
    x: Variable,
    to_file: os.PathLike | str,
    view=False,
    cleanup=True,
    verbose=True,
    **attrs: Unpack[GraphAttrs],
) -> Path:
    """Render the Computational-Graph of `x` in `to_file` formatted to `.svg` and Returns its path.

    If `to_file` has non-`.svg` suffix such as `.png`, its suffix is replaced by `.svg`.
    """
    return render(
        x,
        Path(to_file).with_suffix(".svg"),
        view=view,
        verbose=verbose,
        cleanup=cleanup,
        **attrs,
    )
