import heapq
from dezero.core import Function, Variable

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
        add_func(dot, f)
        for x in f.inputs:
            add_var(dot, x, verbose=verbose)
            if x.creator:
                push_func(x.creator)

    return dot

