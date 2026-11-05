from pathlib import Path

from dezero.core import variable, Variable
import dezero.functions as F
import dezero.utils.graph as dg
import tqdm


def dn(n: int = 0) -> Variable:
    x = variable(1.0, name="x")
    y = F.tanh(x)
    y.name = "y"
    y.backward(create_graph=True)

    for step in tqdm.tqdm(range(n), desc=f"Computing d({n + 1})tanh(x)/dx diff"):
        if x.grad is None:
            raise AssertionError(f"x=({x}), grad is None")
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    gx = x.grad
    if gx is None:
        raise AssertionError(f"gx=({gx}), grad is None")

    gx.name = f"g^({n + 1})x"

    return gx


if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:
        n = 0
    else:
        try:
            n = int(sys.argv[1])
        except ValueError:
            n = 0

    gx = dn(n)
    dot = dg.graph(gx, verbose=False)
    dir = Path.home() / "dezero" / "cmp_graph"
    #dir = "resources"
    name = f"tanh_diff{n+1}"
    dot.save(filename=f"{name}.gv", directory=dir)
