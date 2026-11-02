from pathlib import Path

from dezero.core import variable
from dezero.utils.graph import graph

def goldstein_price(x1, x2):
    return (1 + (x1 + x2 + 1)**2 * (19 - 14 * x1 + 3*x1**2 + 6*x1*x2 + 3*x2**2)) * (30 + (2*x1 - 3*x2)**2*(18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))

def main():
    x = variable(1)
    y = variable(1)
    z = goldstein_price(x, y)
    z.backward()

    x.name = 'x'
    y.name = 'y'
    z.name = 'z'

    dot = graph(output=z, verbose=False)
    name="goldstein_price"
    dir = Path.home()/"dezero"/"calc_graph"
    dot.save(filename=f"{name}.gv", directory=dir)
    dot.render(filename=f"{name}", directory=dir, view=False, cleanup=False, format='svg')

if __name__ == "__main__":
    main()
