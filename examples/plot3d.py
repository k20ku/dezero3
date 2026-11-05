from collections import deque
from itertools import count

import matplotlib.pyplot as plt
import numpy as np

from dezero.core import Function, Variable
import dezero.functions as F

import heapq

import sympy


def to_exp(output: Variable):
    cache = {}
    stack = deque([output])
    while stack:
        output = stack.pop()
        f = output.creator
        if f is None:
            name = output.name if output.name is not None else f"x_{id(output)}"
            cache[id(output)] = sympy.Symbol(name)
            stack.pop()
            continue
        pendings = [input for input in f.inputs if id(input) not in cache]
        if pendings:
            stack.extend(pendings)
        else:
            args = tuple(cache[id(x)] for input in f.inputs)
            func = type(f).__name__
            cache[id(output)] = SYMPY[func](*args)
            stack.pop()
        return cache[id(output)]


def main():
    X, Y = np.meshgrid(
        np.linspace(-np.pi, np.pi, 2000),
        np.linspace(-np.pi, np.pi, 2000),
    )

    x = Variable(X, name="x")
    y = Variable(Y, name="y")

    z = F.sin(x) * F.cos(y)

    Z0 = z.data
    z.backward()

    gx = x.grad
    if gx is None:
        print("Z is None")
        return
    Z1 = gx.data
    titles = [
        r"$y = \sin(x) \cdot \cos(y)$",
        r"$\frac{\partial y}{\partial x}",
    ]
    ZS = [Z0, Z1]
    fig, axs = plt.subplots(
        ncols=len(ZS),
        subplot_kw={"projection": "3d"},
        figsize=(24, 10),
    )
    for ax, title, Z in zip(axs, titles, ZS):
        surf = ax.plot_surface(X, Y, Z, cmap="coolwarm")
        ax.set_title(title, fontsize=25)
        ax.set_zlim(-1.01, 1.01)
        # shrinks colorbar since it's too big compared with 3d-surf plot
        fig.colorbar(
            surf,
            shrink=0.25,
        )

    plt.show()


if __name__ == "__main__":
    main()
