import matplotlib.pyplot as plt
import numpy as np

from dezero.core import Variable
import dezero.functions as F
from dezero.utils.expr import to_latex


def main():
    X, Y = np.meshgrid(
        np.linspace(-np.pi, np.pi, 2000),
        np.linspace(-np.pi, np.pi, 2000),
    )

    x = Variable(X, name="x")
    y = Variable(Y, name="y")

    z = F.sin(x) * F.cos(y)

    Z0 = z.data
    z.backward(create_graph=True)

    gx = x.grad
    if gx is None:
        print("Z is None")
        return
    gx.name = "gx"
    import dezero.utils.graph as dg

    dg.save_cmp_graph(gx, "sin_cos", home=True, verbose=True)
    Z1 = gx.data
    titles = [
        rf"$y ={to_latex(z)}$",
        r"$\frac{\partial y}{\partial x}$",
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
