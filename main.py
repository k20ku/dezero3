from functools import partial
from pathlib import Path

from dezero.core import variable
from dezero.utils.grad import dn
from dezero.utils.graph import save_svg
from dezero.views import DARK, with_attrs


def goldstein_price(x1, x2):
    return (
        1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1**2 + 6 * x1 * x2 + 3 * x2**2)
    ) * (
        30
        + (2 * x1 - 3 * x2) ** 2
        * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2)
    )


x = variable(1)
y = variable(1)
z = goldstein_price(x, y)

x.name = "x"
y.name = "y"
z.name = "z"

name = "goldstein_price"
dir = Path.home() / "Downloads"
file_to = dir / f"{name}.svg"
path = save_svg(
    z,
    file_to,
    view=True,
    verbose=True,
    **with_attrs(theme=DARK, graph_attr={"rankdir": "LR"}),
)
print(path)

n = 1
gx = dn(partial(goldstein_price, x2=y), x, n=n)
file_to = dir / f"{name}_diff{n}"
path = save_svg(
    gx,
    file_to,
    view=True,
    verbose=True,
    **with_attrs(theme=DARK, graph_attr={"rankdir": "LR"}),
)
print(path)
