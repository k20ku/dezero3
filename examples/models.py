from pathlib import Path

import dezero.functions as F
import dezero.layers as L
import dezero.views as vue
import numpy as np

from dezero import Variable


class TwoLayerNet(L.Model):
    def __init__(self, hidden_size: int, out_size: int):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


x = Variable(np.random.randn(5, 10).astype(np.float32), name='x')
model = TwoLayerNet(100, 10)
path = model.render(
        x,
        to_file=Path.home()/"Downloads"/"two-layers.svg",
        view=True,
        **vue.with_attrs(
            theme=vue.DARK, 
            graph_attr={"rankdir": "LR"}
        )
    )
print(path)
