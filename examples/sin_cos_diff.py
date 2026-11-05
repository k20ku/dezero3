from pathlib import Path

import matplotlib
import numpy as np

from dezero.core import variable
import dezero.functions as F
import matplotlib.pyplot as plt

def main():
    x = variable(np.linspace(-7, 7, 200))
    y = F.sin(x)
    y.backward(create_graph=True)

    logs = [y.data.flatten()]

    for i in range(3):
        gx = x.grad
        if gx is None:
            raise AssertionError(f"gx is None, gx={gx}")
        logs.append(gx.data.flatten())
        x.cleargrad()
        gx.backward(create_graph=True)

    labels = ["y=sin(x)", "y", "y'", "y''"]
    for log, label in zip(logs, labels):
        plt.plot(x.data, log, label=label)

    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()
    
