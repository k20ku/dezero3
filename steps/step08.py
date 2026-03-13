import numpy as np
import dezero.core as dz

if __name__ == '__main__':

    print("--Functions--")
    F0 = dz.Square()
    F1 = dz.Exp()
    F2 = dz.Square()

    x0 = dz.Variable(np.array(0.5))
    x1 = F0(x0)
    x2 = F1(x1)
    x3 = F2(x2)

    x3.grad = np.array(1.0)
    x3.backward()
    print("x0.grad: ", x0.grad)
