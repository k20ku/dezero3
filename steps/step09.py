import numpy as np
import dezero.core as dz


if __name__ == '__main__':
    print("--Functions--")
    x = dz.Variable(np.array(0.5))
    y = dz.square(dz.exp(dz.square(x)))
    print("-----end-----", end='\n\n')

    y.backward()
    print("x.grad: ", x.grad)
