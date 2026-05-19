from functools import reduce

import numpy as np

from dezero.core import Variable, exp, square


def forward(funcs, arg):
    if funcs is None:
        return arg
    return reduce(lambda _arg, func: func(_arg), funcs, arg)


def main():
    x = Variable(np.random.randn(3, 2, 2))
    funcs = [square, exp, square]
    to = " -> "
    print("forward:")
    print("\t", "x", to, to.join([func.__name__ for func in funcs]), to, "y")
    print("\n", "-" * 60, "\n")
    y = forward(funcs, x)
    print("x = ", x)
    print("y = ", y)
    print("\n", "=" * 60, "\n")
    to = " <- "
    print("backward: ")
    print("\t", "x", to, to.join([func.__name__ for func in funcs]), to, "y")
    print("\n", "-" * 60, "\n")
    y.backward()
    print("y = ", y)
    print("x = ", x)


if __name__ == "__main__":
    main()
