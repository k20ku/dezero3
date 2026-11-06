from dezero.core import Function, Variable, xp
import dezero.core as dc

Exp = dc.Exp
Square = dc.Square

Add = dc.Add
Sub = dc.Sub
Mul = dc.Mul
Div = dc.Div
Neg = dc.Neg
Pow = dc.Pow
exp = dc.exp
square = dc.square


class Sin(Function):
    def forward(self, x):
        y = xp.sin(x)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        return gy * cos(x)


class Cos(Function):
    def forward(self, x):
        y = xp.cos(x)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        return -gy * sin(x)


class Tanh(Function):
    def forward(self, x):
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y_weak = self.outputs[0]
        y = y_weak()
        if y is None:
            raise AssertionError(f"{self} has None output")
        return gy * (1 - y**2)


def sin(x: Variable) -> Variable:
    return Sin()(x)


def cos(x: Variable) -> Variable:
    return Cos()(x)


def tanh(x: Variable) -> Variable:
    return Tanh()(x)
