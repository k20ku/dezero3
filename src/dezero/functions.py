from dezero.core import Function, Variable, xp

class Sin(Function):
    def forward(self, x):
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        return gy * cos(x)

class Cos(Function):
    def forward(self, x):
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        return - gy * sin(x)

def sin(x: Variable) -> Variable:
    return Sin()(x)

def cos(x: Variable) -> Variable:
    return Cos()(x)
