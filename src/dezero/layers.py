import weakref
from abc import ABC
from collections.abc import Generator, Sequence
from typing import Any, SupportsIndex, override

import dezero.functions as F
from dezero.core import Parameter, Variable, as_variables
from dezero.typing import NDArray, ShapeLike, xp


class Layer(ABC):
    inputs: tuple[weakref.ReferenceType[Variable], ...]
    outputs: tuple[weakref.ReferenceType[Variable], ...]
    def __init__(self):
        self._params: set[str] = set()

    @override
    def __setattr__(self, name: str, value: Any, /) -> None:
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs: Variable | NDArray):
        inputs = tuple(as_variables(*inputs))
        ys = self.forward(*inputs)
        if not isinstance(ys, Sequence):
            ys = (ys,)
        outputs: tuple[Variable, ...] = tuple(as_variables(*ys))
        self.inputs = tuple(weakref.ref(input) for input in inputs)
        self.outputs = tuple(weakref.ref(output) for output in outputs)
        if len(outputs) == 0:
            raise AssertionError("Expected code to be unreachable")

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *inputs: Variable | NDArray) -> Any:...

    def params(self) -> Generator[Parameter]:
        for name in self._params:
            yield self.__dict__[name]

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()


class Linear(Layer):
    def __init__(self, out_size: int, in_size: int | None =None, nobias=False, dtype=xp.float32):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        self.w: Parameter | None = None
        self.b: Parameter | None = None if nobias else Parameter(xp.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        I, O = self.in_size, self.out_size
        assert I is not None and O is not None
        W_data = xp.random.randn(I, O).astype(self.dtype) * xp.sqrt(1 / I)
        self.w = Parameter(W_data, name='W')

    def forward(self, x: Variable | NDArray):
        if self.w is None:
            in_size = x.shape[1]
            self.in_size = in_size
            self._init_W()

        y = F.linear(x, self.w, self.b)
        return y
