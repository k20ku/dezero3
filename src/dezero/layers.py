import weakref
from abc import ABC
from collections.abc import Generator, Sequence
from typing import Any, override

from dezero.core import Parameter, Variable, as_variables
from dezero.typing import NDArray


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

