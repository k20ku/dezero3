import os
import pathlib
import weakref
from abc import ABC
from collections.abc import Generator, Sequence
from typing import Any, Unpack, override

import dezero.functions as F
from dezero.core import Parameter, Variable, as_variables
from dezero.typing import NDArray, xp
from dezero.utils.graph import render
from dezero.views import GraphAttrs


class Layer(ABC):
    """
    Base class to construct Layers
    """

    inputs: tuple[weakref.ReferenceType[Variable], ...]
    outputs: tuple[weakref.ReferenceType[Variable], ...]

    def __init__(self):
        self._params: set[str] = set()

    @override
    def __setattr__(self, name: str, value: Any, /) -> None:
        if isinstance(value, (Parameter, Layer)):
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

    def forward(self, *inputs: Variable | NDArray) -> Any: ...

    def params(self) -> Generator[Parameter]:
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()


class Linear(Layer):
    def __init__(
        self, out_size: int, in_size: int | None = None, nobias=False, dtype=xp.float32
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        self.w: Parameter | None = None
        self.b: Parameter | None = (
            None if nobias else Parameter(xp.zeros(out_size, dtype=dtype), name="b")
        )

    def _init_W(self):
        I, O = self.in_size, self.out_size
        assert I is not None and O is not None
        # https://cseweb.ucsd.edu/classes/wi08/cse253/Handouts/lecun-98b.pdf
        weight = xp.sqrt(
            1 / I, dtype=self.dtype
        )  # assign dtype=self.dtype since np.sqrt returns default np.float64
        W_data = weight * xp.random.randn(I, O).astype(self.dtype)
        self.w = Parameter(W_data, name="W")

    def forward(self, x: Variable | NDArray):
        if x.ndim < 2:
            raise ValueError(
                f"x.shape = {x.shape}, x.ndim = {x.ndim}, ndim must be greater than 2."
            )
        if self.w is None:
            in_size = x.shape[1]
            self.in_size = in_size
            self._init_W()

        y = F.linear(x, self.w, self.b)
        return y


class Model(Layer):
    def render(
        self,
        *inputs: Variable,
        to_file: os.PathLike | str = pathlib.Path.home() / "model.png",
        view=False,
        save_dot=False,
        verbose=True,
        **attrs: Unpack[GraphAttrs],
    ) -> pathlib.Path:
        """
        Example:
            >>> import numpy as np
            >>> import dezero
            >>> x = dezero.Variable(np.random.randn(2,3), dtype=np.float32, name='x')
            >>> model = TwoLayerNet(4, 3)
            >>> from pathlib import Path
            >>> model.render(x, "~/quad.svg", save_dot=True)
            PosixPath('home/[user]/quad.svg')
            or
            WindowsPath('C:/Users/[user]/quad.svg')
            >>> model.render(x, Path.home()/"Downloads"/"quad.svg")
            PosixPath('home/[user]/Downloads/quad.svg')
            or
            WindowsPath('C:/Users/[user]/Downloads/quad.svg')
        """
        y = self.forward(*inputs)
        y.name = "y"
        return render(
            y, to_file, view=view, cleanup=not save_dot, verbose=verbose, **attrs
        )
