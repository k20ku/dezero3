from collections.abc import Sequence
from typing import Any, SupportsIndex, TypeAlias

import numpy as np

xp = np

type NDArray = xp.ndarray
ndarray: TypeAlias = xp.ndarray

type DType64 = xp.float64
type DType32 = xp.float32
dtype64: TypeAlias = xp.float64
dtype32: TypeAlias = xp.float32


Shape: TypeAlias = tuple[int, ...]
AnyShape: TypeAlias = tuple[Any, ...]

# Anything that can be coerced to a shape tuple
ShapeLike: TypeAlias = SupportsIndex | Sequence[SupportsIndex]
