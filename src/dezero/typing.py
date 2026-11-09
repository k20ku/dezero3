from collections.abc import Sequence
from typing import Any, SupportsIndex, TypeAlias

import numpy as np

xp = np

type NDArray = xp.ndarray
ndarray: TypeAlias = xp.ndarray

type DType = xp.float64
dtype: TypeAlias = xp.float64


Shape: TypeAlias = tuple[int, ...]
AnyShape: TypeAlias = tuple[Any, ...]

# Anything that can be coerced to a shape tuple
ShapeLike: TypeAlias = SupportsIndex | Sequence[SupportsIndex]
