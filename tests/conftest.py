import numpy as np
import pytest

from dezero.core import using_config


@pytest.fixture(autouse=True)
def _default_float64():
    with using_config("default_dtype", np.float64):
        yield
