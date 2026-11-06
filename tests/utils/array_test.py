import numpy as np
import pytest

from dezero.utils.array import sum_to

@pytest.mark.parametrize(
        argnames=(
            "x", "shape", "want",
        ),
        argvalues=[
            (
                [[1, 1, 1] ,[1, 1, 1]], (3,), [2, 2, 2]
            ),
            (
                [[1,2,3],[4,5,6]], (1, 3), [[5,7,9]]
            ),
            (
                [[5, -1, 6], [8, 98, -18]], (2, 1), [[10],[88]],
            )
        ],
)
def test_sum_to(x, shape, want):
    x = np.array(x)
    want = np.array(want)
    y = sum_to(x, shape)
    np.testing.assert_array_equal(y, want)

