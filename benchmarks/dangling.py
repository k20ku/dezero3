import gc
import tracemalloc
import weakref

import numpy as np
import scipy
import scipy.special

from dezero.core import Variable, no_grad, square


def bench():
    tracemalloc.start()
    gc.collect()
    current_base, _ = tracemalloc.get_traced_memory()
    print(
        """
  memory-block-size:
    base-size: {current_base}""",
        end="",
    )
    print(
        f"""
    do-nothing:
      - diff: {current_base - current_base}""",
        end="",
    )

    # create graph
    y = square(Variable(scipy.special.expit(np.random.randn(10000, 5, 2))))
    ref = None

    for i in range(8):
        y = square(y)
        if i == 5:
            ref = weakref.ref(y)
            current, _ = tracemalloc.get_traced_memory()
            print(
                f"""
    in-calculation:
      - diff: {current - current_base}""",
                end="",
            )

    current, _ = tracemalloc.get_traced_memory()

    print(
        f"""
    before-gc:
      - diff: {current - current_base}
      - y.shape: |
          {y.data.shape}
      - y.data[0]: |
          {y.data[0].__repr__().replace("\n", "\n" + " " * (4 * 2 + 2))}""",
        end="",
    )

    del y
    gc.collect()
    current, _ = tracemalloc.get_traced_memory()

    print(
        f"""
    after-gc:
      - diff: {current - current_base}""",
        end="",
    )

    assert ref() is None


if __name__ == "__main__":
    print(
        """
with-grad:""",
        end="",
    )
    bench()
    print()
    with no_grad():
        print(
            """
no-grad:""",
            end="",
        )
        bench()
    print("\n")
