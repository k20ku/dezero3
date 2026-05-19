# dezero3

## venv

**Developers shuold make `venv` activated**.

```bash
source .venv/bin/activate
```

## run

```bash
uv run main.py
```

## tests

To test all

```bash
pytest
```

To test specified class or functions

```bash
pytest tests/core_test.py::TestSquare
```

## repl

```bash
uv run ipython
```

Responsive to file changes

```python
%load_ext autoreload
%autoreload 2
```

For each step, run below on the repl.

```python
# e.g. main.py
%run main.py
```
