# dezero3

## venv

If `venv` is not activated, run

```bash
source .venv/bin/activate
```

## script exection

```bash
uv run ipython -m steps.stepXX
```

## repl

```bash
uv run ipython
```

Responsive to file changes

```ipython
%load_ext autoreload
%autoreload 2
```

For each step, run below on the repl.

```ipython
# e.g. step01.py
%run steps/stepXX.py
```
