# dezero3

Deep Learning Framework repository based on "Deep Learning from Scratch 3" by 斎藤康毅([koki0702](https://github.com/koki0702))

『ゼロから作る Deep Learning ③』をベースにした、自動微分ライブラリの実装リポジトリ。

- 型ヒント・プロジェクト構成・テスト環境を整備しながら、自動微分エンジンの内部構造を理解することを目的としている。
- 現在は Function Evaluation 周辺まで実装済みで、計算グラフと逆伝播の基盤部分を構築中。

## Developer Notices

- **Developers shuold make `venv` activated**.

```bash
uv venv
source .venv/bin/activate
```

- Setting Up dependencies

```bash
uv sync
```

## Run Dezero

```bash
uv run main.py
```

### Run in Colab

![Dezero used from Colab](./resources/colab_example.webp)

### Run in IPython REPL

![Dezero used from REPL](./resources/REPL_example.webp)

```bash
uv run ipython
```

You can make REPL responsive to module-file changes

```python
%load_ext autoreload
%autoreload all
```

You can run specific script file (i.e. `main.py`) on the REPL.

```python
%run main.py
```

### Tests Dezero

To run all test,

```bash
pytest
```

To test specified class or functions (i.e. `class TestSquare` in `tests/core_test.py`)

```bash
pytest tests/core_test.py::TestSquare
```

## Motivation

- 自動微分の仕組みを理解する
- Python の型システムを活用する
- ライブラリとして保守しやすい構成にする
- 実装の背景にある設計を追う

ニューラルネットワークそのものよりも、

- Variable
- Function
- Computation Graph
- Automatic Differentiation

といった基盤技術に興味があり、その理解を目的として実装している。

## Development Process

- 書籍ではステップごとのスクリプトとして実装が進む
- このリポジトリでは最初からライブラリとして利用できる構成を採用している。

構成を考える際には、

- DeZero
- NumPy
- Chainer
- PyTorch

などのリポジトリを参考にしている。

また、環境管理には [uv](https://docs.astral.sh/uv/) を利用している。

1. 実装済み
        - Variable
        - Function
        - Computation Graph
        - Backward Propagation (basic)
        - Gradient Checking
        - Add / 複数入力演算

In implementation,

- Reshape
- NNW usage

## Improvements

### 1. 型ヒント

このリポジトリでは型ヒントを積極的に導入している。
ただし、すべてを厳密に型付けするのではなく、

- 内部実装は基本的に厳密に
- 本質的に密結合な `Function` と `Variable` は動的に
- 利用者向け API は柔軟に
- 境界は厳格に

という方針を採っている。

Example:

強く型付けしているものは，

- `Variable`
- `Function`
- grad
- 計算グラフ内部

一方で利用者は，

```python
def rosenbrock(x0, x1):
    return 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
```

のような Python らしい記述を損なわないことも重視している。

### 

- 書籍では標準の unittest が利用されている
- このリポジトリでは pytest を採用している。

For below reasons, 
- parametrize test

### 3. 動的型と静的型ヒントの区別

```python
type NDArray = np.ndarray

xp = np

ndarray = xp.ndarray

```

np.ndarrayをいちいち書くのはめんどくさい（将来的な`cupy`対応をする）ため，型エイリアスと動的な型は一か所に置いてあるがそれぞれ定義している

```python
isinstance(x, ndarray)
```

```python
def f() -> NDArray:...
```

Basically `np.multi_index` is too primitive to use these days.

Bottleneck of numerical diff is `f` that creates the graph for each index. 
Therefore, so is alternate to `np.ndindex` + `for` loop.

## Benchmark

```bash
uv run benchmarks/dangling.py
```

| Mode      | During computation | Before GC | After GC |
| --------- | -----------------: | --------: | -------: |
| With Grad |            +6.4 MB |   +8.0 MB |   +836 B |
| No Grad   |            +0.8 MB |   +0.8 MB |   +176 B |

