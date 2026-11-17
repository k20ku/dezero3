from collections.abc import Mapping

Attrs = Mapping[str, str]
from typing import TypedDict, Unpack


class GraphAttrs(TypedDict, total=False):
    graph_attr: Attrs
    node_attr: Attrs
    edge_attr: Attrs
    var_attr: Attrs
    func_attr: Attrs


def with_attrs(theme: GraphAttrs, **overrides: Unpack[GraphAttrs]) -> GraphAttrs:
    """Merge overrides into a theme, key by key.
    Example:
        >>> render(y, "out", "quad", "svg", **with_attrs(DARK, graph_attr={"rankdir": "LR"}))
    """
    keys = theme.keys() | overrides.keys()
    return {k: {**theme.get(k, {}), **overrides.get(k, {})} for k in keys}  # type: ignore[misc]


GRAPH_DEFAULT: Attrs = {}  # put your default graph attributes here
NODE_DEFAULT: Attrs = {}
EDGE_DEFAULT: Attrs = {}
VAR_DEFAULT: Attrs = {"style": "filled", "fillcolor": "orange"}
FUNC_DEFAULT: Attrs = {"style": "filled", "fillcolor": "lightblue", "shape": "box"}

# Catppuccin Mocha palette
DARK: GraphAttrs = {
    "graph_attr": {"bgcolor": "#1e1e2e", "pad": "0.4"},
    "node_attr": {
        "fontname": "Helvetica",
        "fontcolor": "#cdd6f4",
        "penwidth": "1.5",
    },
    "edge_attr": {
        "color": "#a6adc8",
        "fontcolor": "#a6adc8",
        "fontname": "Helvetica",
    },
    # Variable nodes: ellipse (graphviz default), peach border
    "var_attr": {
        "style": "filled",
        "fillcolor": "#45475a",
        "color": "#fab387",
    },
    # Function nodes: rounded box, blue border
    "func_attr": {
        "style": "filled,rounded",
        "shape": "box",
        "fillcolor": "#313244",
        "color": "#89b4fa",
    },
}
