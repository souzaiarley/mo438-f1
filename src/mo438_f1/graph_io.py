"""GraphML serialisation with properly typed attributes.

igraph's own writer declares every numeric attribute as ``double``; this module
streams GraphML directly from arrays, declaring integers as ``int``/``long``, so
that Gephi, networkx and igraph read back categories, years and ids as integers.
Paths ending in ``.gz`` are gzip-compressed transparently.
"""

from __future__ import annotations

import gzip
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any
from xml.sax.saxutils import escape, quoteattr

import igraph as ig
import numpy as np

_INT32 = np.iinfo(np.int32)


def _graphml_type(values: Sequence[Any] | np.ndarray) -> str:
    array = np.asarray(values)
    if array.dtype.kind == "b":
        return "boolean"
    if array.dtype.kind in "iu":
        if array.size == 0 or (array.min() >= _INT32.min and array.max() <= _INT32.max):
            return "int"
        return "long"
    if array.dtype.kind == "f":
        return "double"
    return "string"


def _format(value: Any, kind: str) -> str:
    if kind == "boolean":
        return "true" if value else "false"
    if kind == "double":
        return repr(float(value))
    if kind in {"int", "long"}:
        return str(int(value))
    return escape(str(value))


def _open(path: Path, compress: bool) -> IO[str]:
    if compress:
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def write_graphml(
    path: str | Path,
    num_nodes: int,
    edges: np.ndarray,
    directed: bool,
    node_attributes: Mapping[str, Sequence[Any] | np.ndarray] | None = None,
    edge_attributes: Mapping[str, Sequence[Any] | np.ndarray] | None = None,
    graph_attributes: Mapping[str, Any] | None = None,
) -> Path:
    """Stream a graph to GraphML.

    Node ``i`` is written with id ``n{i}`` (igraph's convention), so node order is
    preserved on reading. Every attribute array must have one value per node/edge.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    node_attributes = dict(node_attributes or {})
    edge_attributes = dict(edge_attributes or {})
    graph_attributes = dict(graph_attributes or {})
    for name, values in node_attributes.items():
        if len(values) != num_nodes:
            raise ValueError(f"node attribute {name!r} has {len(values)} values != {num_nodes}")
    for name, values in edge_attributes.items():
        if len(values) != edges.shape[0]:
            raise ValueError(f"edge attribute {name!r} length != number of edges")

    node_kinds = {name: _graphml_type(v) for name, v in node_attributes.items()}
    edge_kinds = {name: _graphml_type(v) for name, v in edge_attributes.items()}
    graph_kinds = {name: _graphml_type([v]) for name, v in graph_attributes.items()}

    tmp = path.with_name(path.name + ".tmp")
    with _open(tmp, compress=path.suffix == ".gz") as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out.write(
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
            'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n'
        )
        for prefix, scope, kinds in (("g", "graph", graph_kinds), ("v", "node", node_kinds),
                                     ("e", "edge", edge_kinds)):
            for name, kind in kinds.items():
                out.write(
                    f"  <key id={quoteattr(prefix + '_' + name)} for=\"{scope}\" "
                    f"attr.name={quoteattr(name)} attr.type=\"{kind}\"/>\n"
                )
        out.write(f'  <graph id="G" edgedefault="{"directed" if directed else "undirected"}">\n')
        for name, value in graph_attributes.items():
            out.write(f'    <data key="g_{name}">{_format(value, graph_kinds[name])}</data>\n')

        node_cols = [(name, node_kinds[name], np.asarray(v)) for name, v in node_attributes.items()]
        for i in range(num_nodes):
            if node_cols:
                data = "".join(
                    f'<data key="v_{name}">{_format(col[i], kind)}</data>'
                    for name, kind, col in node_cols
                )
                out.write(f'    <node id="n{i}">{data}</node>\n')
            else:
                out.write(f'    <node id="n{i}"/>\n')

        edge_cols = [(name, edge_kinds[name], np.asarray(v)) for name, v in edge_attributes.items()]
        src_list = edges[:, 0].tolist()
        dst_list = edges[:, 1].tolist()
        if edge_cols:
            for j, (u, v) in enumerate(zip(src_list, dst_list)):
                data = "".join(
                    f'<data key="e_{name}">{_format(col[j], kind)}</data>'
                    for name, kind, col in edge_cols
                )
                out.write(f'    <edge source="n{u}" target="n{v}">{data}</edge>\n')
        else:
            chunk = 100_000
            for start in range(0, len(src_list), chunk):
                out.write("".join(
                    f'    <edge source="n{u}" target="n{v}"/>\n'
                    for u, v in zip(src_list[start:start + chunk], dst_list[start:start + chunk])
                ))
        out.write("  </graph>\n</graphml>\n")
    tmp.replace(path)
    return path


def write_igraph_graphml(graph: ig.Graph, path: str | Path) -> Path:
    """Write an igraph graph (vertex/edge/graph attributes included) to typed GraphML."""
    edges = np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    node_attrs = {name: graph.vs[name] for name in graph.vs.attributes()}
    edge_attrs = {name: graph.es[name] for name in graph.es.attributes()}
    graph_attrs = {name: graph[name] for name in graph.attributes()}
    return write_graphml(
        path,
        graph.vcount(),
        edges,
        graph.is_directed(),
        node_attributes=node_attrs,
        edge_attributes=edge_attrs,
        graph_attributes=graph_attrs,
    )
