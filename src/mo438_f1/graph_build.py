"""Construction of the directed (G_d) and simple undirected (G_u) representations.

``G_d`` keeps every citation exactly as provided by the dataset (orientation
``u -> v``: *u cites v*). ``G_u`` is the simple undirected graph obtained by

1. dropping the orientation of each edge;
2. consolidating parallel pairs: ``u -> v``, ``v -> u`` and repeated rows become a
   single undirected edge ``{u, v}`` (no multi-edges, no weights);
3. removing self-loops ``u -> u`` (configurable, default ``True``) so that ``G_u``
   is a *simple* graph, which is what degree/clustering/density formulas assume.

All nodes are kept in ``G_u`` (even if they become isolated), with the same index
and attributes as in ``G_d``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import igraph as ig
import numpy as np
import scipy.sparse as sp

from .data import CitationData


@dataclass
class ConversionStats:
    """Bookkeeping of the G_d -> G_u transformation."""

    num_nodes: int
    num_directed_edges: int
    num_directed_self_loops: int
    num_duplicate_directed_edges: int
    num_reciprocal_pairs: int
    num_undirected_edges: int
    self_loops_removed: bool

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (JSON serialisable)."""
        return asdict(self)


def undirected_edge_array(
    edges: np.ndarray, num_nodes: int, remove_self_loops: bool = True
) -> tuple[np.ndarray, ConversionStats]:
    """Collapse directed edges into unique undirected pairs ``(min, max)``.

    Args:
        edges: ``(m, 2)`` directed edge list.
        num_nodes: number of nodes (endpoints must be in ``[0, num_nodes)``).
        remove_self_loops: drop ``(u, u)`` pairs.

    Returns:
        ``(uedges, stats)`` where ``uedges`` is a sorted ``(m_u, 2)`` int64 array with
        ``uedges[:, 0] <= uedges[:, 1]`` and no repeated rows.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    n = np.int64(num_nodes)
    src, dst = edges[:, 0], edges[:, 1]
    loops = src == dst

    directed_keys = np.unique(src * n + dst)
    num_duplicates = int(edges.shape[0] - directed_keys.size)
    num_unique_non_loop_directed = int(np.count_nonzero(directed_keys // n != directed_keys % n))

    low = np.minimum(src, dst)
    high = np.maximum(src, dst)
    keep = ~loops if remove_self_loops else np.ones_like(loops)
    undirected_keys = np.unique(low[keep] * n + high[keep])
    uedges = np.column_stack([undirected_keys // n, undirected_keys % n]).astype(np.int64)
    num_unique_non_loop_pairs = int(np.count_nonzero(uedges[:, 0] != uedges[:, 1]))

    stats = ConversionStats(
        num_nodes=int(num_nodes),
        num_directed_edges=int(edges.shape[0]),
        num_directed_self_loops=int(np.count_nonzero(loops)),
        num_duplicate_directed_edges=num_duplicates,
        # Each unordered pair cited in both directions collapses two unique directed
        # edges into one undirected edge.
        num_reciprocal_pairs=num_unique_non_loop_directed - num_unique_non_loop_pairs,
        num_undirected_edges=int(uedges.shape[0]),
        self_loops_removed=bool(remove_self_loops),
    )
    return uedges, stats


def _set_node_attributes(graph: ig.Graph, data: CitationData) -> None:
    graph.vs["node_id"] = list(range(data.num_nodes))
    graph.vs["category"] = data.labels.tolist()
    graph.vs["category_name"] = [data.category_names[k] for k in data.labels]
    if data.years is not None:
        graph.vs["year"] = data.years.tolist()
    if data.paper_ids is not None:
        graph.vs["paper_id"] = data.paper_ids.tolist()


def build_directed_graph(data: CitationData) -> ig.Graph:
    """Build ``G_d`` (igraph, directed) preserving all edges and node attributes.

    Node attributes: ``node_id`` (OGB node index), ``category`` (label index),
    ``category_name`` (e.g. ``cs.AI``), ``year`` and ``paper_id`` (MAG id) when known.
    """
    graph = ig.Graph(n=data.num_nodes, edges=data.edges.tolist(), directed=True)
    _set_node_attributes(graph, data)
    graph["name"] = f"{data.name}_directed"
    return graph


def build_undirected_graph(
    data: CitationData, remove_self_loops: bool = True
) -> tuple[ig.Graph, ConversionStats, np.ndarray]:
    """Build the simple undirected ``G_u`` from the directed citations.

    Returns:
        ``(G_u, stats, uedges)``; ``uedges`` is the undirected edge array used to build
        the graph (handy to derive a CSR adjacency without copying igraph's data).
    """
    uedges, stats = undirected_edge_array(data.edges, data.num_nodes, remove_self_loops)
    graph = ig.Graph(n=data.num_nodes, edges=uedges.tolist(), directed=False)
    _set_node_attributes(graph, data)
    graph["name"] = f"{data.name}_undirected"
    return graph, stats, uedges


def csr_from_undirected_edges(uedges: np.ndarray, num_nodes: int) -> sp.csr_matrix:
    """Symmetric, unweighted CSR adjacency matrix (int8) of an undirected edge list."""
    uedges = np.asarray(uedges, dtype=np.int64).reshape(-1, 2)
    non_loop = uedges[:, 0] != uedges[:, 1]
    rows = np.concatenate([uedges[:, 0], uedges[non_loop, 1]])
    cols = np.concatenate([uedges[:, 1], uedges[non_loop, 0]])
    data = np.ones(rows.size, dtype=np.int8)
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def induced_subgraph_csr(adjacency: sp.csr_matrix, nodes: np.ndarray) -> sp.csr_matrix:
    """Adjacency of the subgraph induced by ``nodes`` (relabelled ``0..len(nodes)-1``)."""
    nodes = np.asarray(nodes, dtype=np.int64)
    sub = adjacency[nodes][:, nodes].tocsr()
    sub.sort_indices()
    return sub
