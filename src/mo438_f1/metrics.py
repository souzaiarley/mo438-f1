"""Structural metrics required by F1 (G_u) and complementary directed metrics (G_d).

Conventions (as in the course notes, MO438 — bloco 03):

* local clustering ``C_i = 2 T_i / (d_i (d_i - 1))`` with ``C_i = 0`` when ``d_i < 2``;
* **global clustering (mean of local coefficients)**: mean of ``C_i`` over nodes with
  ``d_i >= 2`` (nodes of degree 0/1 are excluded from the mean). The mean over *all*
  nodes (with ``C_i = 0`` for ``d_i < 2``, the networkx convention) is also reported,
  clearly labelled, for comparison;
* **transitivity**: ``3 x #triangles / #connected triples`` — a different quantity;
* density of a simple undirected graph ``2m / (n (n - 1))``; directed ``m / (n (n - 1))``.

Everything here is exact; igraph's C core is used for clustering and components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd


def _distribution(values: np.ndarray, name: str) -> pd.DataFrame:
    values = np.asarray(values, dtype=np.int64)
    uniq, counts = np.unique(values, return_counts=True)
    return pd.DataFrame(
        {name: uniq, "count": counts, "fraction": counts / max(1, values.size)}
    )


def degree_distribution_table(degrees: np.ndarray) -> pd.DataFrame:
    """Number and fraction of nodes for every observed degree."""
    return _distribution(degrees, "degree")


def component_size_tables(sizes: np.ndarray, num_nodes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-component table (largest first) and size distribution table."""
    sizes = np.sort(np.asarray(sizes, dtype=np.int64))[::-1]
    per_component = pd.DataFrame(
        {
            "component_rank": np.arange(sizes.size),
            "size": sizes,
            "fraction_of_nodes": sizes / max(1, num_nodes),
        }
    )
    dist = _distribution(sizes, "size").rename(
        columns={"count": "num_components", "fraction": "fraction_of_components"}
    )
    dist["num_nodes_in_such_components"] = dist["size"] * dist["num_components"]
    return per_component, dist


def _degree_stats(degrees: np.ndarray, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_min": int(degrees.min()) if degrees.size else 0,
        f"{prefix}_max": int(degrees.max()) if degrees.size else 0,
        f"{prefix}_median": float(np.median(degrees)) if degrees.size else 0.0,
        f"{prefix}_std": float(degrees.std()) if degrees.size else 0.0,
    }


@dataclass
class UndirectedAnalysis:
    """Metrics of G_u plus the per-node arrays needed for tables and figures."""

    summary: dict[str, Any]
    degrees: np.ndarray
    local_clustering: np.ndarray | None
    triangles: np.ndarray | None
    membership: np.ndarray
    component_sizes: np.ndarray
    largest_component_nodes: np.ndarray


@dataclass
class DirectedAnalysis:
    """Metrics of G_d plus degree arrays and component sizes."""

    summary: dict[str, Any]
    in_degrees: np.ndarray
    out_degrees: np.ndarray
    weak_component_sizes: np.ndarray | None = None
    strong_component_sizes: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def local_clustering(graph: ig.Graph, degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local clustering (``0`` for degree < 2) and number of triangles of every node."""
    coeff = np.asarray(graph.transitivity_local_undirected(mode="zero"), dtype=np.float64)
    pairs = degrees.astype(np.float64) * (degrees - 1) / 2.0
    triangles = np.rint(coeff * pairs).astype(np.int64)
    return coeff, triangles


def analyze_undirected_graph(graph: ig.Graph, options: dict[str, Any]) -> UndirectedAnalysis:
    """Compute every mandatory F1 metric of ``G_u`` except distances/diameter.

    Args:
        graph: simple undirected igraph graph.
        options: ``metrics.undirected`` section of the configuration.
    """
    if graph.is_directed():
        raise ValueError("analyze_undirected_graph expects an undirected graph")
    n, m = graph.vcount(), graph.ecount()
    degrees = np.asarray(graph.degree(), dtype=np.int64)
    summary: dict[str, Any] = {
        "num_nodes": n,
        "num_edges": m,
        "mean_degree": 2.0 * m / n if n else 0.0,
        **_degree_stats(degrees, "degree"),
        "num_isolated_nodes": int(np.count_nonzero(degrees == 0)),
        "num_nodes_degree_lt_2": int(np.count_nonzero(degrees < 2)),
        "is_simple": bool(graph.is_simple()),
    }
    if options.get("compute_density", True):
        summary["density"] = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0

    coeff = triangles = None
    if options.get("compute_local_clustering", True) or options.get(
        "compute_global_average_clustering", True
    ):
        coeff, triangles = local_clustering(graph, degrees)
        eligible = degrees >= 2
        summary["num_triangles"] = int(triangles.sum() // 3)
        summary["num_connected_triples"] = int((degrees * (degrees - 1) // 2).sum())
        if options.get("compute_global_average_clustering", True):
            summary["average_local_clustering"] = (
                float(coeff[eligible].mean()) if eligible.any() else 0.0
            )
            summary["average_local_clustering_convention"] = (
                "mean of C_i over nodes with degree >= 2 (course convention)"
            )
            summary["average_local_clustering_all_nodes"] = float(coeff.mean()) if n else 0.0
            summary["average_local_clustering_all_nodes_convention"] = (
                "mean of C_i over all nodes with C_i = 0 for degree < 2 (networkx convention)"
            )
            summary["num_nodes_in_clustering_average"] = int(eligible.sum())
    if options.get("compute_transitivity", True):
        transitivity = graph.transitivity_undirected()
        summary["transitivity"] = 0.0 if np.isnan(transitivity) else float(transitivity)

    components = graph.connected_components()
    membership = np.asarray(components.membership, dtype=np.int64)
    sizes = np.asarray(components.sizes(), dtype=np.int64)
    largest = int(np.argmax(sizes)) if sizes.size else 0
    lcc_nodes = np.flatnonzero(membership == largest)
    if options.get("compute_components", True):
        summary.update(
            {
                "num_connected_components": int(sizes.size),
                "is_connected": bool(sizes.size == 1),
                "largest_component_size": int(lcc_nodes.size),
                "largest_component_fraction": float(lcc_nodes.size / n) if n else 0.0,
                "num_singleton_components": int(np.count_nonzero(sizes == 1)),
                "second_largest_component_size": int(np.sort(sizes)[-2]) if sizes.size > 1 else 0,
            }
        )
    return UndirectedAnalysis(
        summary=summary,
        degrees=degrees,
        local_clustering=coeff,
        triangles=triangles,
        membership=membership,
        component_sizes=sizes,
        largest_component_nodes=lcc_nodes,
    )


def analyze_directed_graph(graph: ig.Graph, options: dict[str, Any]) -> DirectedAnalysis:
    """Complementary metrics of ``G_d``: degrees, reciprocity, weak/strong components.

    Args:
        graph: directed igraph graph (edge ``u -> v``: u cites v).
        options: ``metrics.directed`` section of the configuration.
    """
    if not graph.is_directed():
        raise ValueError("analyze_directed_graph expects a directed graph")
    n, m = graph.vcount(), graph.ecount()
    in_deg = np.asarray(graph.indegree(), dtype=np.int64)
    out_deg = np.asarray(graph.outdegree(), dtype=np.int64)
    summary: dict[str, Any] = {
        "num_nodes": n,
        "num_edges": m,
        "mean_in_degree": m / n if n else 0.0,
        "mean_out_degree": m / n if n else 0.0,
        **_degree_stats(in_deg, "in_degree"),
        **_degree_stats(out_deg, "out_degree"),
        "density": m / (n * (n - 1)) if n > 1 else 0.0,
        "num_nodes_in_degree_0": int(np.count_nonzero(in_deg == 0)),
        "num_nodes_out_degree_0": int(np.count_nonzero(out_deg == 0)),
        "num_nodes_total_degree_0": int(np.count_nonzero(in_deg + out_deg == 0)),
        "reciprocity": float(graph.reciprocity(ignore_loops=True, mode="default")),
        "is_dag": bool(graph.is_dag()),
    }
    result = DirectedAnalysis(summary=summary, in_degrees=in_deg, out_degrees=out_deg)
    if options.get("compute_weak_components", True):
        sizes = np.asarray(graph.connected_components(mode="weak").sizes(), dtype=np.int64)
        result.weak_component_sizes = sizes
        summary.update(
            {
                "num_weakly_connected_components": int(sizes.size),
                "largest_wcc_size": int(sizes.max()) if sizes.size else 0,
                "largest_wcc_fraction": float(sizes.max() / n) if n else 0.0,
            }
        )
    if options.get("compute_strong_components", True):
        sizes = np.asarray(graph.connected_components(mode="strong").sizes(), dtype=np.int64)
        result.strong_component_sizes = sizes
        nontrivial = sizes[sizes > 1]
        summary.update(
            {
                "num_strongly_connected_components": int(sizes.size),
                "num_nontrivial_sccs": int(nontrivial.size),
                "num_nodes_in_nontrivial_sccs": int(nontrivial.sum()),
                "largest_scc_size": int(sizes.max()) if sizes.size else 0,
                "largest_scc_fraction": float(sizes.max() / n) if n else 0.0,
            }
        )
    return result


# --------------------------------------------------------------------------- areas


def category_citation_matrix(
    edges: np.ndarray, labels: np.ndarray, num_categories: int
) -> np.ndarray:
    """``M[i, j]`` = number of citations from papers of area ``i`` to papers of area ``j``.

    This is the weighted adjacency of the aggregated area graph ``G_A`` (directed,
    self-loops ``i -> i`` = intra-area citations are kept).
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    keys = labels[edges[:, 0]] * num_categories + labels[edges[:, 1]]
    counts = np.bincount(keys, minlength=num_categories * num_categories)
    return counts.reshape(num_categories, num_categories).astype(np.int64)


def area_summary_table(
    matrix: np.ndarray, labels: np.ndarray, category_names: list[str]
) -> pd.DataFrame:
    """Per-area counts: papers, citations made/received and intra-area share."""
    k = matrix.shape[0]
    made = matrix.sum(axis=1)
    received = matrix.sum(axis=0)
    intra = np.diag(matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame(
            {
                "category": np.arange(k),
                "category_name": category_names[:k],
                "num_papers": np.bincount(labels, minlength=k),
                "citations_made": made,
                "citations_received": received,
                "intra_area_citations": intra,
                "intra_fraction_of_made": np.where(made > 0, intra / made, np.nan),
                "intra_fraction_of_received": np.where(received > 0, intra / received, np.nan),
            }
        )
