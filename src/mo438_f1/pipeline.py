"""End-to-end F1 pipeline orchestration.

Stages: load data -> build G_d / G_u -> GraphML export -> G_u metrics -> distances on
the largest connected component (exact or documented approximation) -> validation of
the approximation -> G_d metrics -> aggregated area graph -> figures -> structured
outputs (JSON/CSV) and log.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__, plots
from .config import load_config, resolve_n_jobs
from .data import CitationData, load_dataset
from .graph_build import (
    build_directed_graph,
    build_undirected_graph,
    csr_from_undirected_edges,
    induced_subgraph_csr,
)
from .graph_io import write_graphml
from .metrics import (
    analyze_directed_graph,
    analyze_undirected_graph,
    area_summary_table,
    category_citation_matrix,
    component_size_tables,
    degree_distribution_table,
)
from .paths import (
    DistanceResult,
    compute_distance_stats,
    sampled_distance_stats,
    summarize_validation,
    validate_approximation,
    validation_row,
)
from .runtime import StageTimer, get_logger, library_versions, peak_memory_mb, setup_logging

_DISTANCE_CACHE_VERSION = 1


# --------------------------------------------------------------------------- helpers


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def _clean_nan(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_nan(v) for v in value]
    return value


def write_json(payload: dict[str, Any], path: Path) -> Path:
    """Write ``payload`` as pretty JSON (NaN -> null)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_clean_nan(payload), indent=2, default=_json_default, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _fingerprint(uedges: np.ndarray, num_nodes: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(num_nodes).encode())
    digest.update(np.ascontiguousarray(uedges, dtype=np.int64).tobytes())
    return digest.hexdigest()[:16]


def _distance_cache_key(fingerprint: str, cfg: dict[str, Any], seed: int) -> str:
    paths_cfg = cfg["paths"]
    relevant = {
        "version": _DISTANCE_CACHE_VERSION,
        "graph": fingerprint,
        "mode": paths_cfg["mode"],
        "max_exact_seconds": paths_cfg["max_exact_seconds"] if paths_cfg["mode"] == "auto" else None,
        "num_sources": paths_cfg["approximation"]["num_sources"],
        "diameter_sweeps": paths_cfg["approximation"]["diameter_sweeps"],
        "seed": seed,
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def _graphml_node_attributes(data: CitationData) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "node_id": np.arange(data.num_nodes),
        "category": data.labels,
        "category_name": [data.category_names[k] for k in data.labels],
    }
    if data.years is not None:
        attrs["year"] = data.years
    if data.paper_ids is not None:
        attrs["paper_id"] = data.paper_ids
    return attrs


def _export_graphml(path: Path, force: bool, **kwargs: Any) -> dict[str, Any]:
    logger = get_logger()
    if path.exists() and not force:
        logger.info("GraphML %s already exists (use --force-rebuild to rewrite)", path)
        status = "cached"
    else:
        write_graphml(path, **kwargs)
        status = "written"
        logger.info("Wrote %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return {"path": str(path), "status": status, "size_bytes": path.stat().st_size}


def _engine_matches_igraph(table: pd.DataFrame) -> bool | None:
    """True if every exact MS-BFS value in the validation table equals igraph's."""
    if table.empty or "igraph_average_distance" not in table:
        return None
    checked = table[table["igraph_average_distance"].notna()]
    if checked.empty:
        return None
    return bool(
        np.allclose(checked["exact_average_distance"].astype(float),
                    checked["igraph_average_distance"].astype(float))
        and (checked["exact_diameter"].astype(int) == checked["igraph_diameter"].astype(int)).all()
    )


def _summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flat, report-friendly rows for summary.csv."""
    rows: list[dict[str, Any]] = []

    def add(graph: str, metric: str, value: Any, exact: bool = True, note: str = "") -> None:
        if value is None:
            return
        rows.append({"graph": graph, "metric": metric, "value": value, "is_exact": exact,
                     "note": note})

    u = summary["undirected"]
    add("G_u", "num_nodes", u["num_nodes"])
    add("G_u", "num_edges", u["num_edges"])
    add("G_u", "mean_degree", u["mean_degree"], note="2m/n")
    add("G_u", "max_degree", u["degree_max"])
    add("G_u", "density", u.get("density"), note="2m/(n(n-1))")
    add("G_u", "average_local_clustering", u.get("average_local_clustering"),
        note=u.get("average_local_clustering_convention", ""))
    add("G_u", "average_local_clustering_all_nodes", u.get("average_local_clustering_all_nodes"),
        note=u.get("average_local_clustering_all_nodes_convention", ""))
    add("G_u", "transitivity", u.get("transitivity"), note="3 x triangles / connected triples")
    add("G_u", "num_triangles", u.get("num_triangles"))
    add("G_u", "num_connected_components", u.get("num_connected_components"))
    add("G_u", "largest_component_size", u.get("largest_component_size"))
    add("G_u", "largest_component_fraction", u.get("largest_component_fraction"))
    add("G_u", "num_isolated_nodes", u.get("num_isolated_nodes"))
    dist = u.get("distances")
    if dist:
        exact = dist["is_exact"]
        scope = (f"on {dist['component']} ({dist['component_size']} nodes, "
                 f"{dist['component_fraction']:.4%} of all nodes); {dist['method']}")
        add("G_u", "average_distance", dist["average_distance"], exact, scope)
        if not exact:
            add("G_u", "average_distance_stderr", dist.get("average_distance_stderr"), False,
                f"{dist['num_sources']} sampled sources, seed {dist['params'].get('seed')}")
        add("G_u", "diameter", dist.get("diameter"), exact, scope)
        if not exact:
            add("G_u", "diameter_lower_bound", dist["diameter_lower_bound"], False, scope)
            add("G_u", "diameter_upper_bound", dist["diameter_upper_bound"], False, scope)
    d = summary.get("directed") or {}
    if d:
        add("G_d", "num_nodes", d["num_nodes"])
        add("G_d", "num_edges", d["num_edges"])
        add("G_d", "mean_in_degree", d["mean_in_degree"], note="m/n")
        add("G_d", "mean_out_degree", d["mean_out_degree"], note="m/n")
        add("G_d", "max_in_degree", d["in_degree_max"])
        add("G_d", "max_out_degree", d["out_degree_max"])
        add("G_d", "density", d["density"], note="m/(n(n-1))")
        add("G_d", "reciprocity", d["reciprocity"])
        add("G_d", "is_dag", d["is_dag"])
        add("G_d", "num_weakly_connected_components", d.get("num_weakly_connected_components"))
        add("G_d", "largest_wcc_size", d.get("largest_wcc_size"))
        add("G_d", "num_strongly_connected_components", d.get("num_strongly_connected_components"))
        add("G_d", "num_nontrivial_sccs", d.get("num_nontrivial_sccs"))
        add("G_d", "largest_scc_size", d.get("largest_scc_size"))
    c = summary["conversion"]
    add("G_d->G_u", "directed_self_loops", c["num_directed_self_loops"])
    add("G_d->G_u", "duplicate_directed_edges", c["num_duplicate_directed_edges"])
    add("G_d->G_u", "reciprocal_pairs_merged", c["num_reciprocal_pairs"])
    return rows


# --------------------------------------------------------------------------- pipeline


def run_pipeline(config_path: str | Path, overrides: dict[str, Any] | None = None
                 ) -> dict[str, Any]:
    """Run the complete F1 analysis pipeline and return the summary dictionary.

    Args:
        config_path: YAML configuration file.
        overrides: nested dict merged over the configuration (CLI flags).
    """
    cfg = load_config(config_path, overrides)
    seed = int(cfg["project"]["seed"])
    data_cfg, out_cfg, runtime = cfg["data"], cfg["outputs"], cfg["runtime"]
    metrics_dir = Path(out_cfg["metrics_dir"])
    figures_dir = Path(out_cfg["figures_dir"])
    logs_dir = Path(out_cfg["logs_dir"])
    processed_dir = Path(data_cfg["processed_dir"])
    for directory in (metrics_dir, figures_dir, logs_dir, processed_dir,
                      Path(data_cfg["graphml_dir"])):
        directory.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(logs_dir / "run.log")
    timer = StageTimer()
    n_jobs = resolve_n_jobs(runtime["n_jobs"])
    dpi = int(cfg["plots"]["dpi"])
    force_rebuild = bool(runtime["force_rebuild_graphs"])
    force_recompute = bool(runtime["force_recompute_metrics"])
    logger.info("mo438-f1 %s | config %s | dataset %s | seed %d | n_jobs %d",
                __version__, config_path, data_cfg["dataset"], seed, n_jobs)
    np.random.seed(seed)  # only for libraries relying on the global RNG
    files: dict[str, str] = {}
    summary: dict[str, Any] = {
        "project": cfg["project"]["name"],
        "package_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "seed": seed,
        "environment": library_versions(),
    }

    # ---------------------------------------------------------------- data + graphs
    with timer.stage("load dataset"):
        data = load_dataset(cfg)
    logger.info("Dataset %s: %d nodes, %d directed edges, %d categories",
                data.name, data.num_nodes, data.num_edges, len(data.category_names))
    summary["dataset"] = {
        "name": data.name,
        "num_nodes": data.num_nodes,
        "num_directed_edges": data.num_edges,
        "num_categories": len(data.category_names),
        "category_names": data.category_names,
        "source": data.source,
    }

    with timer.stage("build G_d and G_u"):
        g_d = build_directed_graph(data)
        g_u, conversion, uedges = build_undirected_graph(
            data, remove_self_loops=bool(data_cfg["remove_self_loops"])
        )
    summary["conversion"] = {
        **conversion.to_dict(),
        "rules": [
            "orientation removed: u->v and v->u become {u, v}",
            "parallel/duplicate pairs consolidated into a single undirected edge (no weights)",
            "self-loops removed" if conversion.self_loops_removed else "self-loops kept",
            "all nodes kept (isolated nodes included), same indices and attributes as G_d",
        ],
    }
    logger.info("G_u: %d edges (self-loops %d, duplicates %d, reciprocal pairs merged %d)",
                conversion.num_undirected_edges, conversion.num_directed_self_loops,
                conversion.num_duplicate_directed_edges, conversion.num_reciprocal_pairs)

    with timer.stage("export GraphML"):
        node_attrs = _graphml_node_attributes(data)
        summary["graphml"] = {
            "directed": _export_graphml(
                Path(data_cfg["directed_graphml"]), force_rebuild, num_nodes=data.num_nodes,
                edges=data.edges, directed=True, node_attributes=node_attrs,
                graph_attributes={"name": f"{data.name}_directed",
                                  "edge_semantics": "u -> v : paper u cites paper v"}),
            "undirected": _export_graphml(
                Path(data_cfg["undirected_graphml"]), force_rebuild, num_nodes=data.num_nodes,
                edges=uedges, directed=False, node_attributes=node_attrs,
                graph_attributes={"name": f"{data.name}_undirected",
                                  "edge_semantics": "simple undirected citation graph G_u"}),
        }

    # ---------------------------------------------------------------- G_u metrics
    with timer.stage("G_u metrics (degree, clustering, density, components)"):
        und = analyze_undirected_graph(g_u, cfg["metrics"]["undirected"])
    summary["undirected"] = dict(und.summary)
    logger.info("G_u: <k>=%.4f density=%.3e avgC=%.4f transitivity=%.4f components=%d "
                "LCC=%d (%.2f%%)", und.summary["mean_degree"], und.summary.get("density", 0),
                und.summary.get("average_local_clustering", float("nan")),
                und.summary.get("transitivity", float("nan")),
                und.summary["num_connected_components"], und.summary["largest_component_size"],
                100 * und.summary["largest_component_fraction"])

    degree_table = degree_distribution_table(und.degrees)
    files["degree_distribution_undirected_csv"] = str(metrics_dir / "degree_distribution_undirected.csv")
    degree_table.to_csv(files["degree_distribution_undirected_csv"], index=False)
    if und.local_clustering is not None:
        local_df = pd.DataFrame({
            "node_id": np.arange(data.num_nodes),
            "category": [data.category_names[k] for k in data.labels],
            "degree": und.degrees,
            "clustering": und.local_clustering,
            "category_idx": data.labels,
            "triangles": und.triangles,
        })
        files["local_clustering_csv"] = str(metrics_dir / "local_clustering.csv")
        local_df.to_csv(files["local_clustering_csv"], index=False, float_format="%.6g")
        del local_df
    per_component, size_dist = component_size_tables(und.component_sizes, data.num_nodes)
    files["component_sizes_csv"] = str(metrics_dir / "component_sizes.csv")
    per_component.to_csv(files["component_sizes_csv"], index=False)
    files["component_size_distribution_csv"] = str(metrics_dir / "component_size_distribution.csv")
    size_dist.to_csv(files["component_size_distribution_csv"], index=False)

    # ---------------------------------------------------------------- distances
    paths_cfg = cfg["paths"]
    approx_cfg = paths_cfg["approximation"]
    undirected_opts = cfg["metrics"]["undirected"]
    distance: DistanceResult | None = None
    lcc_adj = None
    if undirected_opts.get("compute_average_shortest_path", True) or undirected_opts.get(
        "compute_diameter", True
    ):
        with timer.stage("distances on the largest connected component"):
            full_adj = csr_from_undirected_edges(uedges, data.num_nodes)
            if paths_cfg["use_largest_connected_component_for_distances"] or \
                    und.summary["num_connected_components"] > 1:
                lcc_adj = induced_subgraph_csr(full_adj, und.largest_component_nodes)
                component = "largest connected component of G_u"
            else:
                lcc_adj = full_adj
                component = "G_u (connected)"
            del full_adj
            cache_dir = processed_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            fingerprint = _fingerprint(uedges, data.num_nodes)
            cache_file = cache_dir / f"distances_{_distance_cache_key(fingerprint, cfg, seed)}.json"
            decision: dict[str, Any]
            if cache_file.exists() and not force_recompute:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                distance = DistanceResult.from_dict(cached["result"])
                decision = {**cached["decision"], "loaded_from_cache": str(cache_file)}
                logger.info("Loaded distance results from cache %s", cache_file)
            else:
                distance, decision = compute_distance_stats(
                    lcc_adj,
                    mode=paths_cfg["mode"],
                    num_sources=int(approx_cfg["num_sources"]),
                    seed=seed,
                    batch_words=int(paths_cfg["batch_words"]),
                    n_jobs=n_jobs,
                    max_exact_seconds=float(paths_cfg["max_exact_seconds"]),
                    diameter_sweeps=int(approx_cfg["diameter_sweeps"]),
                )
                write_json({"result": distance.to_dict(), "decision": decision,
                            "graph_fingerprint": fingerprint}, cache_file)
        lcc_size = int(und.largest_component_nodes.size)
        summary["undirected"]["distances"] = {
            "component": component,
            "component_size": lcc_size,
            "component_fraction": lcc_size / data.num_nodes,
            "component_num_edges": distance.num_edges,
            **{k: v for k, v in distance.to_dict().items()
               if k != "distance_histogram_ordered_pairs"},
            "decision": decision,
        }
        logger.info("Distances (%s) on %s (%d nodes): <d>=%.4f, diameter=%s [%d, %d]",
                    "EXACT" if distance.is_exact else "APPROXIMATE", component, lcc_size,
                    distance.average_distance, distance.diameter,
                    distance.diameter_lower_bound, distance.diameter_upper_bound)
        hist = distance.histogram
        hist_df = pd.DataFrame({"distance": np.arange(1, hist.size), "num_ordered_pairs": hist[1:],
                                "fraction": hist[1:] / max(1, hist[1:].sum())})
        hist_df["is_exact"] = distance.is_exact
        files["distance_distribution_csv"] = str(metrics_dir / "distance_distribution.csv")
        hist_df.to_csv(files["distance_distribution_csv"], index=False)

    # ---------------------------------------------------------------- validation
    if lcc_adj is not None and distance is not None and approx_cfg["validation_enabled"]:
        with timer.stage("validation of the distance approximation"):
            table = validate_approximation(
                lcc_adj,
                sizes=[int(s) for s in approx_cfg["validation_sizes"]],
                num_sources_list=[int(k) for k in approx_cfg["validation_num_sources"]],
                repeats=int(approx_cfg["validation_repeats"]),
                seed=seed,
                batch_words=int(paths_cfg["batch_words"]),
                diameter_sweeps=int(approx_cfg["diameter_sweeps"]),
            )
            if distance.is_exact:
                # Also compare the production approximation with the exact full-LCC values.
                approx_full = sampled_distance_stats(
                    lcc_adj, int(approx_cfg["num_sources"]), seed,
                    int(paths_cfg["batch_words"]), n_jobs, int(approx_cfg["diameter_sweeps"]))
                table = pd.concat([table, pd.DataFrame([validation_row(
                    "largest_component", lcc_adj, distance, approx_full, 0)])], ignore_index=True)
            files["path_validation_csv"] = str(metrics_dir / "path_approximation_validation.csv")
            table.to_csv(files["path_validation_csv"], index=False, float_format="%.6g")
            agg = summarize_validation(table)
            files["path_validation_summary_csv"] = str(
                metrics_dir / "path_approximation_validation_summary.csv")
            agg.to_csv(files["path_validation_summary_csv"], index=False, float_format="%.6g")
        summary["path_approximation_validation"] = {
            "purpose": "exact vs. sampled-source estimates on connected BFS-ball subgraphs"
                       + (" and on the full largest component" if distance.is_exact else ""),
            "exact_engine_crosscheck": "exact values also recomputed with igraph path_length_hist",
            "engine_matches_igraph": _engine_matches_igraph(table),
            "params": {k: approx_cfg[k] for k in ("validation_sizes", "validation_num_sources",
                                                  "validation_repeats", "diameter_sweeps")},
            "seed": seed,
            "summary": agg.to_dict(orient="records"),
        }
    del lcc_adj

    # ---------------------------------------------------------------- G_d metrics
    with timer.stage("G_d metrics (in/out degree, weak/strong components)"):
        dirm = analyze_directed_graph(g_d, cfg["metrics"]["directed"])
    summary["directed"] = dirm.summary
    del g_d
    for name, values in (("in_degree", dirm.in_degrees), ("out_degree", dirm.out_degrees)):
        key = f"{name}_distribution_directed_csv"
        files[key] = str(metrics_dir / f"{name}_distribution_directed.csv")
        degree_distribution_table(values).rename(columns={"degree": name}).to_csv(
            files[key], index=False)
    for kind, sizes in (("weak", dirm.weak_component_sizes),
                        ("strong", dirm.strong_component_sizes)):
        if sizes is not None:
            _, dist_table = component_size_tables(sizes, data.num_nodes)
            key = f"{kind}_component_size_distribution_csv"
            files[key] = str(metrics_dir / f"{kind}_component_size_distribution_directed.csv")
            dist_table.to_csv(files[key], index=False)

    # ---------------------------------------------------------------- area graph
    matrix = None
    if cfg["metrics"].get("compute_area_graph", True):
        with timer.stage("aggregated area graph G_A"):
            k = len(data.category_names)
            matrix = category_citation_matrix(data.edges, data.labels, k)
            matrix_df = pd.DataFrame(matrix, index=data.category_names,
                                     columns=data.category_names)
            matrix_df.index.name = "citing\\cited"
            files["area_citation_matrix_csv"] = str(metrics_dir / "area_citation_matrix.csv")
            matrix_df.to_csv(files["area_citation_matrix_csv"])
            area_df = area_summary_table(matrix, data.labels, data.category_names)
            files["area_summary_csv"] = str(metrics_dir / "area_summary.csv")
            area_df.to_csv(files["area_summary_csv"], index=False, float_format="%.6g")
            rows, cols = np.nonzero(matrix)
            summary["graphml"]["area_graph"] = _export_graphml(
                Path(data_cfg["area_graphml"]), True, num_nodes=k,
                edges=np.column_stack([rows, cols]), directed=True,
                node_attributes={"category": np.arange(k), "category_name": data.category_names,
                                 "num_papers": area_df["num_papers"].to_numpy()},
                edge_attributes={"weight": matrix[rows, cols]},
                graph_attributes={"name": f"{data.name}_area_graph",
                                  "edge_semantics": "i -> j weight = citations from area i to "
                                                    "area j (self-loops = intra-area)"})
            total = int(matrix.sum())
            summary["areas"] = {
                "num_areas": k,
                "total_citations": total,
                "intra_area_citations": int(np.trace(matrix)),
                "intra_area_fraction": float(np.trace(matrix) / total) if total else 0.0,
                "num_area_pairs_with_citations": int(np.count_nonzero(matrix)),
            }

    # ---------------------------------------------------------------- figures
    with timer.stage("figures"):
        p = cfg["plots"]
        figs: dict[str, Path] = {}
        figs["degree_distribution_undirected"] = plots.plot_degree_distribution(
            und.degrees, figures_dir / "degree_distribution_undirected.png", dpi,
            f"Degree distribution of G_u ({data.name}, n = {data.num_nodes:,})",
            log_x=bool(p["degree_log_x"]), log_y=bool(p["degree_log_y"]))
        if und.local_clustering is not None:
            figs["local_clustering_distribution"] = plots.plot_local_clustering(
                und.local_clustering, und.degrees,
                figures_dir / "local_clustering_distribution.png", dpi, int(p["bins"]),
                float(und.summary.get("average_local_clustering", float("nan"))),
                float(und.summary.get("transitivity", float("nan"))),
                f"Local clustering coefficients of G_u ({data.name})")
        if und.component_sizes.size > 1:
            figs["component_size_distribution"] = plots.plot_component_sizes(
                und.component_sizes, figures_dir / "component_size_distribution.png", dpi,
                f"Connected component sizes of G_u ({data.name})")
        else:
            logger.info("G_u is connected: component size figure not generated")
        figs["in_out_degree_distribution_directed"] = plots.plot_in_out_degree(
            dirm.in_degrees, dirm.out_degrees,
            figures_dir / "in_out_degree_distribution_directed.png", dpi,
            f"In- and out-degree distributions of G_d ({data.name})")
        if distance is not None:
            figs["distance_distribution"] = plots.plot_distance_distribution(
                distance.histogram, figures_dir / "distance_distribution.png", dpi,
                f"Shortest-path distances, largest component of G_u ({data.name})",
                distance.is_exact, distance.average_distance)
        if matrix is not None:
            area_df = area_summary_table(matrix, data.labels, data.category_names)
            figs["area_citation_heatmap"] = plots.plot_area_heatmap(
                matrix, data.category_names, figures_dir / "area_citation_heatmap.png", dpi,
                f"Citations between primary categories ({data.name})")
            figs["area_graph"] = plots.plot_area_graph(
                matrix, data.category_names, area_df["num_papers"].to_numpy(),
                figures_dir / "area_graph.png", dpi, int(p["area_graph_top_edges"]),
                f"Aggregated area graph G_A ({data.name}): top {p['area_graph_top_edges']} "
                f"inter-area flows")
        if p.get("sample_visualization", True):
            figs["sample_graph"] = _sample_visualization(
                data, uedges, und.largest_component_nodes, int(p["sample_size"]), seed,
                figures_dir / "sample_graph.png", dpi)
        files.update({f"{k}_png": str(v) for k, v in figs.items()})

    # ---------------------------------------------------------------- outputs
    summary["timings_seconds"] = timer.durations
    summary["peak_memory_mib"] = peak_memory_mb()
    summary["peak_memory_largest_worker_mib"] = peak_memory_mb(children=True)
    summary["n_jobs"] = n_jobs
    summary["config"] = {k: v for k, v in cfg.items() if not k.startswith("_")}
    files["summary_json"] = str(metrics_dir / "summary.json")
    files["summary_csv"] = str(metrics_dir / "summary.csv")
    files["run_log"] = str(logs_dir / "run.log")
    summary["outputs"] = files
    pd.DataFrame(_summary_rows(summary)).to_csv(files["summary_csv"], index=False)
    write_json(summary, Path(files["summary_json"]))
    logger.info("Wrote %s and %s", files["summary_json"], files["summary_csv"])
    logger.info("Pipeline finished in %.1f s (peak RSS %.0f MiB)",
                sum(timer.durations.values()), summary["peak_memory_mib"] or float("nan"))
    return summary


def _sample_visualization(data: CitationData, uedges: np.ndarray, lcc_nodes: np.ndarray,
                          size: int, seed: int, path: Path, dpi: int) -> Path:
    """Draw the BFS ball of ``size`` nodes around a random node of the largest component."""
    from .paths import bfs_ball

    adjacency = csr_from_undirected_edges(uedges, data.num_nodes)
    rng = np.random.default_rng(seed)
    root = int(rng.choice(lcc_nodes))
    nodes = bfs_ball(adjacency.indptr.astype(np.int64), adjacency.indices.astype(np.int64),
                     root, size)
    sub = induced_subgraph_csr(adjacency, nodes)
    upper = sub.tocoo()
    mask = upper.row < upper.col
    edges = np.column_stack([upper.row[mask], upper.col[mask]])
    return plots.plot_sample_graph(
        edges, data.labels[nodes], data.category_names, path, dpi,
        f"Sample of G_u: BFS ball of {nodes.size:,} nodes around node {root} ({data.name})",
        seed)

