"""Configuration loading, defaults, environment and CLI overrides.

Relative paths in the YAML are resolved against the current working directory
(run the pipeline from the repository root). Two environment variables can
relocate the large artefacts, which is convenient on clusters with a separate
scratch filesystem:

* ``MO438_DATA_DIR``   — replaces the ``data/`` prefix of ``data.raw_dir`` and
  ``data.processed_dir``;
* ``MO438_OUTPUT_DIR`` — replaces the ``outputs/`` prefix of every output path.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "project": {"name": "mo438-f1", "seed": 438},
    "data": {
        "dataset": "ogbn-arxiv",
        "source_url": "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "graphml_dir": "outputs/graphml",
        "directed_graphml": "outputs/graphml/ogbn_arxiv_directed.graphml",
        "undirected_graphml": "outputs/graphml/ogbn_arxiv_undirected.graphml",
        "area_graphml": "outputs/graphml/ogbn_arxiv_area_graph.graphml",
        "remove_self_loops": True,
        "synthetic": {
            "num_nodes": 3000,
            "num_categories": 8,
            "mean_out_degree": 6.0,
            "intra_category_prob": 0.7,
            "preferential_prob": 0.4,
            "closure_prob": 0.3,
            "num_small_components": 12,
            "num_isolated": 5,
            "num_self_loops": 4,
            "num_reciprocal": 15,
            "num_duplicates": 6,
        },
    },
    "outputs": {
        "metrics_dir": "outputs/metrics",
        "figures_dir": "outputs/figures",
        "logs_dir": "outputs/logs",
    },
    "metrics": {
        "undirected": {
            "compute_local_clustering": True,
            "compute_global_average_clustering": True,
            "compute_transitivity": True,
            "compute_density": True,
            "compute_components": True,
            "compute_average_shortest_path": True,
            "compute_diameter": True,
        },
        "directed": {
            "compute_degree_distributions": True,
            "compute_weak_components": True,
            "compute_strong_components": True,
        },
        "compute_area_graph": True,
    },
    "paths": {
        "use_largest_connected_component_for_distances": True,
        "mode": "auto",
        "max_exact_seconds": 6 * 3600,
        "batch_words": 1,
        "approximation": {
            "num_sources": 512,
            "diameter_sweeps": 4,
            "validation_enabled": True,
            "validation_sizes": [500, 2000, 5000],
            "validation_num_sources": [16, 64, 256],
            "validation_repeats": 5,
        },
    },
    "plots": {
        "dpi": 180,
        "degree_log_x": True,
        "degree_log_y": True,
        "bins": 80,
        "sample_visualization": True,
        "sample_size": 1500,
        "area_graph_top_edges": 80,
    },
    "runtime": {
        "n_jobs": "auto",
        "force_download": False,
        "force_rebuild_graphs": False,
        "force_recompute_metrics": False,
    },
}

_OUTPUT_PATH_KEYS = (
    ("data", "graphml_dir"),
    ("data", "directed_graphml"),
    ("data", "undirected_graphml"),
    ("data", "area_graphml"),
    ("outputs", "metrics_dir"),
    ("outputs", "figures_dir"),
    ("outputs", "logs_dir"),
)
_DATA_PATH_KEYS = (("data", "raw_dir"), ("data", "processed_dir"))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` recursively updated with ``override``."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _rebase(path: str, old_prefix: str, new_root: str) -> str:
    parts = Path(path).parts
    if parts and parts[0] == old_prefix:
        return str(Path(new_root, *parts[1:]))
    return path


def apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Relocate data/output paths according to ``MO438_DATA_DIR``/``MO438_OUTPUT_DIR``."""
    config = copy.deepcopy(config)
    data_root = os.environ.get("MO438_DATA_DIR")
    output_root = os.environ.get("MO438_OUTPUT_DIR")
    if data_root:
        for section, key in _DATA_PATH_KEYS:
            config[section][key] = _rebase(config[section][key], "data", data_root)
    if output_root:
        for section, key in _OUTPUT_PATH_KEYS:
            config[section][key] = _rebase(config[section][key], "outputs", output_root)
    return config


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the YAML configuration, fill defaults and apply env/CLI overrides.

    Args:
        path: YAML file.
        overrides: nested dict merged on top of the file (e.g. from CLI flags).

    Raises:
        FileNotFoundError: if the configuration file does not exist.
        ValueError: if a value is invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    config = deep_merge(DEFAULTS, user_config)
    if overrides:
        config = deep_merge(config, overrides)
    config = apply_env_overrides(config)
    validate_config(config)
    config["_config_path"] = str(path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Raise ``ValueError`` for invalid configuration values."""
    dataset = config["data"]["dataset"]
    if dataset not in {"ogbn-arxiv", "synthetic"}:
        raise ValueError(f"data.dataset must be 'ogbn-arxiv' or 'synthetic', got {dataset!r}")
    mode = config["paths"]["mode"]
    if mode not in {"exact", "approximate", "auto"}:
        raise ValueError(f"paths.mode must be exact|approximate|auto, got {mode!r}")
    if int(config["paths"]["batch_words"]) < 1:
        raise ValueError("paths.batch_words must be >= 1")
    if int(config["paths"]["approximation"]["num_sources"]) < 1:
        raise ValueError("paths.approximation.num_sources must be >= 1")
    n_jobs = config["runtime"]["n_jobs"]
    if n_jobs != "auto" and int(n_jobs) < 1:
        raise ValueError("runtime.n_jobs must be 'auto' or a positive integer")


def resolve_n_jobs(value: Any) -> int:
    """Translate ``runtime.n_jobs`` into a worker count.

    ``auto`` uses ``SLURM_CPUS_PER_TASK`` when running under Slurm, otherwise the
    number of CPUs available to this process.
    """
    if value != "auto":
        return max(1, int(value))
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return max(1, int(slurm))
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except AttributeError:
        return max(1, os.cpu_count() or 1)
