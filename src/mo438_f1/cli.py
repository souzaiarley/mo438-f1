"""Command-line interface: ``mo438-f1 run`` / ``mo438-f1 download``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the ``mo438-f1`` command."""
    parser = argparse.ArgumentParser(
        prog="mo438-f1",
        description="MO438 F1 pipeline: structural analysis of the ogbn-arxiv citation network.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="Run the full F1 pipeline (data, graphs, GraphML, metrics, figures)",
        description="Run the full F1 pipeline. Options override the YAML configuration.",
    )
    run.add_argument("--config", default="config/default.yaml", help="YAML configuration file")
    run.add_argument("--force-download", action="store_true",
                     help="re-download the raw dataset even if cached")
    run.add_argument("--force-rebuild", action="store_true",
                     help="re-parse the dataset and rewrite the GraphML files")
    run.add_argument("--force-recompute", action="store_true",
                     help="ignore cached distance results")
    run.add_argument("--paths-mode", choices=["exact", "approximate", "auto"],
                     help="distance/diameter computation mode (default: from config)")
    run.add_argument("--num-sources", type=int,
                     help="number of sampled BFS sources for the approximate mode")
    run.add_argument("--n-jobs", help="worker processes for distances ('auto' or an integer)")
    run.add_argument("--no-validation", action="store_true",
                     help="skip the exact-vs-approximate validation table")

    download = subparsers.add_parser(
        "download", help="Only download/cache the raw ogbn-arxiv archive (needs internet)")
    download.add_argument("--config", default="config/default.yaml")
    download.add_argument("--force", action="store_true", help="re-download even if cached")
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {"runtime": {}, "paths": {"approximation": {}}}
    if args.force_download:
        overrides["runtime"]["force_download"] = True
    if args.force_rebuild:
        overrides["runtime"]["force_rebuild_graphs"] = True
    if args.force_recompute:
        overrides["runtime"]["force_recompute_metrics"] = True
    if args.n_jobs is not None:
        overrides["runtime"]["n_jobs"] = args.n_jobs if args.n_jobs == "auto" else int(args.n_jobs)
    if args.paths_mode:
        overrides["paths"]["mode"] = args.paths_mode
    if args.num_sources is not None:
        overrides["paths"]["approximation"]["num_sources"] = args.num_sources
    if args.no_validation:
        overrides["paths"]["approximation"]["validation_enabled"] = False
    return overrides


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``mo438-f1`` console script."""
    args = build_parser().parse_args(argv)
    if args.command == "run":
        from .pipeline import run_pipeline

        run_pipeline(args.config, _overrides(args))
        return 0
    if args.command == "download":
        from .config import load_config
        from .data import download_ogbn_arxiv
        from .runtime import setup_logging

        setup_logging(None)
        cfg = load_config(args.config)
        if cfg["data"]["dataset"] != "ogbn-arxiv":
            print("Nothing to download: the configured dataset is synthetic.")
            return 0
        path = download_ogbn_arxiv(Path(cfg["data"]["raw_dir"]), cfg["data"]["source_url"],
                                   force=args.force)
        print(path)
        return 0
    return 2  # pragma: no cover - argparse enforces the subcommand


if __name__ == "__main__":
    sys.exit(main())
