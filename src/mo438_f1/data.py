"""Dataset loading utilities (ogbn-arxiv and a small synthetic citation graph).

The ogbn-arxiv archive is downloaded once from the official OGB URL into
``data/raw/`` and parsed directly from the zip (only the edge list, labels,
years and id mappings are read; the 128-d node features are never extracted).
The parsed arrays are cached as a compressed ``.npz`` in ``data/processed/``.

Edge orientation: OGB documents that in ogbn-arxiv "each directed edge indicates
that one paper cites another one"; the first column of ``raw/edge.csv.gz`` is the
citing paper and the second one the cited paper (``edge_index[0] -> edge_index[1]``
in the OGB loaders). We keep that orientation: a row ``(u, v)`` means *u cites v*.

No data is downloaded at import time.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .runtime import get_logger

OGBN_ARXIV_NUM_NODES = 169_343
OGBN_ARXIV_NUM_EDGES = 1_166_243
OGBN_ARXIV_NUM_CLASSES = 40
_ZIP_NAME = "arxiv.zip"
_CACHE_VERSION = 1


@dataclass
class CitationData:
    """Directed citation graph as plain arrays.

    Attributes:
        name: dataset identifier.
        num_nodes: number of papers.
        edges: ``(m, 2)`` int64 array; row ``(u, v)`` means paper ``u`` cites ``v``.
        labels: ``(n,)`` int64 primary-category index of each paper.
        category_names: ``category_names[k]`` is the readable name of label ``k``.
        years: ``(n,)`` publication year, or ``None``.
        paper_ids: ``(n,)`` original (MAG) paper id, or ``None``.
        source: provenance metadata (URL, files, notes).
    """

    name: str
    num_nodes: int
    edges: np.ndarray
    labels: np.ndarray
    category_names: list[str]
    years: np.ndarray | None = None
    paper_ids: np.ndarray | None = None
    source: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        if self.labels.shape != (self.num_nodes,):
            raise ValueError(
                f"labels must have shape ({self.num_nodes},), got {self.labels.shape}"
            )
        if self.edges.size and (self.edges.min() < 0 or self.edges.max() >= self.num_nodes):
            raise ValueError("edge endpoints must lie in [0, num_nodes)")
        if self.labels.size and (
            self.labels.min() < 0 or self.labels.max() >= len(self.category_names)
        ):
            raise ValueError("labels must index into category_names")

    @property
    def num_edges(self) -> int:
        """Number of directed edges (citations) as stored."""
        return int(self.edges.shape[0])


# --------------------------------------------------------------------------- ogbn-arxiv


def normalize_category_name(raw: str) -> str:
    """Convert OGB's ``'arxiv cs ai'`` into the arXiv style ``'cs.AI'``."""
    tokens = str(raw).strip().split()
    if len(tokens) >= 2 and tokens[0].lower() == "arxiv":
        tokens = tokens[1:]
    if len(tokens) >= 2:
        return f"{tokens[0].lower()}.{tokens[1].upper()}"
    return str(raw).strip()


def download_ogbn_arxiv(raw_dir: Path, url: str, force: bool = False) -> Path:
    """Download the official ogbn-arxiv zip into ``raw_dir`` (cached).

    Returns:
        Path of the local zip file.
    """
    logger = get_logger()
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / _ZIP_NAME
    if target.exists() and not force and zipfile.is_zipfile(target):
        logger.info("Using cached raw archive %s (%.1f MB)", target, target.stat().st_size / 1e6)
        return target
    tmp = target.with_suffix(".zip.part")
    logger.info("Downloading %s -> %s", url, target)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download ogbn-arxiv from {url}: {exc}. If this machine has no "
            f"internet access (e.g. a cluster compute node), run `mo438-f1 download` on a "
            f"node with access, or copy {_ZIP_NAME} manually into {raw_dir}."
        ) from exc
    if not zipfile.is_zipfile(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file from {url} is not a valid zip archive")
    tmp.replace(target)
    logger.info("Downloaded %.1f MB", target.stat().st_size / 1e6)
    return target


def _find_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if not matches:
        raise RuntimeError(f"File ending with {suffix!r} not found in {archive.filename}")
    return min(matches, key=len)


def _read_csv_member(archive: zipfile.ZipFile, suffix: str, **kwargs: Any) -> pd.DataFrame:
    with archive.open(_find_member(archive, suffix)) as handle:
        return pd.read_csv(handle, compression="gzip", **kwargs)


def parse_ogbn_arxiv_zip(zip_path: Path) -> CitationData:
    """Parse the raw OGB archive into :class:`CitationData` and sanity-check it."""
    logger = get_logger()
    with zipfile.ZipFile(zip_path) as archive:
        edges = _read_csv_member(archive, "raw/edge.csv.gz", header=None, dtype=np.int64)
        labels = _read_csv_member(archive, "raw/node-label.csv.gz", header=None)
        years = _read_csv_member(archive, "raw/node_year.csv.gz", header=None)
        num_nodes = int(
            _read_csv_member(archive, "raw/num-node-list.csv.gz", header=None).iloc[0, 0]
        )
        num_edges = int(
            _read_csv_member(archive, "raw/num-edge-list.csv.gz", header=None).iloc[0, 0]
        )
        mapping = _read_csv_member(archive, "mapping/labelidx2arxivcategeory.csv.gz")
        paper_map = _read_csv_member(archive, "mapping/nodeidx2paperid.csv.gz")

    edge_array = edges.to_numpy(dtype=np.int64)
    if edge_array.shape != (num_edges, 2):
        raise RuntimeError(f"edge.csv has shape {edge_array.shape}, expected ({num_edges}, 2)")
    label_array = labels.iloc[:, 0].to_numpy(dtype=np.int64)
    if label_array.shape[0] != num_nodes:
        raise RuntimeError("node-label.csv length does not match num-node-list.csv")

    mapping = mapping.sort_values(mapping.columns[0])
    if not np.array_equal(mapping.iloc[:, 0].to_numpy(), np.arange(len(mapping))):
        raise RuntimeError("label mapping is not a contiguous 0..K-1 index")
    category_names = [normalize_category_name(v) for v in mapping.iloc[:, 1]]

    paper_map = paper_map.sort_values(paper_map.columns[0])
    paper_ids = paper_map.iloc[:, 1].to_numpy(dtype=np.int64)
    if paper_ids.shape[0] != num_nodes:
        raise RuntimeError("nodeidx2paperid.csv length does not match num-node-list.csv")

    for name, value, expected in (
        ("nodes", num_nodes, OGBN_ARXIV_NUM_NODES),
        ("edges", num_edges, OGBN_ARXIV_NUM_EDGES),
        ("classes", len(category_names), OGBN_ARXIV_NUM_CLASSES),
    ):
        if value != expected:
            logger.warning("ogbn-arxiv has %d %s; documentation says %d", value, name, expected)

    return CitationData(
        name="ogbn-arxiv",
        num_nodes=num_nodes,
        edges=edge_array,
        labels=label_array,
        category_names=category_names,
        years=years.iloc[:, 0].to_numpy(dtype=np.int64),
        paper_ids=paper_ids,
        source={
            "dataset": "ogbn-arxiv",
            "homepage": "https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv",
            "archive": str(zip_path.name),
            "edge_orientation": "row (u, v) of raw/edge.csv.gz: paper u cites paper v",
        },
    )


# --------------------------------------------------------------------------- cache


def save_citation_data(data: CitationData, path: Path) -> None:
    """Cache :class:`CitationData` as a compressed ``.npz`` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "edges": data.edges,
        "labels": data.labels,
        "meta": np.array(
            json.dumps(
                {
                    "cache_version": _CACHE_VERSION,
                    "name": data.name,
                    "num_nodes": data.num_nodes,
                    "category_names": data.category_names,
                    "source": data.source,
                }
            )
        ),
    }
    if data.years is not None:
        arrays["years"] = data.years
    if data.paper_ids is not None:
        arrays["paper_ids"] = data.paper_ids
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_citation_data(path: Path) -> CitationData | None:
    """Load a cache written by :func:`save_citation_data` (``None`` if stale/invalid)."""
    try:
        with np.load(path, allow_pickle=False) as npz:
            meta = json.loads(str(npz["meta"]))
            if meta.get("cache_version") != _CACHE_VERSION:
                return None
            return CitationData(
                name=meta["name"],
                num_nodes=int(meta["num_nodes"]),
                edges=npz["edges"],
                labels=npz["labels"],
                category_names=list(meta["category_names"]),
                years=npz["years"] if "years" in npz.files else None,
                paper_ids=npz["paper_ids"] if "paper_ids" in npz.files else None,
                source=meta.get("source", {}),
            )
    except (OSError, KeyError, ValueError):
        return None


def load_ogbn_arxiv(
    raw_dir: Path,
    processed_dir: Path,
    url: str,
    force_download: bool = False,
    force_rebuild: bool = False,
) -> CitationData:
    """Load ogbn-arxiv, using the processed cache, the raw zip cache or the network."""
    logger = get_logger()
    cache = processed_dir / "ogbn_arxiv_citations.npz"
    if not (force_download or force_rebuild) and cache.exists():
        data = load_citation_data(cache)
        if data is not None:
            logger.info("Loaded processed cache %s", cache)
            return data
        logger.warning("Ignoring invalid/stale cache %s", cache)
    zip_path = download_ogbn_arxiv(raw_dir, url, force=force_download)
    logger.info("Parsing %s", zip_path)
    data = parse_ogbn_arxiv_zip(zip_path)
    save_citation_data(data, cache)
    logger.info("Wrote processed cache %s", cache)
    return data


# --------------------------------------------------------------------------- synthetic


def generate_synthetic_citations(
    num_nodes: int = 3000,
    num_categories: int = 8,
    mean_out_degree: float = 6.0,
    intra_category_prob: float = 0.7,
    preferential_prob: float = 0.4,
    closure_prob: float = 0.3,
    num_small_components: int = 12,
    num_isolated: int = 5,
    num_self_loops: int = 4,
    num_reciprocal: int = 15,
    num_duplicates: int = 6,
    seed: int = 0,
) -> CitationData:
    """Generate a small, reproducible citation-like directed graph for smoke tests.

    Papers arrive in index order and cite earlier papers, preferring the same
    category, popular papers (copying a random existing citation) and papers cited by
    their own references (triadic closure). A few small detached components, isolated
    nodes, self-loops, reciprocal and duplicated citations are added on purpose so
    that every branch of the G_d -> G_u conversion is exercised.
    """
    rng = np.random.default_rng(seed)
    num_main = num_nodes - num_isolated - 3 * num_small_components
    if num_main < 10:
        raise ValueError("num_nodes too small for the requested small components/isolated")
    weights = rng.dirichlet(np.full(num_categories, 2.0))
    labels = rng.choice(num_categories, size=num_nodes, p=weights)

    by_category: list[list[int]] = [[] for _ in range(num_categories)]
    out_lists: list[list[int]] = [[] for _ in range(num_nodes)]
    edges: list[tuple[int, int]] = []
    by_category[labels[0]].append(0)
    for node in range(1, num_main):
        k = min(node, max(1, int(rng.poisson(mean_out_degree))))
        targets: set[int] = set()
        attempts = 0
        while len(targets) < k and attempts < 20 * k:
            attempts += 1
            r = rng.random()
            if targets and r < closure_prob:
                ref = list(targets)[rng.integers(len(targets))]
                if not out_lists[ref]:
                    continue
                candidate = out_lists[ref][rng.integers(len(out_lists[ref]))]
            elif edges and r < closure_prob + preferential_prob:
                candidate = edges[rng.integers(len(edges))][1]
            elif by_category[labels[node]] and rng.random() < intra_category_prob:
                pool = by_category[labels[node]]
                candidate = pool[rng.integers(len(pool))]
            else:
                candidate = int(rng.integers(node))
            targets.add(int(candidate))
        for target in sorted(targets):
            edges.append((node, target))
            out_lists[node].append(target)
        by_category[labels[node]].append(node)

    # Detached small components (paths of 2-3 nodes) -> component-size distribution.
    next_node = num_main
    for i in range(num_small_components):
        size = 2 + (i % 2)
        nodes = list(range(next_node, next_node + size))
        edges.extend((nodes[j + 1], nodes[j]) for j in range(size - 1))
        next_node += 3
    # Remaining nodes stay isolated (including the unused slots of the small components).

    main_edges = np.asarray(edges, dtype=np.int64)
    extra: list[tuple[int, int]] = []
    for node in rng.choice(num_main, size=num_self_loops, replace=False):
        extra.append((int(node), int(node)))
    for idx in rng.choice(len(main_edges), size=num_reciprocal, replace=False):
        u, v = main_edges[idx]
        extra.append((int(v), int(u)))
    for idx in rng.choice(len(main_edges), size=num_duplicates, replace=False):
        u, v = main_edges[idx]
        extra.append((int(u), int(v)))
    all_edges = np.vstack([main_edges, np.asarray(extra, dtype=np.int64).reshape(-1, 2)])

    return CitationData(
        name="synthetic",
        num_nodes=num_nodes,
        edges=all_edges,
        labels=labels,
        category_names=[f"syn.C{k:02d}" for k in range(num_categories)],
        years=2000 + (np.arange(num_nodes) * 20) // num_nodes,
        paper_ids=np.arange(num_nodes, dtype=np.int64),
        source={"dataset": "synthetic", "seed": seed, "generator": "generate_synthetic_citations"},
    )


def load_dataset(config: dict[str, Any]) -> CitationData:
    """Load the dataset selected by ``config['data']['dataset']``."""
    data_cfg = config["data"]
    runtime = config["runtime"]
    if data_cfg["dataset"] == "synthetic":
        params = dict(data_cfg["synthetic"])
        return generate_synthetic_citations(**params, seed=int(config["project"]["seed"]))
    return load_ogbn_arxiv(
        raw_dir=Path(data_cfg["raw_dir"]),
        processed_dir=Path(data_cfg["processed_dir"]),
        url=data_cfg["source_url"],
        force_download=bool(runtime["force_download"]),
        force_rebuild=bool(runtime["force_rebuild_graphs"]),
    )
