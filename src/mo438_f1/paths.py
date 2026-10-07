"""Average shortest-path length and diameter: exact engine and documented approximation.

Engine (exact)
--------------
Bit-parallel multi-source BFS ("MS-BFS"): a batch of ``64 * W`` sources is explored
simultaneously. Every node keeps ``W`` 64-bit words; bit ``j`` of ``seen[v]`` says
whether source ``j`` has already reached ``v``. One BFS level for the whole batch is

    next[v] = OR_{u in N(v)} frontier[u]  AND NOT seen[v]

which is a gather over the CSR ``indices`` followed by ``np.bitwise_or.reduceat``
(vectorised C loops; no Python per-node work). ``popcount(next)`` gives the number of
(source, target) pairs at distance ``d``. Running all nodes of a connected graph as
sources yields the *exact* distance histogram over all ordered pairs, hence the exact
mean distance and the exact diameter (largest ``d`` with a non-empty level). Batches are
independent and are distributed over worker processes.

Cost: ``O(n / (64 W) * L * nnz * W) = O(n * L * m / 32)`` word operations, where ``L``
is the number of BFS levels (≈ diameter). For ogbn-arxiv this is a few CPU-minutes to
tens of CPU-minutes, i.e. feasible, so ``paths.mode: auto`` normally selects the exact
computation; the decision is based on a timed probe batch and recorded in the outputs.

Approximation (only if selected)
--------------------------------
``k`` sources are sampled uniformly without replacement (seeded) from the component.
The mean distance is estimated by the mean of the per-source mean distances (unbiased
for the all-pairs mean in a connected graph); its standard error uses the finite
population correction. The diameter is reported as an interval: the lower bound is the
largest eccentricity observed among the sampled sources and a few *double-sweep*
BFS runs; the upper bound is ``2 * min ecc(v)`` over the BFS roots (``diam <= 2 ecc(v)``
for every ``v``). Accuracy is validated against exact values on connected subgraphs
(:func:`validate_approximation`).
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .graph_build import induced_subgraph_csr
from .runtime import get_logger

WORD_BITS = 64
# Assumed parallel efficiency when extrapolating the exact cost (memory-bound kernels
# rarely scale linearly); conservative on purpose.
PARALLEL_EFFICIENCY = 0.5

_POPCOUNT_TABLE = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def _popcount_total(words: np.ndarray) -> int:
    """Total number of set bits in a uint64 array."""
    if hasattr(np, "bitwise_count"):
        return int(np.bitwise_count(words).sum(dtype=np.int64))
    return int(_POPCOUNT_TABLE[words.view(np.uint8)].sum(dtype=np.int64))


def _popcount_per_bit(words: np.ndarray) -> np.ndarray:
    """Number of rows with bit ``b`` set, for every bit of a ``(n, W)`` uint64 array.

    Returns an int64 array of length ``64 * W``; entry ``64 * w + b`` refers to bit ``b``
    of word ``w`` (i.e. source ``64 * w + b`` of the batch).
    """
    n, width = words.shape
    as_bytes = np.ascontiguousarray(words, dtype="<u8").view(np.uint8).reshape(n, width * 8)
    bits = np.unpackbits(as_bytes, axis=1, bitorder="little")
    return bits.sum(axis=0, dtype=np.int64)


@dataclass
class BFSBatchResult:
    """Result of one multi-source BFS batch.

    Attributes:
        histogram: ``histogram[d]`` = number of (source, target) pairs at distance ``d``
            (``histogram[0] = 0``: the source itself is not counted).
        source_distance_sum: per-source sum of distances (only if requested).
        source_reached: per-source number of reached targets (only if requested).
        source_eccentricity: per-source eccentricity (only if requested).
    """

    histogram: np.ndarray
    source_distance_sum: np.ndarray | None = None
    source_reached: np.ndarray | None = None
    source_eccentricity: np.ndarray | None = None

    @property
    def max_distance(self) -> int:
        """Largest distance reached from any source of the batch."""
        return int(self.histogram.size - 1)


def msbfs(
    indptr: np.ndarray, indices: np.ndarray, sources: np.ndarray, per_source: bool = False
) -> BFSBatchResult:
    """Run a bit-parallel BFS from all ``sources`` at once on a CSR graph.

    Args:
        indptr, indices: CSR structure of a symmetric (undirected) adjacency.
        sources: node ids (any number; processed with ``ceil(len/64)`` words).
        per_source: also return per-source distance sums, reach counts and
            eccentricities (costs one extra bit-unpacking per level).
    """
    n = indptr.size - 1
    sources = np.asarray(sources, dtype=np.int64)
    k = sources.size
    if k == 0:
        return BFSBatchResult(histogram=np.zeros(1, dtype=np.int64))
    width = (k + WORD_BITS - 1) // WORD_BITS
    position = np.arange(k)
    frontier = np.zeros((n, width), dtype=np.uint64)
    np.bitwise_or.at(
        frontier,
        (sources, position // WORD_BITS),
        np.left_shift(np.uint64(1), (position % WORD_BITS).astype(np.uint64)),
    )
    seen = frontier.copy()
    nonempty = np.diff(indptr) > 0
    starts = indptr[:-1][nonempty]

    histogram = [0]
    if per_source:
        dist_sum = np.zeros(k, dtype=np.int64)
        reached = np.zeros(k, dtype=np.int64)
        ecc = np.zeros(k, dtype=np.int64)
    level = 0
    while starts.size:
        gathered = frontier[indices]
        nxt = np.zeros_like(frontier)
        nxt[nonempty] = np.bitwise_or.reduceat(gathered, starts, axis=0)
        del gathered
        nxt &= ~seen
        count = _popcount_total(nxt)
        if count == 0:
            break
        level += 1
        histogram.append(count)
        if per_source:
            counts = _popcount_per_bit(nxt)[:k]
            dist_sum += level * counts
            reached += counts
            ecc[counts > 0] = level
        seen |= nxt
        frontier = nxt
    result = BFSBatchResult(histogram=np.asarray(histogram, dtype=np.int64))
    if per_source:
        result.source_distance_sum = dist_sum
        result.source_reached = reached
        result.source_eccentricity = ecc
    return result


def bfs_distances(indptr: np.ndarray, indices: np.ndarray, source: int) -> np.ndarray:
    """Single-source BFS distances (``-1`` for unreachable nodes); vectorised by level."""
    n = indptr.size - 1
    dist = np.full(n, -1, dtype=np.int64)
    dist[source] = 0
    frontier = np.array([source], dtype=np.int64)
    level = 0
    while frontier.size:
        level += 1
        begin = indptr[frontier]
        lengths = indptr[frontier + 1] - begin
        total = int(lengths.sum())
        if total == 0:
            break
        offsets = np.repeat(begin - np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths)
        neighbours = indices[offsets + np.arange(total)]
        neighbours = np.unique(neighbours[dist[neighbours] < 0])
        dist[neighbours] = level
        frontier = neighbours
    return dist


def bfs_ball(indptr: np.ndarray, indices: np.ndarray, start: int, size: int) -> np.ndarray:
    """First ``size`` nodes in BFS order from ``start`` (level by level, ids ascending).

    The induced subgraph of the result is connected.
    """
    dist = bfs_distances(indptr, indices, start)
    reachable = np.flatnonzero(dist >= 0)
    order = reachable[np.lexsort((reachable, dist[reachable]))]
    return np.sort(order[:size])


# --------------------------------------------------------------------------- parallel


_WORKER_GRAPH: tuple[np.ndarray, np.ndarray] | None = None


def _init_worker(indptr: np.ndarray, indices: np.ndarray) -> None:
    global _WORKER_GRAPH
    _WORKER_GRAPH = (indptr, indices)


def _run_worker_batch(task: tuple[np.ndarray, bool]) -> BFSBatchResult:
    assert _WORKER_GRAPH is not None
    sources, per_source = task
    return msbfs(_WORKER_GRAPH[0], _WORKER_GRAPH[1], sources, per_source)


def run_batches(
    indptr: np.ndarray,
    indices: np.ndarray,
    sources: np.ndarray,
    batch_words: int,
    n_jobs: int,
    per_source: bool = False,
    label: str = "MS-BFS",
) -> list[BFSBatchResult]:
    """Split ``sources`` into batches of ``64 * batch_words`` and run them (in parallel).

    Results are returned in batch order.
    """
    logger = get_logger()
    batch_size = WORD_BITS * batch_words
    batches = [sources[i:i + batch_size] for i in range(0, sources.size, batch_size)]
    if not batches:
        return []
    workers = max(1, min(n_jobs, len(batches)))
    start = time.perf_counter()
    report_every = max(1, math.ceil(len(batches) / 10))

    def _progress(done: int) -> None:
        if len(batches) >= 10 and (done % report_every == 0 or done == len(batches)):
            elapsed = time.perf_counter() - start
            eta = elapsed / done * (len(batches) - done)
            logger.info("%s: %d/%d batches (%.0f%%), elapsed %.1f s, ETA %.1f s",
                        label, done, len(batches), 100 * done / len(batches), elapsed, eta)

    results: list[BFSBatchResult] = []
    if workers == 1:
        for i, batch in enumerate(batches, 1):
            results.append(msbfs(indptr, indices, batch, per_source))
            _progress(i)
        return results
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(indptr, indices)
    ) as pool:
        tasks = [(batch, per_source) for batch in batches]
        for i, result in enumerate(pool.map(_run_worker_batch, tasks), 1):
            results.append(result)
            _progress(i)
    return results


def _merge_histograms(results: list[BFSBatchResult]) -> np.ndarray:
    length = max((r.histogram.size for r in results), default=1)
    total = np.zeros(length, dtype=np.int64)
    for r in results:
        total[: r.histogram.size] += r.histogram
    return total


# --------------------------------------------------------------------------- results


@dataclass
class DistanceResult:
    """Distance statistics of a connected graph (exact or approximate)."""

    is_exact: bool
    method: str
    num_nodes: int
    num_edges: int
    num_sources: int
    average_distance: float
    diameter: int | None
    diameter_lower_bound: int
    diameter_upper_bound: int
    histogram: np.ndarray
    average_distance_stderr: float | None = None
    average_distance_ci95: tuple[float, float] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "is_exact": self.is_exact,
            "method": self.method,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_sources": self.num_sources,
            "average_distance": self.average_distance,
            "average_distance_stderr": self.average_distance_stderr,
            "average_distance_ci95": list(self.average_distance_ci95)
            if self.average_distance_ci95 else None,
            "diameter": self.diameter,
            "diameter_lower_bound": self.diameter_lower_bound,
            "diameter_upper_bound": self.diameter_upper_bound,
            "distance_histogram_ordered_pairs": self.histogram.tolist(),
            "params": self.params,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DistanceResult:
        """Inverse of :meth:`to_dict` (used by the on-disk cache)."""
        ci = payload.get("average_distance_ci95")
        return cls(
            is_exact=payload["is_exact"],
            method=payload["method"],
            num_nodes=payload["num_nodes"],
            num_edges=payload["num_edges"],
            num_sources=payload["num_sources"],
            average_distance=payload["average_distance"],
            diameter=payload["diameter"],
            diameter_lower_bound=payload["diameter_lower_bound"],
            diameter_upper_bound=payload["diameter_upper_bound"],
            histogram=np.asarray(payload["distance_histogram_ordered_pairs"], dtype=np.int64),
            average_distance_stderr=payload.get("average_distance_stderr"),
            average_distance_ci95=tuple(ci) if ci else None,
            params=payload.get("params", {}),
            elapsed_seconds=payload.get("elapsed_seconds", 0.0),
        )


def _csr_arrays(adjacency: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    return adjacency.indptr.astype(np.int64), adjacency.indices.astype(np.int64)


def exact_distance_stats(
    adjacency: sp.csr_matrix, batch_words: int = 1, n_jobs: int = 1
) -> DistanceResult:
    """Exact mean distance and diameter of a *connected* undirected graph (all sources).

    Raises:
        ValueError: if the graph is not connected.
    """
    start = time.perf_counter()
    indptr, indices = _csr_arrays(adjacency)
    n = indptr.size - 1
    results = run_batches(indptr, indices, np.arange(n, dtype=np.int64), batch_words, n_jobs,
                          label="exact all-pairs MS-BFS")
    histogram = _merge_histograms(results)
    pairs = int(histogram.sum())
    if pairs != n * (n - 1):
        raise ValueError(
            f"graph is not connected: {pairs} reachable ordered pairs, expected {n * (n - 1)}"
        )
    distance_sum = int((np.arange(histogram.size) * histogram).sum())
    diameter = int(histogram.size - 1)
    return DistanceResult(
        is_exact=True,
        method="exact all-pairs BFS (bit-parallel multi-source BFS, all nodes as sources)",
        num_nodes=n,
        num_edges=int(indices.size // 2),
        num_sources=n,
        average_distance=distance_sum / pairs if pairs else 0.0,
        diameter=diameter,
        diameter_lower_bound=diameter,
        diameter_upper_bound=diameter,
        histogram=histogram,
        params={"batch_words": batch_words, "n_jobs": n_jobs},
        elapsed_seconds=time.perf_counter() - start,
    )


def sampled_distance_stats(
    adjacency: sp.csr_matrix,
    num_sources: int,
    seed: int,
    batch_words: int = 1,
    n_jobs: int = 1,
    diameter_sweeps: int = 4,
) -> DistanceResult:
    """Approximate mean distance and diameter bounds from ``num_sources`` random sources.

    See the module docstring for the estimator and the bounds.
    """
    start = time.perf_counter()
    indptr, indices = _csr_arrays(adjacency)
    n = indptr.size - 1
    k = min(int(num_sources), n)
    rng = np.random.default_rng(seed)
    sources = np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)
    results = run_batches(indptr, indices, sources, batch_words, n_jobs, per_source=True,
                          label="sampled MS-BFS")
    histogram = _merge_histograms(results)
    dist_sum = np.concatenate([r.source_distance_sum for r in results])
    reached = np.concatenate([r.source_reached for r in results])
    ecc = np.concatenate([r.source_eccentricity for r in results])
    if np.any(reached != n - 1):
        raise ValueError("graph is not connected: some sampled sources do not reach all nodes")

    per_source_mean = dist_sum / reached
    estimate = float(per_source_mean.mean())
    if k > 1:
        fpc = math.sqrt((n - k) / (n - 1)) if n > 1 else 0.0
        stderr = float(per_source_mean.std(ddof=1) / math.sqrt(k) * fpc)
    else:
        stderr = float("nan")
    ci = (estimate - 1.96 * stderr, estimate + 1.96 * stderr)

    lower = int(ecc.max())
    upper = int(2 * ecc.min())
    current = int(sources[int(np.argmax(ecc))])
    sweep_log = []
    for _ in range(max(0, diameter_sweeps)):
        dist = bfs_distances(indptr, indices, current)
        node_ecc = int(dist.max())
        lower = max(lower, node_ecc)
        upper = min(upper, 2 * node_ecc)
        farthest = int(np.flatnonzero(dist == node_ecc)[0])
        sweep_log.append({"root": current, "eccentricity": node_ecc, "farthest": farthest})
        current = farthest

    return DistanceResult(
        is_exact=False,
        method=(
            "approximate: uniform sample of BFS sources (mean of per-source mean distances); "
            "diameter bounds from sampled eccentricities + double sweep"
        ),
        num_nodes=n,
        num_edges=int(indices.size // 2),
        num_sources=k,
        average_distance=estimate,
        diameter=lower if lower == upper else None,
        diameter_lower_bound=lower,
        diameter_upper_bound=upper,
        histogram=histogram,
        average_distance_stderr=stderr,
        average_distance_ci95=ci,
        params={
            "num_sources": k,
            "seed": int(seed),
            "sampling": "uniform without replacement (numpy default_rng)",
            "diameter_sweeps": int(diameter_sweeps),
            "sweeps": sweep_log,
            "batch_words": batch_words,
            "n_jobs": n_jobs,
        },
        elapsed_seconds=time.perf_counter() - start,
    )


def estimate_exact_seconds(
    adjacency: sp.csr_matrix, batch_words: int, n_jobs: int, seed: int
) -> dict[str, Any]:
    """Time one probe batch and extrapolate the wall-clock cost of the exact computation."""
    indptr, indices = _csr_arrays(adjacency)
    n = indptr.size - 1
    batch_size = WORD_BITS * batch_words
    probe = np.sort(np.random.default_rng(seed).choice(n, size=min(n, batch_size), replace=False))
    start = time.perf_counter()
    msbfs(indptr, indices, probe)
    probe_seconds = time.perf_counter() - start
    num_batches = math.ceil(n / batch_size)
    workers = max(1, min(n_jobs, num_batches))
    speedup = 1.0 if workers == 1 else max(1.0, workers * PARALLEL_EFFICIENCY)
    return {
        "probe_batch_sources": int(probe.size),
        "probe_seconds": probe_seconds,
        "num_batches": num_batches,
        "n_jobs": workers,
        "assumed_parallel_efficiency": PARALLEL_EFFICIENCY if workers > 1 else 1.0,
        "estimated_seconds": probe_seconds * num_batches / speedup,
        "estimated_cpu_seconds": probe_seconds * num_batches,
    }


def compute_distance_stats(
    adjacency: sp.csr_matrix,
    mode: str,
    num_sources: int,
    seed: int,
    batch_words: int,
    n_jobs: int,
    max_exact_seconds: float,
    diameter_sweeps: int,
) -> tuple[DistanceResult, dict[str, Any]]:
    """Compute distance statistics following ``mode`` (``exact|approximate|auto``).

    Returns:
        ``(result, decision)`` where ``decision`` documents why the method was chosen.
    """
    logger = get_logger()
    decision: dict[str, Any] = {"requested_mode": mode}
    if mode == "auto":
        estimate = estimate_exact_seconds(adjacency, batch_words, n_jobs, seed)
        decision["exact_cost_estimate"] = estimate
        decision["max_exact_seconds"] = max_exact_seconds
        chosen = "exact" if estimate["estimated_seconds"] <= max_exact_seconds else "approximate"
        logger.info(
            "Exact path computation estimated at %.0f s (%.0f CPU-s) with %d worker(s); "
            "limit %.0f s -> %s",
            estimate["estimated_seconds"], estimate["estimated_cpu_seconds"],
            estimate["n_jobs"], max_exact_seconds, chosen,
        )
    else:
        chosen = mode
    decision["chosen_mode"] = chosen
    if chosen == "exact":
        result = exact_distance_stats(adjacency, batch_words, n_jobs)
    else:
        logger.warning("Distances will be APPROXIMATE (%d sampled sources, seed %d)",
                       num_sources, seed)
        result = sampled_distance_stats(adjacency, num_sources, seed, batch_words, n_jobs,
                                        diameter_sweeps)
    return result, decision


# --------------------------------------------------------------------------- validation


def _igraph_reference(adjacency: sp.csr_matrix) -> tuple[float, int]:
    """Mean distance and diameter computed independently with igraph."""
    upper = sp.triu(adjacency, k=1).tocoo()
    graph = ig.Graph(n=adjacency.shape[0], edges=list(zip(upper.row.tolist(), upper.col.tolist())))
    hist = graph.path_length_hist(directed=False)
    bins = list(hist.bins())
    total = sum(count for _, _, count in bins)
    mean = sum(start * count for start, _, count in bins) / total if total else 0.0
    diameter = int(max((start for start, _, count in bins if count), default=0))
    return mean, diameter


def validation_row(
    label: str, adjacency: sp.csr_matrix, exact: DistanceResult, approx: DistanceResult,
    repeat: int, igraph_mean: float | None = None, igraph_diameter: int | None = None,
) -> dict[str, Any]:
    """One row of the validation table comparing an exact and an approximate result."""
    rel_error = (approx.average_distance - exact.average_distance) / exact.average_distance
    low, high = approx.average_distance_ci95 or (float("nan"), float("nan"))
    return {
        "instance": label,
        "num_nodes": exact.num_nodes,
        "num_edges": exact.num_edges,
        "num_sources": approx.num_sources,
        "repeat": repeat,
        "seed": approx.params.get("seed"),
        "exact_average_distance": exact.average_distance,
        "igraph_average_distance": igraph_mean,
        "approx_average_distance": approx.average_distance,
        "approx_stderr": approx.average_distance_stderr,
        "relative_error": rel_error,
        "abs_relative_error": abs(rel_error),
        "exact_within_ci95": bool(low <= exact.average_distance <= high),
        "exact_diameter": exact.diameter,
        "igraph_diameter": igraph_diameter,
        "approx_diameter_lower": approx.diameter_lower_bound,
        "approx_diameter_upper": approx.diameter_upper_bound,
        "diameter_lower_equals_exact": approx.diameter_lower_bound == exact.diameter,
        "diameter_bounds_hold": approx.diameter_lower_bound <= (exact.diameter or 0)
        <= approx.diameter_upper_bound,
    }


def validate_approximation(
    adjacency: sp.csr_matrix,
    sizes: list[int],
    num_sources_list: list[int],
    repeats: int,
    seed: int,
    batch_words: int = 1,
    n_jobs: int = 1,
    diameter_sweeps: int = 4,
) -> pd.DataFrame:
    """Compare exact vs. sampled distance statistics on connected subgraphs.

    For each size ``s`` a connected subgraph is obtained as the BFS ball of ``s`` nodes
    around a random root (seeded). Exact values come from the MS-BFS engine and are
    cross-checked with igraph; the approximation is repeated ``repeats`` times for each
    number of sources smaller than ``s``.
    """
    logger = get_logger()
    indptr, indices = _csr_arrays(adjacency)
    n = indptr.size - 1
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for size in sizes:
        if size >= n:
            logger.info("Validation size %d skipped: component has only %d nodes", size, n)
            continue
        root = int(rng.integers(n))
        nodes = bfs_ball(indptr, indices, root, int(size))
        sub = induced_subgraph_csr(adjacency, nodes)
        exact = exact_distance_stats(sub, batch_words, n_jobs=1)
        ig_mean, ig_diam = _igraph_reference(sub)
        logger.info("Validation subgraph |V|=%d |E|=%d: exact <d>=%.4f diam=%d (igraph %.4f, %d)",
                    exact.num_nodes, exact.num_edges, exact.average_distance, exact.diameter,
                    ig_mean, ig_diam)
        for k in num_sources_list:
            if k >= exact.num_nodes:
                continue
            for rep in range(repeats):
                approx = sampled_distance_stats(
                    sub, k, seed=seed + 7919 * rep + k, batch_words=batch_words,
                    n_jobs=1, diameter_sweeps=diameter_sweeps,
                )
                rows.append(validation_row(f"bfs_ball_{size}", sub, exact, approx, rep,
                                           ig_mean, ig_diam))
    return pd.DataFrame(rows)


def summarize_validation(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the validation table per (instance, num_sources)."""
    if table.empty:
        return table
    grouped = table.groupby(["instance", "num_nodes", "num_sources"], sort=False)
    return grouped.agg(
        repeats=("repeat", "count"),
        exact_average_distance=("exact_average_distance", "first"),
        mean_approx_average_distance=("approx_average_distance", "mean"),
        mean_abs_relative_error=("abs_relative_error", "mean"),
        max_abs_relative_error=("abs_relative_error", "max"),
        ci95_coverage=("exact_within_ci95", "mean"),
        exact_diameter=("exact_diameter", "first"),
        min_diameter_lower=("approx_diameter_lower", "min"),
        max_diameter_lower=("approx_diameter_lower", "max"),
        fraction_lower_equals_exact=("diameter_lower_equals_exact", "mean"),
        bounds_always_hold=("diameter_bounds_hold", "all"),
    ).reset_index()
