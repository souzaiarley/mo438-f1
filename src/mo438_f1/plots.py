"""Matplotlib figures for F1 (static PNGs for the report).

Style: one hue per series (blue first, orange second), thin marks, hairline grid,
titles and axis labels that state the unit/count. Category-coloured figures show
the 7 largest categories and fold the rest into a grey "other" class.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#b5b4ad"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
SEQUENTIAL = "Blues"

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "font.size": 9.5,
}


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _log_bins(values: np.ndarray, bins: int) -> np.ndarray:
    positive = values[values > 0]
    hi = max(2.0, float(positive.max()) + 1) if positive.size else 2.0
    return np.unique(np.floor(np.logspace(0, np.log10(hi), bins)).astype(int))


def _ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values[values > 0])
    uniq, idx = np.unique(values, return_index=True)
    return uniq, 1.0 - idx / values.size


def plot_degree_distribution(
    degrees: np.ndarray, path: Path, dpi: int, title: str, log_x: bool = True, log_y: bool = True
) -> Path:
    """P(k) (counts per degree) and complementary CDF of the degree of G_u."""
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        uniq, counts = np.unique(degrees[degrees > 0], return_counts=True)
        ax1.scatter(uniq, counts, s=9, color=SERIES[0], linewidths=0, alpha=0.85)
        ax1.set_xlabel("degree k")
        ax1.set_ylabel("number of nodes with degree k")
        ax1.set_title("Degree distribution")
        if log_x:
            ax1.set_xscale("log")
        if log_y:
            ax1.set_yscale("log")
        isolated = int(np.count_nonzero(degrees == 0))
        ax1.text(0.98, 0.97,
                 f"n = {degrees.size:,}\n<k> = {degrees.mean():.2f}\nmax k = {degrees.max():,}"
                 + (f"\nk = 0 (not shown): {isolated:,}" if isolated else ""),
                 transform=ax1.transAxes, ha="right", va="top", color=INK_2, fontsize=8.5)
        x, y = _ccdf(degrees)
        ax2.step(x, y, where="post", color=SERIES[0], linewidth=1.6)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("degree k")
        ax2.set_ylabel("fraction of nodes with degree ≥ k")
        ax2.set_title("Complementary CDF")
        fig.suptitle(title, fontsize=12.5, fontweight="bold", color=INK)
        fig.tight_layout()
        return _save(fig, path, dpi)


def plot_in_out_degree(in_deg: np.ndarray, out_deg: np.ndarray, path: Path, dpi: int,
                       title: str) -> Path:
    """In-degree (citations received) and out-degree (references made) of G_d."""
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for values, color, label in ((in_deg, SERIES[0], "in-degree (citations received)"),
                                     (out_deg, SERIES[1], "out-degree (references made)")):
            uniq, counts = np.unique(values[values > 0], return_counts=True)
            ax1.scatter(uniq, counts, s=9, color=color, linewidths=0, alpha=0.8, label=label)
            x, y = _ccdf(values)
            ax2.step(x, y, where="post", color=color, linewidth=1.6, label=label)
        for ax in (ax1, ax2):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("degree k")
            ax.legend(loc="lower left", fontsize=8.5)
        ax1.set_ylabel("number of nodes")
        ax1.set_title("Degree distribution (k ≥ 1)")
        ax2.set_ylabel("fraction of nodes with degree ≥ k")
        ax2.set_title("Complementary CDF")
        fig.suptitle(title, fontsize=12.5, fontweight="bold", color=INK)
        fig.tight_layout()
        return _save(fig, path, dpi)


def plot_local_clustering(coeff: np.ndarray, degrees: np.ndarray, path: Path, dpi: int,
                          bins: int, average: float, transitivity: float, title: str) -> Path:
    """Histogram of C_i (all nodes) and mean C(k) as a function of the degree."""
    eligible = degrees >= 2
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.hist(coeff, bins=np.linspace(0, 1, min(bins, 50) + 1), color=SERIES[0],
                 edgecolor="white", linewidth=0.5)
        ax1.set_yscale("log")
        ax1.set_xlabel("local clustering coefficient $C_i$")
        ax1.set_ylabel("number of nodes")
        ax1.set_title("Distribution of $C_i$")
        ax1.axvline(average, color=INK, linewidth=1, linestyle="--")
        ax1.text(0.98, 0.97,
                 f"mean $C_i$ (k ≥ 2) = {average:.4f}\ntransitivity = {transitivity:.4f}\n"
                 f"nodes with k < 2 (C = 0, included): {int((~eligible).sum()):,}",
                 transform=ax1.transAxes, ha="right", va="top", color=INK_2, fontsize=8.5)
        ks = degrees[eligible]
        uniq, inverse = np.unique(ks, return_inverse=True)
        mean_c = np.bincount(inverse, weights=coeff[eligible]) / np.bincount(inverse)
        positive = mean_c > 0
        ax2.scatter(uniq[positive], mean_c[positive], s=9, color=SERIES[0], linewidths=0)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("degree k")
        ax2.set_ylabel("mean $C_i$ of nodes with degree k")
        ax2.set_title("Clustering vs. degree, C(k)")
        fig.suptitle(title, fontsize=12.5, fontweight="bold", color=INK)
        fig.tight_layout()
        return _save(fig, path, dpi)


def plot_component_sizes(sizes: np.ndarray, path: Path, dpi: int, title: str) -> Path:
    """Number of connected components of each size (log-log)."""
    uniq, counts = np.unique(sizes, return_counts=True)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.scatter(uniq, counts, s=22, color=SERIES[0], linewidths=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("component size (number of nodes)")
        ax.set_ylabel("number of components")
        lcc = int(sizes.max())
        ax.annotate(f"largest component\n{lcc:,} nodes ({lcc / sizes.sum():.2%})",
                    xy=(lcc, 1), xytext=(0.62, 0.55), textcoords="axes fraction",
                    color=INK_2, fontsize=8.5,
                    arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})
        ax.set_title(title)
        ax.text(0.02, 0.03, f"{sizes.size:,} components in total", transform=ax.transAxes,
                color=INK_2, fontsize=8.5)
        fig.tight_layout()
        return _save(fig, path, dpi)


def plot_distance_distribution(histogram: np.ndarray, path: Path, dpi: int, title: str,
                               is_exact: bool, average: float) -> Path:
    """Fraction of (ordered) node pairs at each distance in the largest component."""
    dist = np.arange(histogram.size)[1:]
    frac = histogram[1:] / max(1, histogram[1:].sum())
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.bar(dist, frac, width=0.8, color=SERIES[0], edgecolor="white", linewidth=0.5)
        ax.axvline(average, color=INK, linewidth=1, linestyle="--")
        ax.text(0.98, 0.97, f"mean distance = {average:.3f} (dashed)\nmax distance = {dist.max()}",
                transform=ax.transAxes, ha="right", va="top", color=INK_2, fontsize=8.5)
        ax.set_xlabel("shortest-path distance (hops)")
        ax.set_ylabel("fraction of node pairs" if is_exact else "fraction of sampled pairs")
        ax.set_title(title + ("" if is_exact else " — APPROXIMATE (sampled sources)"))
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        return _save(fig, path, dpi)


def _category_colors(labels: np.ndarray, names: list[str], top: int = 7
                     ) -> tuple[list[str], list[tuple[str, str]]]:
    counts = np.bincount(labels, minlength=len(names))
    order = np.argsort(-counts)[:top]
    palette = {int(c): SERIES[i] for i, c in enumerate(order)}
    colors = [palette.get(int(lab), OTHER) for lab in labels]
    legend = [(names[int(c)], palette[int(c)]) for c in order] + [("other", OTHER)]
    return colors, legend


def plot_sample_graph(edges: np.ndarray, labels: np.ndarray, names: list[str], path: Path,
                      dpi: int, title: str, seed: int) -> Path:
    """Force-directed drawing of a small connected sample (nodes coloured by category)."""
    import random

    import igraph as ig

    random.seed(seed)
    graph = ig.Graph(n=labels.size, edges=edges.tolist())
    layout = np.asarray(graph.layout_fruchterman_reingold(niter=1000).coords)
    colors, legend = _category_colors(labels, names)
    degrees = np.asarray(graph.degree())
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(8, 8))
        segments = layout[edges]
        ax.add_collection(LineCollection(segments, colors="#9f9e98", linewidths=0.25, alpha=0.5))
        ax.scatter(layout[:, 0], layout[:, 1], s=4 + 3 * np.sqrt(degrees), c=colors,
                   edgecolors="white", linewidths=0.3, zorder=3)
        ax.set_axis_off()
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=n, markersize=6)
                   for n, c in legend]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8.5,
                  title="primary category")
        ax.set_title(title)
        fig.tight_layout()
        return _save(fig, path, dpi)


def plot_area_heatmap(matrix: np.ndarray, names: list[str], path: Path, dpi: int,
                      title: str) -> Path:
    """Row-normalised citation matrix between areas (share of each area's references)."""
    made = matrix.sum(axis=1, keepdims=True)
    share = np.divide(matrix, made, out=np.zeros(matrix.shape, dtype=float), where=made > 0)
    k = matrix.shape[0]
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(9.5, 8.5))
        floor = 1e-4
        image = ax.imshow(np.clip(share, floor, 1), cmap=SEQUENTIAL,
                          norm=LogNorm(vmin=floor, vmax=1))
        ax.set_xticks(range(k), names, rotation=90, fontsize=7)
        ax.set_yticks(range(k), names, fontsize=7)
        ax.set_xlabel("cited area")
        ax.set_ylabel("citing area")
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        bar.set_label("share of the citing area's references (log scale)")
        ax.set_title(title)
        fig.tight_layout()
        return _save(fig, path, dpi)


def plot_area_graph(matrix: np.ndarray, names: list[str], num_papers: np.ndarray, path: Path,
                    dpi: int, top_edges: int, title: str) -> Path:
    """Aggregated area graph G_A: node size ∝ papers, colour = intra-area share,
    arrows = the ``top_edges`` heaviest inter-area citation flows (width ∝ citations)."""
    k = matrix.shape[0]
    angles = 2 * np.pi * np.arange(k) / k
    pos = np.column_stack([np.cos(angles), np.sin(angles)])
    made = matrix.sum(axis=1)
    intra_share = np.divide(np.diag(matrix), made, out=np.zeros(k), where=made > 0)
    off = matrix.astype(float).copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:top_edges]
    flat = flat[off.flat[flat] > 0]
    weights = off.flat[flat]
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(9, 9))
        wmax = weights.max() if weights.size else 1.0
        for idx, w in zip(flat, weights):
            i, j = divmod(int(idx), k)
            ax.annotate("", xy=pos[j], xytext=pos[i],
                        arrowprops={"arrowstyle": "-|>", "color": "#6f6e69",
                                    "alpha": 0.25 + 0.6 * w / wmax, "linewidth": 0.4 + 4 * w / wmax,
                                    "shrinkA": 8, "shrinkB": 8,
                                    "connectionstyle": "arc3,rad=0.15"})
        sizes = 30 + 900 * num_papers / max(1, num_papers.max())
        points = ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=intra_share, cmap=SEQUENTIAL,
                            vmin=0, vmax=1, edgecolors="#52514e", linewidths=0.6, zorder=3)
        for i in range(k):
            ax.text(1.17 * pos[i, 0], 1.17 * pos[i, 1], names[i], ha="center", va="center",
                    fontsize=7.5, color=INK)
        bar = fig.colorbar(points, ax=ax, fraction=0.04, pad=0.02, shrink=0.7)
        bar.set_label("intra-area share of references (self-loop weight / out-strength)")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        return _save(fig, path, dpi)

