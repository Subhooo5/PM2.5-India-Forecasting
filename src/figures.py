from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

matplotlib.use("Agg")

SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REF = "#52514e"
HATCH = [None, None, "///"]
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#256abf", "#86b6ef", "#f0efec", "#f19a96", "#c23a3a"])
DASH = (0, (3, 1.5))
LABEL_BOX = {"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.9}
DPI = 200


def _style(ax, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)


def _errorbars(ax, x, values, errs) -> None:
    for xi, v, e in zip(x, values, errs):
        if e is not None and np.isfinite(e):
            ax.errorbar(xi, v, yerr=e, color=INK, elinewidth=1, capsize=3, zorder=6)


def _legend(fig, handles, labels, y: float) -> None:
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(labels), fontsize=8, frameon=False)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def metric_ylim(metric: str, values, errs, refs) -> tuple[float, float]:
    """Zero-based limits for error metrics; R² limits also include negative bars, errors and references."""
    errs = [0.0 if (e is None or not np.isfinite(e)) else float(e) for e in errs]
    refs = [float(r) for r in refs if r is not None]
    top = max([v + e for v, e in zip(values, errs)] + refs)
    if metric != "R²":
        return 0.0, top * 1.18
    lo, hi = min([0.0] + [v - e for v, e in zip(values, errs)] + refs), max(0.0, top)
    if lo >= 0.0:
        return 0.0, (hi * 1.18) if hi > 0 else 1.0
    return lo - 0.12 * (hi - lo), hi + 0.18 * (hi - lo)


def performance_comparison(perf: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    for ax, (metric, d) in zip(axes, perf.items()):
        x = np.arange(len(d["labels"]))
        bars = ax.bar(x, d["values"], width=0.68, color=[SLOT[g] for g in d["groups"]], edgecolor=SURFACE, linewidth=1.5)
        _errorbars(ax, x, d["values"], d["err"])
        for bar, g in zip(bars, d["groups"]):
            if HATCH[g]:
                bar.set_hatch(HATCH[g])
        for xi, ref in zip(x, d["persistence"]):
            if ref is not None:
                ax.plot([xi - 0.38, xi + 0.38], [ref, ref], color=REF, linewidth=2, linestyle=DASH, zorder=5)
        fmt = "{:.3f}" if metric == "R²" else "{:.1f}"
        for bar, v, e in zip(bars, d["values"], d["err"]):
            neg = v < 0
            ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, v + (e or 0) if neg else v - (e or 0)),
                        xytext=(0, 3 if neg else -3), textcoords="offset points", ha="center",
                        va="bottom" if neg else "top", fontsize=7, color=INK, zorder=7, bbox=LABEL_BOX)
        ax.axvline(1.5, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xticks(x, d["labels"], fontsize=8, color=INK)
        _style(ax, metric + (" (µg/m³)" if metric != "R²" else ""))
        ax.set_title(metric, fontsize=10, color=INK, loc="left")
        y0, y1 = metric_ylim(metric, d["values"], d["err"], d["persistence"])
        ax.set_ylim(y0, y1)
        if y0 < 0:
            ax.axhline(0, color=INK2, linewidth=0.8, zorder=4)
        y_txt = y1 - 0.03 * (y1 - y0)
        ax.text(0.5, y_txt, "same-day\nestimation", ha="center", va="top", fontsize=7, color=INK2)
        ax.text(3.0, y_txt, "genuine forecasting", ha="center", va="top", fontsize=7, color=INK2)
    _legend(fig, [Rectangle((0, 0), 1, 1, color=SLOT[0]), Rectangle((0, 0), 1, 1, color=SLOT[1]),
                  Rectangle((0, 0), 1, 1, facecolor=SLOT[2], hatch="///", edgecolor=SURFACE),
                  Line2D([0], [0], color=REF, linewidth=2, linestyle=DASH)],
            ["A: random split (mean ± sd, 5 seeds)", "B: chronological split", "C: causal forecast (h = 1, 3, 7)",
             "Persistence, same rows (C only)"], -0.11)
    fig.suptitle("Leakage audit: point accuracy on each protocol's own test set "
                 "(A/B = same-day estimation, C = forecasting; different tasks)", fontsize=10, color=INK)
    return _save(fig, path)


def leakage_inflation(infl: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), constrained_layout=True)
    for ax, (metric, d) in zip(axes, infl.items()):
        x = np.arange(len(d["labels"]))
        bars = ax.bar(x, d["values"], width=0.62, color=[SLOT[g] for g in d["groups"]], edgecolor=SURFACE, linewidth=1.5)
        _errorbars(ax, x, d["values"], d["err"])
        ax.axhline(0, color=INK2, linewidth=0.8)
        for bar, v, e in zip(bars, d["values"], d["err"]):
            ax.annotate(f"{v:.1f}%", (bar.get_x() + bar.get_width() / 2, v + (e or 0) if v >= 0 else v - (e or 0)),
                        xytext=(0, 3 if v >= 0 else -10), textcoords="offset points", ha="center", fontsize=7, color=INK)
        ax.set_xticks(x, d["labels"], fontsize=8, color=INK)
        _style(ax, f"Relative {metric} inflation (%)")
        ax.set_title(metric, fontsize=10, color=INK, loc="left")
        lo = min(0, min(v - (e or 0) for v, e in zip(d["values"], d["err"])))
        ax.set_ylim(lo * 1.25 - 2, max(v + (e or 0) for v, e in zip(d["values"], d["err"])) * 1.2 + 2)
    _legend(fig, [Rectangle((0, 0), 1, 1, color=SLOT[0]), Rectangle((0, 0), 1, 1, color=SLOT[1])],
            ["A vs B, matched 2019 city-dates (mean ± sd, 5 seeds)",
             "B vs C, common city-date set per horizon (two different tasks)"], -0.12)
    fig.suptitle("Leakage audit: relative inflation = 100·(1 − error_left / error_right); left = optimistic protocol",
                 fontsize=10, color=INK)
    return _save(fig, path)


def heatmaps(mats: dict, cities: list, horizons: list, path: Path, title: str, cbar_label: str,
             ncols: int | None = None, fmt: str = "{:+.3f}") -> Path:
    ncols = ncols or len(mats)
    nrows = -(-len(mats) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols + 1, 3.6 * nrows), constrained_layout=True)
    lim = max(float(np.nanmax(np.abs(m))) for m in mats.values()) or 1.0
    for ax, (ptitle, m) in zip(np.atleast_1d(axes).ravel(), mats.items()):
        im = ax.imshow(m, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if abs(m[i, j]) > 0.6 * lim else INK)
        ax.set_xticks(range(len(horizons)), [f"h = {h}" for h in horizons], fontsize=8, color=INK)
        ax.set_yticks(range(len(cities)), cities, fontsize=8, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(ptitle, fontsize=9, color=INK, loc="left")
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label(cbar_label, fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    if title:
        fig.suptitle(title, fontsize=10, color=INK)
    return _save(fig, path)


def model_comparison(cities: list, horizons: list, mae: dict, persistence: dict, path: Path) -> Path:
    models = [("pooled", "Pooled (5 core cities)"), ("per_city", "Per-city"), ("loco", "Leave-one-city-out")]
    fig, axes = plt.subplots(1, len(horizons), figsize=(11, 3.6), sharey=True, constrained_layout=True)
    w = 0.26
    for ax, h in zip(axes, horizons):
        x = np.arange(len(cities))
        for k, (key, _) in enumerate(models):
            ax.bar(x + (k - 1) * w, mae[key][h], width=w, color=SLOT[k], hatch=HATCH[k], edgecolor=SURFACE, linewidth=1.2)
        for xi, ref in zip(x, persistence[h]):
            ax.plot([xi - 1.5 * w, xi + 1.5 * w], [ref, ref], color=REF, linewidth=2, linestyle=DASH, zorder=5)
        ax.set_xticks(x, cities, fontsize=8, color=INK, rotation=20)
        _style(ax, "MAE (µg/m³)" if h == horizons[0] else None)
        ax.set_title(f"h = {h} day{'s' if h > 1 else ''}", fontsize=10, color=INK, loc="left")
    _legend(fig, [Rectangle((0, 0), 1, 1, color=SLOT[0]), Rectangle((0, 0), 1, 1, color=SLOT[1]),
                  Rectangle((0, 0), 1, 1, facecolor=SLOT[2], hatch="///", edgecolor=SURFACE),
                  Line2D([0], [0], color=REF, linewidth=2, linestyle=DASH)],
            [m[1] for m in models] + ["Persistence (same rows)"], -0.1)
    fig.suptitle("Cross-city: 2019 test MAE by city and horizon (common eval_mask rows; numbers in cross_city_*.csv)",
                 fontsize=10, color=INK)
    return _save(fig, path)


def bars(panels: dict, path: Path, ylabel: str, title: str, ref: float | None = None) -> Path:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6), sharey=True, constrained_layout=True,
                             squeeze=False)
    top = max(max(max(v) for v in d["series"].values()) for d in panels.values())
    for ax, (ptitle, d) in zip(axes[0], panels.items()):
        x, k = np.arange(len(d["x"])), len(d["series"])
        w = 0.8 / k
        for i, (name, vals) in enumerate(d["series"].items()):
            for b, v in zip(ax.bar(x + (i - (k - 1) / 2) * w, vals, width=w, color=SLOT[i], hatch=HATCH[i],
                                   edgecolor=SURFACE, linewidth=1.2, label=name), vals):
                ax.annotate(f"{v:.3f}" if top < 2 else f"{v:.1f}", (b.get_x() + w / 2, v), xytext=(0, -3),
                            textcoords="offset points", ha="center", va="top", fontsize=6.5, color=INK, zorder=7,
                            bbox={**LABEL_BOX, "boxstyle": "round,pad=0.12"})
        if ref is not None:
            ax.axhline(ref, color=REF, linewidth=1.5, linestyle=DASH, zorder=5, label=f"nominal {ref:.2f}")
        ax.set_xticks(x, d["x"], fontsize=8, color=INK)
        ax.set_ylim(0, max(top, ref or 0) * 1.22)
        _style(ax, ylabel if ax is axes[0][0] else None)
        ax.set_title(ptitle, fontsize=9, color=INK, loc="left")
    handles, labels = axes[0][0].get_legend_handles_labels()
    _legend(fig, handles, labels, -0.1)
    fig.suptitle(title, fontsize=10, color=INK)
    return _save(fig, path)


def stream_grid(series: dict, cities: list, horizons: list, path: Path, ylabel: str, title: str, refs: dict,
                series2: dict | None = None, label2: str = "", vline: tuple | None = None, col_fmt: str = "h = {}",
                xlabel: str = "2019 (date)", sharey: bool = True) -> Path:
    """Small multiples (rows = cities, columns = horizons); vline = (x or {column: x}, (left label, right label))."""
    fig, axes = plt.subplots(len(cities), len(horizons), figsize=(10.5, 1.55 * len(cities) + 0.8), sharex=True,
                             sharey=sharey, constrained_layout=True, squeeze=False)
    for r, city in enumerate(cities):
        for c, h in enumerate(horizons):
            ax = axes[r][c]
            if series2:
                ax.plot(*series2[city, h], color=INK2, linewidth=0.9, alpha=0.45)
            ax.plot(*series[city, h], color=SLOT[0], linewidth=1.2)
            for v in refs:
                ax.axhline(v, color=REF, linewidth=1, linestyle=DASH)
            if vline:
                xv = vline[0][h] if isinstance(vline[0], dict) else vline[0]
                ax.axvline(xv, color=INK, linewidth=0.9)
                for lab, ha, dx in zip(vline[1] if r == 0 else (), ("right", "left"), (-3, 3)):
                    ax.annotate(lab, (xv, 1.0), xycoords=("data", "axes fraction"), xytext=(dx, -2),
                                textcoords="offset points", ha=ha, va="top", fontsize=7, color=INK, zorder=7,
                                bbox=LABEL_BOX)
            _style(ax, city if c == 0 else None)
            if r == 0:
                ax.set_title(col_fmt.format(h), fontsize=9, color=INK, loc="left")
            ax.tick_params(axis="x", labelrotation=30)
    fig.supxlabel(xlabel, fontsize=8, color=INK)
    handles = [(Line2D([0], [0], color=SLOT[0], linewidth=1.2), ylabel)]
    handles += [(Line2D([0], [0], color=INK2, linewidth=0.9, alpha=0.45), label2)] if series2 else []
    handles += [(Line2D([0], [0], color=REF, linewidth=1, linestyle=DASH), " / ".join(refs.values()))] if refs else []
    _legend(fig, [h for h, _ in handles], [lab for _, lab in handles], -0.04)
    fig.suptitle(f"{title}\n{ylabel}", fontsize=10, color=INK)
    return _save(fig, path)
