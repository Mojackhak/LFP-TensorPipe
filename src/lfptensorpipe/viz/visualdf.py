# -*- coding: utf-8 -*-
"""
visualdf.py

Plotting helpers for single-effect series/scalar and DataFrame heatmap APIs.
"""

from __future__ import annotations

import heapq
import re
from typing import Dict, List, Optional, Tuple, Union
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ----------------------- global matplotlib defaults -----------------------

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.major.pad'] = 1
mpl.rcParams['ytick.major.pad'] = 1
mpl.rcParams['axes.labelpad'] = 1
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['xtick.minor.size'] = 1
mpl.rcParams['ytick.minor.size'] = 1
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1

MM_PER_INCH = 25.4

DEFAULT_BOXSIZE_MM: Tuple[float, float] = (60, 50.0)

_OUTSIDE_LEGEND_SPECS: Dict[str, Tuple[str, Tuple[float, float]]] = {
    # New (preferred) API, matching visualdf_old.
    "outside_top": ("upper center", (0.5, 1.02)),
    "outside_bottom": ("lower center", (0.5, -0.04)),
    "outside_right": ("center left", (1.02, 0.5)),
    "outside_left": ("center right", (-0.02, 0.5)),

    # Legacy shorthands kept for backward compatibility.
    "right": ("right", (1.02, 0.5)),
    # "left" is not a valid Matplotlib loc string; interpret it as outside_left.
    "left": ("center right", (-0.02, 0.5)),
    "upper right": ("upper right", (1.02, 1.0)),
    "lower right": ("lower right", (1.02, 0.0)),
    "upper left": ("upper left", (-0.02, 1.0)),
    "lower left": ("lower left", (-0.02, 0.0)),
}

# ----------------------- p-value helpers -----------------------

_P_VALUE_PRIORITY: Tuple[str, ...] = ("p_across", "p_tukey", "p.value", "p_raw")

_P_VALUE_FALLBACK: Tuple[str, ...] = (
    "pvalue",
    "p",
    "p_val",
    "p_value",
    "pval",
    "pval_adj",
    "Pr(>F)",
    "Pr(>|t|)",
    "q",
    "qval",
    "q_value",
)


def _mm_to_in(x_mm: float) -> float:
    """Convert millimeters to inches."""
    return float(x_mm) / MM_PER_INCH


def _tuple_mm_to_in(size_mm: Tuple[float, float]) -> Tuple[float, float]:
    """Convert a (w_mm, h_mm) tuple to inches."""
    return _mm_to_in(size_mm[0]), _mm_to_in(size_mm[1])


def _ordered_levels(series, user_levels):
    """Return ordered unique levels, preserving user-specified order if provided."""
    if user_levels is not None:
        return list(user_levels)
    if isinstance(series.dtype, pd.CategoricalDtype):
        return list(series.cat.categories)
    return list(pd.unique(series.dropna()))


def _rgba_color(c):
    """Convert a color specification to an RGBA tuple.

    Supported inputs:
      - Any Matplotlib color (e.g., "gray", (r, g, b), (r, g, b, a))
      - Hex "#RRGGBB"
      - Hex with alpha "#RRGGBBAA" (alpha last; matches Matplotlib/CSS)

    Notes:
      - "#AARRGGBB" (alpha first) is intentionally NOT supported.

    If parsing fails, the original value is returned unchanged.
    """
    if c is None:
        return c

    if isinstance(c, str) and c.startswith("#"):
        s = c.strip()
        try:
            # Matplotlib accepts #RRGGBB and #RRGGBBAA (alpha last).
            return mpl.colors.to_rgba(s)
        except ValueError:
            return c

    try:
        return mpl.colors.to_rgba(c)
    except Exception:
        return c


def _build_color_map(levels, palette):
    """
    Build a dict mapping each level -> color.

    palette can be:
    - a matplotlib colormap name
    - a list of colors
    - a dict mapping levels -> colors
    """
    if isinstance(palette, dict):
        return dict(palette)

    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        n = max(1, len(levels))
        cols = [cmap(i / (n - 1 if n > 1 else 1)) for i in range(n)]
        return dict(zip(levels, cols))

    cols = list(palette)
    if len(cols) < len(levels):
        raise ValueError("Palette list is shorter than the number of levels.")
    return dict(zip(levels, cols))


def _series_grid_stats(series_list: List[pd.Series], n_grid: int = 400, ribbon: str = "sem"):
    """
    Aggregate multiple pd.Series on a common x-grid.

    Returns:
        x, mean, low, high  (arrays)
    """
    if len(series_list) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    x_min = np.nanmin([s.index.min() for s in series_list if len(s) > 0])
    x_max = np.nanmax([s.index.max() for s in series_list if len(s) > 0])
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        return np.array([]), np.array([]), np.array([]), np.array([])

    x_grid = np.linspace(x_min, x_max, int(n_grid))

    ys = []
    for s in series_list:
        if not isinstance(s, pd.Series) or s.empty:
            continue
        y_interp = np.interp(x_grid, s.index.to_numpy(dtype=float), s.to_numpy(dtype=float))
        ys.append(y_interp)

    if len(ys) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    Y = np.vstack(ys)
    mean = np.nanmean(Y, axis=0)
    sd = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean)
    sem = sd / np.sqrt(max(1, Y.shape[0]))

    if ribbon == "sd":
        low, high = mean - sd, mean + sd
    elif ribbon == "95%ci":
        low, high = mean - 1.96 * sem, mean + 1.96 * sem
    else:  # "sem"
        low, high = mean - sem, mean + sem

    return x_grid, mean, low, high


def _safe_axvspan(ax, x0: float, x1: float, **kwargs):
    """axvspan that avoids invalid spans on log scales."""
    if ax.get_xscale() == "log":
        if x1 <= 0:
            return None
        x0 = max(x0, np.finfo(float).tiny)
    return ax.axvspan(x0, x1, **kwargs)


def _safe_axhspan(ax, y0: float, y1: float, **kwargs):
    """axhspan that avoids invalid spans on log scales."""
    if ax.get_yscale() == "log":
        if y1 <= 0:
            return None
        y0 = max(y0, np.finfo(float).tiny)
    return ax.axhspan(y0, y1, **kwargs)


def _to_numeric_index(idx):
    """
    Convert an index/array to a numeric numpy array when possible.

    Priority:
      1) Already-numeric dtype -> float
      2) pd.to_numeric (handles numeric strings) when fully convertible
      3) datetime-like -> seconds since epoch
      4) fallback -> 0..N-1
    """
    if isinstance(idx, pd.Index):
        idx = idx.to_numpy()
    arr = np.asarray(idx)

    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float)

    # Try numeric conversion (e.g., "10", "20.5")
    try:
        num = pd.to_numeric(arr, errors="coerce")
        num_arr = np.asarray(num, dtype=float)
        if np.all(np.isfinite(num_arr)):
            return num_arr
    except Exception:
        pass

    dt = _coerce_datetime_index(arr)
    if dt is not None:
        return dt.astype("int64").to_numpy(dtype=float) / 1e9

    return np.arange(len(arr), dtype=float)


def _coerce_datetime_index(arr: np.ndarray) -> Optional[pd.Index]:
    """Return a datetime index only when every non-null value is datetime-like."""
    if arr.size == 0:
        return None

    if np.issubdtype(arr.dtype, np.datetime64):
        try:
            return pd.to_datetime(arr, errors="raise")
        except Exception:
            return None

    if arr.dtype.kind not in {"O", "U", "S"}:
        return None

    try:
        # Probe with mixed-format parsing so categorical string axes do not emit
        # pandas' per-element datetime inference warning.
        probe = pd.to_datetime(arr, errors="coerce", format="mixed")
    except Exception:
        return None

    arr_na = np.asarray(pd.isna(arr), dtype=bool)
    probe_na = np.asarray(pd.isna(probe), dtype=bool)
    if not np.all(arr_na | ~probe_na):
        return None

    try:
        return pd.to_datetime(arr, errors="raise", format="mixed")
    except Exception:
        return None


def _measure_text_inches(
    text: str,
    fontsize: float = 12.0,
    rotation: float = 0.0,
    font_family: str = "Arial",
    dpi: int = 300,
) -> Tuple[float, float]:
    """Measure rendered text size in inches using a tiny offscreen figure."""
    if not text:
        return (0.0, 0.0)

    fig = plt.figure(figsize=(2, 2), dpi=dpi)
    fig.patch.set_alpha(0)
    plt.rcParams["font.family"] = font_family

    t = fig.text(0, 0, text, fontsize=fontsize, rotation=rotation)
    fig.canvas.draw()
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    plt.close(fig)

    return bbox.width / dpi, bbox.height / dpi


def _compute_fig_layout_from_box_mm(
    *,
    nrows: int,
    ncols: int,
    boxsize_mm: Tuple[float, float],
    panel_gap_mm: Tuple[float, float] = (0.0, 0.0),
    strip_top_height_mm: float = 0.0,
    strip_right_width_mm: float = 0.0,
    strip_pad_mm: float = 0.0,
    colorbar_width_mm: float = 0.0,
    colorbar_pad_mm: float = 0.0,
    single_x_label: bool = True,
    single_y_label: bool = True,
    axis_label_fontsize: float = 16.0,
    include_global_label_margins: bool = True,
    x_label_text: str = "",
    y_label_text: str = "",
    colorbar_label_text: Optional[str] = None,
    x_label_offset_mm: float = 0.0,
    y_label_offset_mm: float = 0.0,
    cbar_label_offset_mm: float = 0.0,
    font_family: str = "Arial",
    dpi: int = 300,
) -> Dict[str, float | Tuple[float, float] | List[float]]:
    """
    Compute deterministic figure size and grid rectangle from physical mm inputs.

    Returns a layout dict containing:
        - figsize_in: (w_in, h_in)
        - fig_w_in, fig_h_in
        - rect: [L, B, R, T] in figure fractions for the panel grid region
        - box_w_in, box_h_in, gap_x_in, gap_y_in
        - strip_top_h_in, strip_right_w_in, strip_pad_in
        - cb_w_in, cb_pad_in
        - x_off_in, y_off_in, cbar_off_in (label offsets in inches)
    """
    box_w_in, box_h_in = _tuple_mm_to_in(boxsize_mm)
    gap_x_in, gap_y_in = _tuple_mm_to_in(panel_gap_mm)

    strip_top_h_in = _mm_to_in(strip_top_height_mm)
    strip_right_w_in = _mm_to_in(strip_right_width_mm)
    strip_pad_in = _mm_to_in(strip_pad_mm)

    cb_w_in = _mm_to_in(colorbar_width_mm)
    cb_pad_in = _mm_to_in(colorbar_pad_mm)

    x_off_in = _mm_to_in(x_label_offset_mm) if single_x_label else 0.0
    y_off_in = _mm_to_in(y_label_offset_mm) if single_y_label else 0.0
    cbar_off_in = _mm_to_in(cbar_label_offset_mm) if colorbar_label_text else 0.0

    grid_w_in = ncols * box_w_in + (ncols - 1) * gap_x_in
    grid_h_in = nrows * box_h_in + (nrows - 1) * gap_y_in

    top_extras_in = (strip_top_h_in + strip_pad_in) if strip_top_h_in > 0 else 0.0

    right_extras_in = 0.0
    if strip_right_w_in > 0:
        right_extras_in += strip_pad_in + strip_right_w_in
    if cb_w_in > 0:
        right_extras_in += cb_pad_in + cb_w_in

    # Space for global axis labels
    pad_in = 0.03  # ~0.76 mm
    x_label_h_in = 0.0
    y_label_w_in = 0.0

    if include_global_label_margins:
        if single_x_label and x_label_text:
            _, x_label_h_in = _measure_text_inches(
                x_label_text, fontsize=axis_label_fontsize, font_family=font_family, dpi=dpi
            )
        if single_y_label and y_label_text:
            y_label_w_in, _ = _measure_text_inches(
                y_label_text, fontsize=axis_label_fontsize, rotation=90, font_family=font_family, dpi=dpi
            )

    bottom_margin_in = (x_off_in + x_label_h_in + pad_in) if (include_global_label_margins and single_x_label) else 0.0
    left_margin_in = (y_off_in + y_label_w_in + pad_in) if (include_global_label_margins and single_y_label) else 0.0

    # Space for colorbar label (outside to the right)
    if include_global_label_margins and colorbar_label_text:
        cbar_label_w_in, _ = _measure_text_inches(
            colorbar_label_text, fontsize=axis_label_fontsize, rotation=270, font_family=font_family, dpi=dpi
        )
        right_extras_in += cbar_off_in + cbar_label_w_in + pad_in

    fig_w_in = left_margin_in + grid_w_in + right_extras_in
    fig_h_in = bottom_margin_in + grid_h_in + top_extras_in

    if fig_w_in <= 0 or fig_h_in <= 0:
        raise ValueError("Invalid figure size computed from box layout. Check inputs.")

    L = left_margin_in / fig_w_in
    B = bottom_margin_in / fig_h_in
    R = 1.0 - (right_extras_in / fig_w_in)
    T = 1.0 - (top_extras_in / fig_h_in)

    return {
        "figsize_in": (fig_w_in, fig_h_in),
        "fig_w_in": fig_w_in,
        "fig_h_in": fig_h_in,
        "rect": [L, B, R, T],
        "box_w_in": box_w_in,
        "box_h_in": box_h_in,
        "gap_x_in": gap_x_in,
        "gap_y_in": gap_y_in,
        "strip_top_h_in": strip_top_h_in,
        "strip_right_w_in": strip_right_w_in,
        "strip_pad_in": strip_pad_in,
        "cb_w_in": cb_w_in,
        "cb_pad_in": cb_pad_in,
        "x_off_in": x_off_in,
        "y_off_in": y_off_in,
        "cbar_off_in": cbar_off_in,
    }


def _place_panels_fixed(
    fig,
    axes,
    rect: List[float],
    box_w_in: float,
    box_h_in: float,
    gap_x_in: float,
    gap_y_in: float,
    align: str = "left",
):
    """
    Place a (nrows, ncols) axes grid with fixed per-panel sizes (inches) inside rect.

    rect is in figure fractions: [L, B, R, T] where R and T are absolute figure fractions.
    """
    L, B, R, T = rect
    fig_w_in, fig_h_in = fig.get_size_inches()

    total_w_frac = R - L
    total_h_frac = T - B
    avail_w_in = total_w_frac * fig_w_in
    avail_h_in = total_h_frac * fig_h_in

    nrows, ncols = axes.shape
    need_w_in = ncols * box_w_in + (ncols - 1) * gap_x_in
    need_h_in = nrows * box_h_in + (nrows - 1) * gap_y_in

    if align == "center":
        x0_in = (avail_w_in - need_w_in) / 2.0
    elif align == "right":
        x0_in = avail_w_in - need_w_in
    else:
        x0_in = 0.0

    y0_in = 0.0

    base_x0 = L + (x0_in / fig_w_in)
    base_y0 = B + (y0_in / fig_h_in)

    box_w_frac = box_w_in / fig_w_in
    box_h_frac = box_h_in / fig_h_in
    gap_x_frac = gap_x_in / fig_w_in
    gap_y_frac = gap_y_in / fig_h_in

    for i in range(nrows):
        for j in range(ncols):
            x0 = base_x0 + j * (box_w_frac + gap_x_frac)
            y0 = base_y0 + (nrows - 1 - i) * (box_h_frac + gap_y_frac)
            axes[i, j].set_position([x0, y0, box_w_frac, box_h_frac])


def _add_strips_mm(
    fig,
    axes,
    *,
    col_labels: Optional[List[str]] = None,
    row_labels: Optional[List[str]] = None,
    strip_top_height_mm: float = 0.0,
    strip_right_width_mm: float = 0.0,
    strip_pad_mm: float = 0.0,
    label_fontsize: float = 16.0,
    label_top_bg_color: str = "lightgray",
    label_right_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
):
    """
    Add facet strips using absolute mm units.

    Returns:
        right_edge (figure fraction): right-most edge after right strips (for colorbar placement).
    """
    nrows, ncols = axes.shape
    fig_w_in, fig_h_in = fig.get_size_inches()

    top_h_in = _mm_to_in(strip_top_height_mm)
    right_w_in = _mm_to_in(strip_right_width_mm)
    pad_in = _mm_to_in(strip_pad_mm)

    top_h_frac = top_h_in / fig_h_in if top_h_in > 0 else 0.0
    right_w_frac = right_w_in / fig_w_in if right_w_in > 0 else 0.0
    pad_y_frac = pad_in / fig_h_in if pad_in > 0 else 0.0
    pad_x_frac = pad_in / fig_w_in if pad_in > 0 else 0.0

    # Top strips
    if col_labels is not None and top_h_frac > 0:
        for j, lab in enumerate(col_labels):
            pos = axes[0, j].get_position()
            ax_strip = fig.add_axes([pos.x0, pos.y1 + pad_y_frac, pos.width, top_h_frac])
            ax_strip.set_facecolor(label_top_bg_color)
            ax_strip.text(
                0.5,
                0.5,
                str(lab),
                ha="center",
                va="center",
                fontsize=label_fontsize,
                color=label_text_color,
                fontweight=label_fontweight,
            )
            ax_strip.set_xticks([])
            ax_strip.set_yticks([])
            for spine in ax_strip.spines.values():
                spine.set_visible(False)

    # Right strips
    right_edge = 0.0
    if row_labels is not None and right_w_frac > 0:
        for i, lab in enumerate(row_labels):
            pos = axes[i, ncols - 1].get_position()
            ax_strip = fig.add_axes([pos.x1 + pad_x_frac, pos.y0, right_w_frac, pos.height])
            ax_strip.set_facecolor(label_right_bg_color)
            ax_strip.text(
                0.5,
                0.5,
                str(lab),
                ha="center",
                va="center",
                rotation=-90,
                fontsize=label_fontsize,
                color=label_text_color,
                fontweight=label_fontweight,
            )
            ax_strip.set_xticks([])
            ax_strip.set_yticks([])
            for spine in ax_strip.spines.values():
                spine.set_visible(False)
            right_edge = max(right_edge, pos.x1 + pad_x_frac + right_w_frac)
    else:
        # Right edge defaults to the rightmost panel edge
        right_edge = max(ax.get_position().x1 for ax in axes.ravel())

    return right_edge


def _init_box_figure(
    *,
    nrows: int,
    ncols: int,
    boxsize_mm: Tuple[float, float],
    panel_gap_mm: Tuple[float, float] = (0.0, 0.0),
    strip_top_height_mm: float = 0.0,
    strip_right_width_mm: float = 0.0,
    strip_pad_mm: float = 0.0,
    colorbar_width_mm: float = 0.0,
    colorbar_pad_mm: float = 0.0,
    single_x_label: bool = True,
    single_y_label: bool = True,
    axis_label_fontsize: float = 16.0,
    include_global_label_margins: bool = True,
    x_label_text: str = "",
    y_label_text: str = "",
    colorbar_label_text: Optional[str] = None,
    x_label_offset_mm: float = 0.0,
    y_label_offset_mm: float = 0.0,
    cbar_label_offset_mm: float = 0.0,
    dpi: int = 300,
    font_family: str = "Arial",
    sharex: bool = False,
    sharey: bool = False,
    transparent: bool = False,
):
    """Create a fixed-geometry figure and axes grid from mm layout parameters."""
    plt.rcParams["font.family"] = font_family

    layout = _compute_fig_layout_from_box_mm(
        nrows=nrows,
        ncols=ncols,
        boxsize_mm=boxsize_mm,
        panel_gap_mm=panel_gap_mm,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=strip_right_width_mm,
        strip_pad_mm=strip_pad_mm,
        colorbar_width_mm=colorbar_width_mm,
        colorbar_pad_mm=colorbar_pad_mm,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=x_label_text,
        y_label_text=y_label_text,
        colorbar_label_text=colorbar_label_text,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=cbar_label_offset_mm,
        font_family=font_family,
        dpi=dpi,
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=layout["figsize_in"],  # inches
        dpi=dpi,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )

    if transparent:
        fig.patch.set_alpha(0)
        for ax in axes.ravel():
            ax.patch.set_alpha(0)

    _place_panels_fixed(
        fig,
        axes,
        rect=layout["rect"],
        box_w_in=layout["box_w_in"],
        box_h_in=layout["box_h_in"],
        gap_x_in=layout["gap_x_in"],
        gap_y_in=layout["gap_y_in"],
        align="left",
    )

    return fig, axes, layout


def _draw_global_labels_and_title(
    fig,
    layout,
    *,
    x_label: Optional[str],
    y_label: Optional[str],
    axis_label_fontsize: float,
    title: Optional[str],
    title_fontsize: float = 18.0,
    single_x_label: bool = True,
    single_y_label: bool = True,
    font_family: str = "Arial",
):
    """Draw global X/Y labels (using mm offsets) and a title."""
    plt.rcParams["font.family"] = font_family
    L, B, R, T = layout["rect"]
    fig_w_in = float(layout["fig_w_in"])
    fig_h_in = float(layout["fig_h_in"])

    cx = L + (R - L) / 2.0
    cy = B + (T - B) / 2.0

    if single_x_label and x_label:
        x_off_frac = float(layout["x_off_in"]) / fig_h_in
        fig.text(cx, B - x_off_frac, x_label, ha="center", va="top", fontsize=axis_label_fontsize)

    if single_y_label and y_label:
        y_off_frac = float(layout["y_off_in"]) / fig_w_in
        fig.text(L - y_off_frac, cy, y_label, ha="right", va="center", rotation=90, fontsize=axis_label_fontsize)

    if title:
        fig.suptitle(title, fontsize=title_fontsize, y=min(0.995, T + 0.01))


def _add_global_colorbar(
    fig,
    layout,
    mappable,
    *,
    right_edge: float,
    colorbar_label: Optional[str],
    tick_label_fontsize: float,
    axis_label_fontsize: float,
    font_family: str = "Arial",
):
    """Add a global colorbar using mm-based width/padding and label offset."""
    plt.rcParams["font.family"] = font_family
    L, B, R, T = layout["rect"]
    fig_w_in = float(layout["fig_w_in"])

    cb_pad_in = float(layout["cb_pad_in"])
    cb_w_in = float(layout["cb_w_in"])
    if cb_w_in <= 0:
        return None

    cb_pad_frac = cb_pad_in / fig_w_in
    cb_w_frac = cb_w_in / fig_w_in

    x0 = right_edge + cb_pad_frac
    # Keep within figure bounds
    if x0 + cb_w_frac > 0.99:
        x0 = max(0.0, 0.99 - cb_w_frac)

    cax = fig.add_axes([x0, B, cb_w_frac, T - B])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(labelsize=tick_label_fontsize)
    # Keep scientific-notation offset text aligned with the colorbar tick labels.
    cbar.ax.xaxis.get_offset_text().set_fontsize(tick_label_fontsize)
    cbar.ax.yaxis.get_offset_text().set_fontsize(tick_label_fontsize)

    if colorbar_label:
        cbar_off_frac = float(layout["cbar_off_in"]) / fig_w_in
        cx_right = x0 + cb_w_frac
        fig.text(
            cx_right + cbar_off_frac,
            B + (T - B) / 2.0,
            colorbar_label,
            ha="left",
            va="center",
            rotation=90,
            fontsize=axis_label_fontsize,
        )

    return cbar


def _legend_outside_spec(loc: str) -> Optional[Tuple[str, Tuple[float, float]]]:
    """Return (legend_loc, bbox_to_anchor) for supported outside legend placements."""
    if loc is None:
        return None
    key = str(loc).strip().lower()
    return _OUTSIDE_LEGEND_SPECS.get(key)


def _is_outside_legend_loc(loc: str) -> bool:
    """True if loc is one of the supported outside legend placements."""
    return _legend_outside_spec(loc) is not None


def _place_legend(
    fig,
    ax,
    handles,
    labels,
    *,
    legend_loc: str,
    legend_fontsize: float,
    legend_ncol: int,
    legend_framealpha: float,
    inside_map: bool,
):
    """Standardized legend placement (inside or outside an anchor axis)."""
    if inside_map:
        ax.legend(
            handles,
            labels,
            loc=legend_loc,
            fontsize=legend_fontsize,
            framealpha=legend_framealpha,
            ncol=legend_ncol,
        )
        return

    spec = _legend_outside_spec(legend_loc)
    if spec is None:
        out_loc, anchor = (str(legend_loc), (1.02, 0.5))
    else:
        out_loc, anchor = spec

    fig.legend(
        handles,
        labels,
        loc=out_loc,
        bbox_to_anchor=anchor,
        fontsize=legend_fontsize,
        framealpha=legend_framealpha,
        ncol=legend_ncol,
    )


def _ensure_strip_background_opaque(fig):
    """
    If figure background is transparent, ensure strip axes are fully opaque
    (matplotlib sometimes inherits alpha unexpectedly).
    """
    if fig.get_facecolor()[-1] < 1.0:
        for ax in fig.axes:
            if ax.get_facecolor()[-1] < 1.0:
                ax.set_facecolor((*ax.get_facecolor()[:3], 1.0))


# --------------------- scalar plotting ---------------------

def p_to_stars(p):
    """Convert a p-value to significance stars."""
    try:
        p = float(p)
    except Exception:
        return "n.s."
    if np.isnan(p):
        return "n.s."
    # if p < 1e-4:
    #     return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."

def _coerce_float_or_nan(x: object) -> float:
    """Best-effort float conversion; return NaN for non-numeric or non-finite values."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")

def _unique_preserve_order(items: List[str]) -> List[str]:
    """Deduplicate while preserving order."""
    seen: set[str] = set()
    out: List[str] = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out

def _resolve_p_value_from_row(row: pd.Series, *, preferred: Optional[str] = None) -> float:
    """
    Resolve a single p-value from a row using a strict priority order.

    Priority (highest -> lowest):
        preferred (if provided)
        p_across -> p_tukey -> p.value -> p_raw
        then common legacy aliases (p, pvalue, q, ...)

    Any missing / empty / non-finite values are skipped.
    """
    candidates: List[str] = []
    if preferred:
        candidates.append(str(preferred))
    candidates.extend(list(_P_VALUE_PRIORITY))
    candidates.extend(list(_P_VALUE_FALLBACK))
    candidates = _unique_preserve_order([c for c in candidates if c])

    for c in candidates:
        if c in row.index:
            v = _coerce_float_or_nan(row.get(c))
            if np.isfinite(v):
                return float(v)
    return float("nan")

def _resolve_p_value_series(df: pd.DataFrame, *, preferred: Optional[str] = None) -> pd.Series:
    """
    Resolve a per-row p-value Series from a DataFrame using a strict priority order.

    This is the vectorized counterpart of `_resolve_p_value_from_row`.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    candidates: List[str] = []
    if preferred:
        candidates.append(str(preferred))
    candidates.extend(list(_P_VALUE_PRIORITY))
    candidates.extend(list(_P_VALUE_FALLBACK))
    candidates = _unique_preserve_order([c for c in candidates if c and c in df.columns])

    p = pd.Series(np.nan, index=df.index, dtype=float)
    p_arr = p.to_numpy(copy=True)

    for c in candidates:
        v = pd.to_numeric(df[c], errors="coerce").astype(float)
        v_arr = v.to_numpy()
        fill_mask = (~np.isfinite(p_arr)) & np.isfinite(v_arr)
        if np.any(fill_mask):
            p_arr[fill_mask] = v_arr[fill_mask]

    # Final sanitize: keep only finite values
    p_arr = np.where(np.isfinite(p_arr), p_arr, np.nan)
    return pd.Series(p_arr, index=df.index, dtype=float)

def _is_ns_label(label: object) -> bool:
    """Return True if a star/label should be treated as non-significant."""
    s = str(label).strip().lower()
    return s in {"", "ns", "n.s.", "n.s", "na", "nan", "none"}

def _stars_sort_rank(label: object) -> int:
    """
    Turn a star label into an integer rank for sorting:
    smaller => more significant.
    """
    if _is_ns_label(label):
        return 10_000
    s = str(label)
    n = s.count("*")
    if n <= 0:
        return 9_999
    return 4 - min(n, 4)

def _parse_whiskers(w):
    if isinstance(w, (list, tuple)) and len(w) == 2: return (float(w[0]), float(w[1]))
    if isinstance(w, (int, float)): return float(w)
    if not isinstance(w, str): return 1.5
    s = w.strip().lower()
    if s in ("tukey", "iqr", "iqr1.5"): return 1.5
    if s == "minmax": return (0, 100)
    if s.startswith("iqr"):
        try: return float(s.replace("iqr", ""))
        except Exception: return 1.5
    if "," in s:
        try:
            lo, hi = s.split(",", 1); return (float(lo), float(hi))
        except Exception: return 1.5
    return 1.5

def _draw_tukey_brackets(
    ax,
    tuk_cell: pd.DataFrame,
    x_to_num: Dict,
    *,
    y_limits_local: Optional[Tuple[float, float]],
    hide_ns: bool,
    y_start: float,
    y_end: float,
    y_step: Optional[float],          # set to None to auto-compute
    bracket_height_frac: float,
    color: str,
    lw: float,
    text_size: int,
    label_pad_frac: float = 0.015     # small extra pad above the bracket for the text
):
    """
    Draw Tukey-style brackets with a minimal number of vertical layers (no horizontal overlap).

    Notes
    -----
    - Overlap is inclusive on endpoints (touching counts as overlap).
    - If y_step is None, the function packs all layers uniformly between y_start and y_end.
    - Coordinates y_start/y_end/bracket_height_frac/label_pad_frac are FRACTIONS of the axis y-range.
    - p-values are resolved with the following priority:
        p_across -> p_tukey -> p.value -> p_raw
      If a candidate is missing/empty/non-finite, the next one is tried.

    Parameters
    ----------
    ax : matplotlib Axes
    tuk_cell : DataFrame with at least columns {group1, group2} and any p-value column.
              'contrast' like 'A - B' is also accepted.
              If a 'stars' column exists, it is used directly.
    x_to_num : dict mapping each x-level (str) -> numeric x position on the axis
    y_limits_local : (ymin, ymax) or None to use ax.get_ylim()
    hide_ns : bool, if True drop non-significant labels ('ns', 'n.s.', empty)
    y_start, y_end : float in [0,1], bottom/top of the bracket zone, as fractions of y-range
    y_step : float or None. If None, computed automatically to fit the minimal number of layers
    bracket_height_frac : float of y-range, vertical size of the bracket (leg height)
    color, lw, text_size : styling
    label_pad_frac : extra vertical pad (in y-range fraction) between bracket and its text
    """
    if tuk_cell is None or tuk_cell.empty:
        return

    # --- normalize Tukey columns ---
    tk = tuk_cell.copy()

    # group1/group2 from contrast if needed
    if "group1" not in tk.columns or "group2" not in tk.columns:
        if "contrast" in tk.columns:
            parts = tk["contrast"].astype(str).str.split("-", n=1, expand=True)
            if parts.shape[1] == 2:
                tk["group1"] = parts[0].str.strip()
                tk["group2"] = parts[1].str.strip()

    if not {"group1", "group2"}.issubset(tk.columns):
        return

    # Resolve p-values row-wise with strict priority (may stay NaN if none found).
    tk["p.value"] = _resolve_p_value_series(tk)

    # Stars: prefer an explicit stars column; otherwise compute from p.value.
    if "stars" not in tk.columns:
        tk["stars"] = np.nan
    missing_star = tk["stars"].isna() | (tk["stars"].astype(str).str.strip() == "")
    if bool(missing_star.any()):
        tk["stars"] = tk["stars"].astype("object")
        tk.loc[missing_star, "stars"] = tk.loc[missing_star, "p.value"].apply(p_to_stars)
    # Optional hide ns
    if hide_ns:
        tk = tk[~tk["stars"].apply(_is_ns_label)]
    if tk.empty:
        return

    # Deduplicate symmetric pairs (A-B vs B-A): keep the most significant (smallest p);
    # if p is unavailable, fall back to star count.
    tk["group1"] = tk["group1"].astype(str).str.strip()
    tk["group2"] = tk["group2"].astype(str).str.strip()

    tk[["g_lo", "g_hi"]] = tk.apply(
        lambda r: pd.Series(sorted([str(r["group1"]), str(r["group2"])])),
        axis=1,
    )

    p_rank = pd.to_numeric(tk["p.value"], errors="coerce").astype(float)
    p_rank = p_rank.where(np.isfinite(p_rank), np.inf)
    tk["_p_rank"] = p_rank
    tk["_star_rank"] = tk["stars"].apply(_stars_sort_rank)

    tk = (
        tk.sort_values(["_p_rank", "_star_rank"], kind="mergesort")
          .drop_duplicates(subset=["g_lo", "g_hi"], keep="first")
    )

    # Build interval list using numeric x positions; drop pairs not on axis
    intervals = []
    for _, r in tk.iterrows():
        g1, g2 = str(r["g_lo"]), str(r["g_hi"])
        if g1 not in x_to_num or g2 not in x_to_num:
            continue
        x1, x2 = float(x_to_num[g1]), float(x_to_num[g2])
        if x1 == x2:
            continue
        lo, hi = (x1, x2) if x1 < x2 else (x2, x1)
        intervals.append(
            {
                "lo": lo,
                "hi": hi,
                "label": str(r["stars"]),
                "group1": g1,
                "group2": g2,
            }
        )
    if not intervals:
        return

    # --- Interval partitioning (touching counts as overlap) ---
    intervals.sort(key=lambda d: (d["lo"], d["hi"]))

    # Assign layers greedily with a min-heap of (end, layer_id)
    heap: List[Tuple[float, int]] = []   # (current_end, layer_id)
    next_layer_id = 0
    for d in intervals:
        lo, hi = d["lo"], d["hi"]
        # Reuse only if current_end < lo  (strict, because touching = overlap)
        if heap and heap[0][0] < lo:
            _, lid = heapq.heappop(heap)
        else:
            lid = next_layer_id
            next_layer_id += 1
        d["layer"] = lid
        heapq.heappush(heap, (hi, lid))
    n_layers = max(d["layer"] for d in intervals) + 1

    # --- y geometry ---
    if y_limits_local is None:
        y0, y1 = ax.get_ylim()
    else:
        y0, y1 = y_limits_local
    y_range = float(y1 - y0)
    if y_range <= 0:
        return

    # Bracket “tick” (vertical leg height) and text pad in data units
    tick = y_range * float(bracket_height_frac)
    text_pad = y_range * float(label_pad_frac)

    # Compute the step between layers
    top = y0 + y_range * float(y_end)
    base = y0 + y_range * float(y_start)
    usable = max(0.0, top - base)

    if y_step is None:
        # Pack all layers evenly in the available zone
        if n_layers <= 1:
            step = 0.0
        else:
            step = usable / (n_layers - 1)
        # Ensure some minimum vertical separation so text doesn't sit on the bracket below
        min_step = max(0.0, tick + text_pad * 0.8)
        if n_layers > 1 and step < min_step:
            step = min_step
    else:
        step = y_range * abs(float(y_step))

    # --- Draw brackets, top layer first (more readable) ---
    intervals.sort(key=lambda d: d["layer"])  # small layer id first (top)

    for d in intervals:
        lo, hi, lab, layer = d["lo"], d["hi"], d["label"], d["layer"]
        y = top - layer * step
        # Clamp if we're out of room
        if y < base:
            y = base

        ax.plot(
            [lo, lo, hi, hi],
            [y, y + tick, y + tick, y],
            color=color,
            lw=lw,
            zorder=5,
            clip_on=False,
        )
        ax.text(
            (lo + hi) / 2.0,
            y + tick + (-0.035 * y_range),
            lab,
            ha="center",
            va="bottom",
            fontsize=text_size,
            color=color,
            zorder=6,
            clip_on=False,
        )

def _normalize_emm_ci_columns(emw: pd.DataFrame) -> pd.DataFrame:
    """Normalize common CI column name variants to lower.CL / upper.CL."""
    if emw.empty:
        return emw
    if "lower.CL" in emw.columns and "upper.CL" in emw.columns:
        return emw
    alt = {
        "lower": "lower.CL",
        "upper": "upper.CL",
        "LCL": "lower.CL",
        "UCL": "upper.CL",
        "asymp.LCL": "lower.CL",
        "asymp.UCL": "upper.CL",
    }
    out = emw.copy()
    for old, new in alt.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})
    return out

def _auto_y_limits_scalar(
    dfw: pd.DataFrame,
    emw: pd.DataFrame,
    *,
    value_col: str,
    y_limits: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """Compute reasonable y-limits for scalar plots based on raw + EMM CI."""
    if y_limits is not None:
        return y_limits

    raw = dfw[value_col].to_numpy(dtype=float) if value_col in dfw.columns else np.array([0.0, 1.0])
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        raw = np.array([0.0, 1.0])

    ymin = float(np.min(raw))
    ymax = float(np.max(raw))

    if "lower.CL" in emw.columns:
        v = emw["lower.CL"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            ymin = min(ymin, float(np.min(v)))
    if "upper.CL" in emw.columns:
        v = emw["upper.CL"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            ymax = max(ymax, float(np.max(v)))

    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
        return (0.0, 1.0)

    yr = ymax - ymin
    return (ymin - 0.10 * yr, ymax + 0.3 * yr)

def _plot_interaction_scalar_grid(
    *,
    df: pd.DataFrame,
    emm: Optional[pd.DataFrame],
    tuk: Optional[pd.DataFrame],
    value_col: str,
    x_var: str,
    col_var: Optional[str],
    row_var: Optional[str],
    col_levels: Optional[List],
    row_levels: Optional[List],
    jitter_var: Optional[str],
    fill_var: Optional[str],
    outline_var: Optional[str],
    x_levels: Optional[List],
    # labels/scales
    x_label: Optional[str],
    single_x_label: bool,
    y_label: Optional[str],
    single_y_label: bool,
    x_limits: Optional[Tuple[float, float]],
    y_limits: Optional[Tuple[float, float]],
    x_log: bool,
    y_log: bool,
    xtick_rotation: float,
    ytick_rotation: float,
    # style
    title: Optional[str],
    font_family: str,
    jitter_palette: Union[str, List, Dict],
    fill_palette: Union[str, List, Dict],
    outline_palette: Union[str, List, Dict, None],
    jitter_width: float,
    jitter_alpha: float,
    jitter_size: float,
    grid: bool,
    grid_alpha: float,
    dpi: int,
    seed: int,
    show_top_right_axes: bool,
    # box
    show_box: bool,
    box_width: float,
    fill_alpha: float,
    whiskers: Union[str, float, Tuple[float, float]],
    box_edge_color: str,
    box_edge_width: float,
    median_color: str,
    median_linewidth: float,
    whisker_color: str,
    whisker_linewidth: float,
    cap_linewidth: float,
    outlier_marker: str,
    outlier_markersize: float,
    # overlays
    show_emm_line: bool,
    show_emm_ci: bool,
    show_raw_mean_line: bool,
    emm_line_width: float,
    emm_line_color: str,
    emm_line_style: str,
    raw_mean_line_width: float,
    raw_mean_line_color: str,
    raw_mean_line_style: str,
    error_bar_linewidth: float,
    error_bar_cap: float,
    error_bar_color: str,
    # refs
    vertical_lines: Optional[Union[List, np.ndarray]],
    vline_color: str,
    vline_style: str,
    vline_width: float,
    vline_alpha: float,
    horizontal_lines: Optional[Union[List, np.ndarray]],
    hline_color: str,
    hline_style: str,
    hline_width: float,
    hline_alpha: float,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]],
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]],
    # strips
    label_fontsize: int,
    label_top_bg_color: str,
    label_right_bg_color: str,
    label_text_color: str,
    label_fontweight: str,
    strip_top_height_mm: float,
    strip_right_width_mm: float,
    strip_pad_mm: float,
    # text sizes
    title_fontsize: int,
    axis_label_fontsize: int,
    tick_label_fontsize: int,
    # legend
    legend_loc: str,
    legend_ncol: Optional[int],
    legend_framealpha: float,
    legend_fontsize: int,
    # geometry (mm)
    boxsize: Tuple[float, float],
    panel_gap: Tuple[float, float],
    include_global_label_margins: bool,
    x_label_offset_mm: float,
    y_label_offset_mm: float,
    # brackets
    show_brackets: bool,
    hide_ns: bool,
    y_start: float,
    y_end: float,
    y_step: Optional[float],
    bracket_height_frac: float,
    bracket_color: str,
    bracket_linewidth: float,
    bracket_text_size: int,
    # transparent
    transparent: bool,
) -> plt.Figure:
    """Core scalar grid plotter (supports 1D/2D faceting)."""
    rng = np.random.default_rng(seed)
    plt.rcParams["font.family"] = font_family

    dfw = df.copy()
    emw_in = emm if isinstance(emm, pd.DataFrame) else pd.DataFrame()
    tkw_in = tuk if isinstance(tuk, pd.DataFrame) else pd.DataFrame()
    emw = _normalize_emm_ci_columns(emw_in.copy())
    tkw = tkw_in.copy()

    # Defaults for roles
    if jitter_var is None:
        jitter_var = x_var
    if fill_var is None:
        fill_var = x_var

    # Levels
    if x_levels is None:
        x_levels = _ordered_levels(dfw[x_var], None)
    if col_var is not None:
        if col_levels is None:
            col_levels = _ordered_levels(dfw[col_var], None)
    else:
        col_levels = [None]
    if row_var is not None:
        if row_levels is None:
            row_levels = _ordered_levels(dfw[row_var], None)
    else:
        row_levels = [None]

    # Enforce categorical ordering
    if x_var in dfw.columns:
        dfw[x_var] = pd.Categorical(dfw[x_var], categories=x_levels, ordered=True)
    if x_var in emw.columns:
        emw[x_var] = pd.Categorical(emw[x_var], categories=x_levels, ordered=True)
    if col_var and col_var in dfw.columns:
        dfw[col_var] = pd.Categorical(dfw[col_var], categories=col_levels, ordered=True)
    if col_var and col_var in emw.columns:
        emw[col_var] = pd.Categorical(emw[col_var], categories=col_levels, ordered=True)
    if row_var and row_var in dfw.columns:
        dfw[row_var] = pd.Categorical(dfw[row_var], categories=row_levels, ordered=True)
    if row_var and row_var in emw.columns:
        emw[row_var] = pd.Categorical(emw[row_var], categories=row_levels, ordered=True)

    # Color maps
    jitter_levels = _ordered_levels(dfw[jitter_var], None) if jitter_var in dfw.columns else x_levels
    fill_levels = _ordered_levels(dfw[fill_var], None) if fill_var in dfw.columns else x_levels

    outline_levels: List = []
    if outline_var is not None and outline_var in dfw.columns:
        outline_levels = _ordered_levels(dfw[outline_var], None)

    jitter_cmap = _build_color_map(jitter_levels, jitter_palette)
    fill_cmap = _build_color_map(fill_levels, fill_palette)
    outline_cmap = _build_color_map(outline_levels, outline_palette) if (outline_palette is not None and outline_levels) else {}

    x_to_num = {str(lv): i for i, lv in enumerate(x_levels)}

    y_limits_use = _auto_y_limits_scalar(dfw, emw, value_col=value_col, y_limits=y_limits)

    nrows = len(row_levels)
    ncols = len(col_levels)

    # Layout: strips only if the dimension exists
    top_h = strip_top_height_mm if (col_var is not None and ncols > 1) else 0.0
    right_w = strip_right_width_mm if (row_var is not None and nrows > 1) else 0.0

    fig, axes, layout = _init_box_figure(
        nrows=nrows,
        ncols=ncols,
        boxsize_mm=boxsize,
        panel_gap_mm=panel_gap,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        colorbar_width_mm=0.0,
        colorbar_pad_mm=0.0,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=(x_label or x_var),
        y_label_text=(y_label or value_col),
        colorbar_label_text=None,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=0.0,
        dpi=dpi,
        font_family=font_family,
        sharex=True,
        sharey=True,
        transparent=transparent,
    )

    # --------------------- plotting ---------------------
    for i, row_lv in enumerate(row_levels):
        for j, col_lv in enumerate(col_levels):
            ax = axes[i, j]

            # background shadows
            if vertical_shadows:
                for (x0, x1), c in vertical_shadows.items():
                    ax.axvspan(x0, x1, color=_rgba_color(c), zorder=0)
            if horizontal_shadows:
                for (y0, y1), c in horizontal_shadows.items():
                    ax.axhspan(y0, y1, color=_rgba_color(c), zorder=0)

            cell_raw = dfw
            if col_var is not None and col_lv is not None:
                cell_raw = cell_raw[cell_raw[col_var] == col_lv]
            if row_var is not None and row_lv is not None:
                cell_raw = cell_raw[cell_raw[row_var] == row_lv]

            cell_emm = emw
            if col_var is not None and col_lv is not None and col_var in cell_emm.columns:
                cell_emm = cell_emm[cell_emm[col_var] == col_lv]
            if row_var is not None and row_lv is not None and row_var in cell_emm.columns:
                cell_emm = cell_emm[cell_emm[row_var] == row_lv]

            # Boxplot
            if show_box:
                data_for_boxes: List[np.ndarray] = []
                pos_for_boxes: List[float] = []
                box_levels_used: List = []

                for lv in x_levels:
                    sub = cell_raw[cell_raw[x_var] == lv] if x_var in cell_raw.columns else cell_raw.iloc[0:0]
                    vals = sub[value_col].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        data_for_boxes.append(vals)
                        pos_for_boxes.append(float(x_to_num[str(lv)]))
                        box_levels_used.append(lv)

                if data_for_boxes:
                    whis = _parse_whiskers(whiskers)
                    bp = ax.boxplot(
                        data_for_boxes,
                        positions=pos_for_boxes,
                        widths=box_width,
                        patch_artist=True,
                        showfliers=True,
                        whis=whis,
                        medianprops=dict(color=median_color, linewidth=median_linewidth),
                        whiskerprops=dict(color=whisker_color, linewidth=whisker_linewidth),
                        capprops=dict(color=whisker_color, linewidth=cap_linewidth),
                        flierprops=dict(
                            marker=outlier_marker,
                            markersize=outlier_markersize,
                            markerfacecolor=whisker_color,
                            markeredgecolor=whisker_color,
                            alpha=0.7,
                        ),
                    )

                    use_outline_for_lines = (outline_var is not None and outline_var in cell_raw.columns)

                    for k, b in enumerate(bp["boxes"]):
                        lv = box_levels_used[k] if k < len(box_levels_used) else None

                        # fill by fill_var (typically x_var)
                        if fill_var in cell_raw.columns and lv is not None:
                            sub_x = cell_raw[cell_raw[x_var] == lv]
                            fill_level = sub_x[fill_var].iloc[0] if not sub_x.empty else lv
                        else:
                            fill_level = lv
                        face = fill_cmap.get(fill_level, "gray")
                        b.set_facecolor(mpl.colors.to_rgba(face, fill_alpha))

                        # outline by outline_var (optional)
                        # IMPORTANT: when outline_var is provided, it drives the color of:
                        #   - box outline
                        #   - median line
                        #   - whiskers + caps + fliers
                        outline_c = box_edge_color
                        if use_outline_for_lines and lv is not None:
                            sub_x = cell_raw[cell_raw[x_var] == lv]
                            if not sub_x.empty:
                                edge_level = sub_x[outline_var].iloc[0]
                                outline_c = outline_cmap.get(edge_level, box_edge_color)

                        b.set_edgecolor(outline_c)
                        b.set_linewidth(box_edge_width)

                        if use_outline_for_lines:
                            # median is 1 per box
                            if k < len(bp.get("medians", [])):
                                bp["medians"][k].set_color(outline_c)
                                bp["medians"][k].set_linewidth(median_linewidth)

                            # whiskers/caps are 2 per box (low/high)
                            for idx in (2 * k, 2 * k + 1):
                                if idx < len(bp.get("whiskers", [])):
                                    bp["whiskers"][idx].set_color(outline_c)
                                    bp["whiskers"][idx].set_linewidth(whisker_linewidth)
                                if idx < len(bp.get("caps", [])):
                                    bp["caps"][idx].set_color(outline_c)
                                    bp["caps"][idx].set_linewidth(cap_linewidth)

                            # fliers are 1 per box
                            if k < len(bp.get("fliers", [])):
                                fl = bp["fliers"][k]
                                fl.set_markerfacecolor(outline_c)
                                fl.set_markeredgecolor(outline_c)

            # Jitter points
            if not cell_raw.empty:
                xs = cell_raw[x_var].astype(str).map(x_to_num).to_numpy(dtype=float)
                xs_j = xs + (rng.random(xs.shape[0]) - 0.5) * jitter_width

                if jitter_var in cell_raw.columns:
                    j_levels = cell_raw[jitter_var].tolist()
                    cols = [jitter_cmap.get(lv, "gray") for lv in j_levels]
                else:
                    cols = "gray"

                ax.scatter(
                    xs_j,
                    cell_raw[value_col].to_numpy(dtype=float),
                    c=cols,
                    alpha=jitter_alpha,
                    s=jitter_size,
                    linewidths=0,
                    zorder=2.5,
                )

            # EMM CI + mean line
            if (show_emm_ci or show_emm_line) and not cell_emm.empty and "emmean" in cell_emm.columns:
                # order by x_levels
                cell_emm = cell_emm.copy()
                cell_emm["_x_str"] = cell_emm[x_var].astype(str)
                cell_emm["_x_num"] = cell_emm["_x_str"].map(x_to_num)
                cell_emm = cell_emm.dropna(subset=["_x_num"]).sort_values("_x_num")
                x_sorted = cell_emm["_x_num"].to_numpy(dtype=float)
                emmean = cell_emm["emmean"].to_numpy(dtype=float)

                if show_emm_ci and {"lower.CL", "upper.CL"}.issubset(cell_emm.columns):
                    lower = cell_emm["lower.CL"].to_numpy(dtype=float)
                    upper = cell_emm["upper.CL"].to_numpy(dtype=float)
                    ax.errorbar(
                        x_sorted,
                        emmean,
                        yerr=[emmean - lower, upper - emmean],
                        fmt="o",
                        ms=0,
                        elinewidth=error_bar_linewidth,
                        ecolor=error_bar_color,
                        capsize=error_bar_cap,
                        zorder=3.0,
                    )

                if show_emm_line:
                    ax.plot(
                        x_sorted,
                        emmean,
                        color=emm_line_color,
                        lw=emm_line_width,
                        ls=emm_line_style,
                        zorder=3.1,
                        label="EMM",
                    )

            # Raw mean line
            if show_raw_mean_line and not cell_raw.empty:
                grp = cell_raw.groupby(x_var, observed=True)[value_col].mean().reindex(x_levels)
                x_pos = np.array([x_to_num[str(k)] for k in grp.index], dtype=float)
                ax.plot(
                    x_pos,
                    grp.to_numpy(dtype=float),
                    color=raw_mean_line_color,
                    lw=raw_mean_line_width,
                    ls=raw_mean_line_style,
                    zorder=2.9,
                    label="Raw mean",
                )

            # references
            if vertical_lines is not None:
                for xv in vertical_lines:
                    ax.axvline(
                        x=xv,
                        color=vline_color,
                        linestyle=vline_style,
                        linewidth=vline_width,
                        alpha=vline_alpha,
                        zorder=4,
                    )
            if horizontal_lines is not None:
                for yv in horizontal_lines:
                    ax.axhline(
                        y=yv,
                        color=hline_color,
                        linestyle=hline_style,
                        linewidth=hline_width,
                        alpha=hline_alpha,
                        zorder=4,
                    )

            # axis cosmetics
            if y_limits_use is not None:
                ax.set_ylim(*y_limits_use)
            if y_log:
                ax.set_yscale("log")
            if x_limits is not None:
                ax.set_xlim(*x_limits)
            if x_log:
                ax.set_xscale("log")

            ax.set_xlim(-0.5, len(x_levels) - 0.5)
            ax.set_xticks(range(len(x_levels)))
            ax.set_xticklabels([str(x) for x in x_levels], fontsize=tick_label_fontsize, rotation=xtick_rotation)
            for t in ax.get_yticklabels():
                t.set_fontsize(tick_label_fontsize)
                t.set_rotation(ytick_rotation)
            ax.tick_params(labelsize=tick_label_fontsize)

            ax.xaxis.get_offset_text().set_fontsize(tick_label_fontsize)
            ax.yaxis.get_offset_text().set_fontsize(tick_label_fontsize)

            if grid:
                ax.grid(True, alpha=grid_alpha, linestyle="--", zorder=0)

            ax.spines["top"].set_visible(show_top_right_axes)
            ax.spines["right"].set_visible(show_top_right_axes)

            if (i == nrows - 1) and (not single_x_label):
                ax.set_xlabel(x_label or x_var, fontsize=axis_label_fontsize)
            if (j == 0) and (not single_y_label):
                ax.set_ylabel(y_label or value_col, fontsize=axis_label_fontsize)

            # Tukey brackets
            if show_brackets:
                cell_tk = tkw
                if col_var is not None and col_lv is not None and col_var in cell_tk.columns:
                    cell_tk = cell_tk[cell_tk[col_var] == col_lv]
                if row_var is not None and row_lv is not None and row_var in cell_tk.columns:
                    cell_tk = cell_tk[cell_tk[row_var] == row_lv]

                _draw_tukey_brackets(
                    ax,
                    cell_tk,
                    x_to_num,
                    y_limits_local=y_limits_use,
                    hide_ns=hide_ns,
                    y_start=y_start,
                    y_end=y_end,
                    y_step=y_step,
                    bracket_height_frac=bracket_height_frac,
                    color=bracket_color,
                    lw=bracket_linewidth,
                    text_size=bracket_text_size,
                )

    # strips
    col_labs = [f"{lv}" for lv in col_levels] if (col_var is not None and ncols > 1) else None
    row_labs = [f"{lv}" for lv in row_levels] if (row_var is not None and nrows > 1) else None
    _ = _add_strips_mm(
        fig,
        axes,
        col_labels=col_labs,
        row_labels=row_labs,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
    )

    if transparent:
        _ensure_strip_background_opaque(fig)

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=(x_label or x_var),
        y_label=(y_label or value_col),
        axis_label_fontsize=axis_label_fontsize,
        title=title,
        title_fontsize=title_fontsize,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        font_family=font_family,
    )

    # legend
    if legend_loc != "none":
        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=mpl.colors.to_rgba("gray", fill_alpha),
                markeredgecolor=box_edge_color,
                label="Box",
            ),
        ]
        if show_emm_line:
            handles.append(Line2D([0], [0], color=emm_line_color, lw=emm_line_width, ls=emm_line_style, label="EMM"))
        if show_raw_mean_line:
            handles.append(
                Line2D([0], [0], color=raw_mean_line_color, lw=raw_mean_line_width, ls=raw_mean_line_style, label="Raw mean")
            )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="gray",
                alpha=jitter_alpha,
                markersize=float(np.sqrt(jitter_size)),
                label=f"Jitter ({jitter_var})",
            )
        )

        if legend_ncol is None:
            legend_ncol = 1

        _place_legend(
            fig,
            axes[0, 0],
            handles,
            [h.get_label() for h in handles],
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_ncol=legend_ncol,
            legend_framealpha=legend_framealpha,
            inside_map=not _is_outside_legend_loc(legend_loc),
        )

    return fig

def plot_single_effect_scalar(
    df: pd.DataFrame,
    emm: Optional[pd.DataFrame] = None,
    tuk: Optional[pd.DataFrame] = None,
    value_col: str = "Value",
    x_var: str = "Phase",
    jitter_var: Optional[str] = None,
    fill_var: Optional[str] = None,
    outline_var: Optional[str] = None,
    x_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    xtick_rotation: float = 0.0,
    ytick_rotation: float = 0.0,
    title: Optional[str] = None,
    font_family: str = "Arial",
    jitter_palette: Union[str, List, Dict] = "viridis",
    fill_palette: Union[str, List, Dict] = "viridis",
    outline_palette: Union[str, List, Dict, None] = None,
    jitter_width: float = 0.18,
    jitter_alpha: float = 0.35,
    jitter_size: float = 12.0,
    grid: bool = False,
    grid_alpha: float = 0.3,
    dpi: int = 300,
    seed: int = 1,
    show_top_right_axes: bool = True,
    show_box: bool = True,
    box_width: float = 0.55,
    fill_alpha: float = 0.40,
    whiskers: Union[str, float, Tuple[float, float]] = "tukey",
    box_edge_color: str = "black",
    box_edge_width: float = 1.0,
    median_color: str = "black",
    median_linewidth: float = 1.0,
    whisker_color: str = "black",
    whisker_linewidth: float = 1.0,
    cap_linewidth: float = 1.0,
    outlier_marker: str = "o",
    outlier_markersize: float = 3.0,
    show_emm_line: bool = True,
    show_emm_ci: bool = True,
    show_raw_mean_line: bool = True,
    emm_line_width: float = 1.0,
    emm_line_color: str = "black",
    emm_line_style: str = "-",
    raw_mean_line_width: float = 1.0,
    raw_mean_line_color: str = "#404040",
    raw_mean_line_style: str = "--",
    error_bar_linewidth: float = 1.0,
    error_bar_cap: float = 3.0,
    error_bar_color: str = "black",
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    legend_loc: str = "none",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 10,
    y_label_offset_mm: float = 18,
    show_brackets: bool = True,
    hide_ns: bool = True,
    y_start: float = 0.70,
    y_end: float = 0.95,
    y_step: Optional[float] = 0.08,
    bracket_height_frac: float = 0.018,
    bracket_color: str = "black",
    bracket_linewidth: float = 1.2,
    bracket_text_size: int = 12,
    transparent: bool = False,
) -> plt.Figure:
    """Single effect scalar plot (no faceting).

    `emm` and `tuk` can be None or empty DataFrames.
    """
    return _plot_interaction_scalar_grid(
        df=df,
        emm=emm,
        tuk=tuk,
        value_col=value_col,
        x_var=x_var,
        col_var=None,
        row_var=None,
        col_levels=None,
        row_levels=None,
        jitter_var=jitter_var,
        fill_var=fill_var,
        outline_var=outline_var,
        x_levels=x_levels,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        xtick_rotation=xtick_rotation,
        ytick_rotation=ytick_rotation,
        title=title,
        font_family=font_family,
        jitter_palette=jitter_palette,
        fill_palette=fill_palette,
        outline_palette=outline_palette,
        jitter_width=jitter_width,
        jitter_alpha=jitter_alpha,
        jitter_size=jitter_size,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        seed=seed,
        show_top_right_axes=show_top_right_axes,
        show_box=show_box,
        box_width=box_width,
        fill_alpha=fill_alpha,
        whiskers=whiskers,
        box_edge_color=box_edge_color,
        box_edge_width=box_edge_width,
        median_color=median_color,
        median_linewidth=median_linewidth,
        whisker_color=whisker_color,
        whisker_linewidth=whisker_linewidth,
        cap_linewidth=cap_linewidth,
        outlier_marker=outlier_marker,
        outlier_markersize=outlier_markersize,
        show_emm_line=show_emm_line,
        show_emm_ci=show_emm_ci,
        show_raw_mean_line=show_raw_mean_line,
        emm_line_width=emm_line_width,
        emm_line_color=emm_line_color,
        emm_line_style=emm_line_style,
        raw_mean_line_width=raw_mean_line_width,
        raw_mean_line_color=raw_mean_line_color,
        raw_mean_line_style=raw_mean_line_style,
        error_bar_linewidth=error_bar_linewidth,
        error_bar_cap=error_bar_cap,
        error_bar_color=error_bar_color,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=16,
        label_top_bg_color="lightgray",
        label_right_bg_color="lightgray",
        label_text_color="black",
        label_fontweight="normal",
        strip_top_height_mm=0.0,
        strip_right_width_mm=0.0,
        strip_pad_mm=0.0,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=(0.0, 0.0),
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        show_brackets=show_brackets,
        hide_ns=hide_ns,
        y_start=y_start,
        y_end=y_end,
        y_step=y_step,
        bracket_height_frac=bracket_height_frac,
        bracket_color=bracket_color,
        bracket_linewidth=bracket_linewidth,
        bracket_text_size=bracket_text_size,
        transparent=transparent,
    )


def plot_triple_interaction_scalar(
    df: pd.DataFrame,
    emm: pd.DataFrame,
    tuk: pd.DataFrame,
    value_col: str = "value",
    x_var: str = "phase",
    panel_var: str = "region",
    facet_var: str = "lat",
    jitter_var: Optional[str] = None,
    fill_var: Optional[str] = None,
    outline_var: Optional[str] = None,
    x_levels: Optional[List] = None,
    panel_levels: Optional[List] = None,
    facet_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    xtick_rotation: float = 0.0,
    ytick_rotation: float = 0.0,
    title: Optional[str] = None,
    font_family: str = "Arial",
    jitter_palette: Union[str, List, Dict] = "viridis",
    fill_palette: Union[str, List, Dict] = "viridis",
    outline_palette: Union[str, List, Dict, None] = None,
    jitter_width: float = 0.18,
    jitter_alpha: float = 0.35,
    jitter_size: float = 12.0,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    seed: int = 1,
    show_top_right_axes: bool = True,
    show_box: bool = True,
    box_width: float = 0.55,
    fill_alpha: float = 0.40,
    whiskers: Union[str, float, Tuple[float, float]] = "tukey",
    box_edge_color: str = "black",
    box_edge_width: float = 1.0,
    median_color: str = "black",
    median_linewidth: float = 1.0,
    whisker_color: str = "black",
    whisker_linewidth: float = 1.0,
    cap_linewidth: float = 1.0,
    outlier_marker: str = "o",
    outlier_markersize: float = 3.0,
    show_emm_line: bool = True,
    show_emm_ci: bool = True,
    show_raw_mean_line: bool = True,
    emm_line_width: float = 1.0,
    emm_line_color: str = "black",
    emm_line_style: str = "-",
    raw_mean_line_width: float = 1.0,
    raw_mean_line_color: str = "#404040",
    raw_mean_line_style: str = "--",
    error_bar_linewidth: float = 1.0,
    error_bar_cap: float = 3.0,
    error_bar_color: str = "black",
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 16,
    label_top_bg_color: str = "lightgray",
    label_right_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_right_width_mm: float = 3.0,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    legend_loc: str = "none",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (0.0, 0.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    show_brackets: bool = True,
    hide_ns: bool = True,
    y_start: float = 0.70,
    y_end: float = 0.95,
    y_step: Optional[float] = 0.08,
    bracket_height_frac: float = 0.018,
    bracket_color: str = "black",
    bracket_linewidth: float = 1.2,
    bracket_text_size: int = 12,
    transparent: bool = False,
) -> plt.Figure:
    """Triple interaction scalar plot (panel × facet)."""
    return _plot_interaction_scalar_grid(
        df=df,
        emm=emm,
        tuk=tuk,
        value_col=value_col,
        x_var=x_var,
        col_var=panel_var,
        row_var=facet_var,
        col_levels=panel_levels,
        row_levels=facet_levels,
        jitter_var=jitter_var,
        fill_var=fill_var,
        outline_var=outline_var,
        x_levels=x_levels,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        xtick_rotation=xtick_rotation,
        ytick_rotation=ytick_rotation,
        title=title,
        font_family=font_family,
        jitter_palette=jitter_palette,
        fill_palette=fill_palette,
        outline_palette=outline_palette,
        jitter_width=jitter_width,
        jitter_alpha=jitter_alpha,
        jitter_size=jitter_size,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        seed=seed,
        show_top_right_axes=show_top_right_axes,
        show_box=show_box,
        box_width=box_width,
        fill_alpha=fill_alpha,
        whiskers=whiskers,
        box_edge_color=box_edge_color,
        box_edge_width=box_edge_width,
        median_color=median_color,
        median_linewidth=median_linewidth,
        whisker_color=whisker_color,
        whisker_linewidth=whisker_linewidth,
        cap_linewidth=cap_linewidth,
        outlier_marker=outlier_marker,
        outlier_markersize=outlier_markersize,
        show_emm_line=show_emm_line,
        show_emm_ci=show_emm_ci,
        show_raw_mean_line=show_raw_mean_line,
        emm_line_width=emm_line_width,
        emm_line_color=emm_line_color,
        emm_line_style=emm_line_style,
        raw_mean_line_width=raw_mean_line_width,
        raw_mean_line_color=raw_mean_line_color,
        raw_mean_line_style=raw_mean_line_style,
        error_bar_linewidth=error_bar_linewidth,
        error_bar_cap=error_bar_cap,
        error_bar_color=error_bar_color,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=strip_right_width_mm,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        show_brackets=show_brackets,
        hide_ns=hide_ns,
        y_start=y_start,
        y_end=y_end,
        y_step=y_step,
        bracket_height_frac=bracket_height_frac,
        bracket_color=bracket_color,
        bracket_linewidth=bracket_linewidth,
        bracket_text_size=bracket_text_size,
        transparent=transparent,
    )


def plot_double_interaction_scalar(
    df: pd.DataFrame,
    emm: pd.DataFrame,
    tuk: pd.DataFrame,
    value_col: str = "value",
    x_var: str = "phase",
    panel_var: str = "region",
    jitter_var: Optional[str] = None,
    fill_var: Optional[str] = None,
    outline_var: Optional[str] = None,
    x_levels: Optional[List] = None,
    panel_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    xtick_rotation: float = 0.0,
    ytick_rotation: float = 0.0,
    title: Optional[str] = None,
    font_family: str = "Arial",
    jitter_palette: Union[str, List, Dict] = "viridis",
    fill_palette: Union[str, List, Dict] = "viridis",
    outline_palette: Union[str, List, Dict, None] = None,
    jitter_width: float = 0.18,
    jitter_alpha: float = 0.35,
    jitter_size: float = 12.0,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    seed: int = 1,
    show_top_right_axes: bool = True,
    show_box: bool = True,
    box_width: float = 0.55,
    fill_alpha: float = 0.40,
    whiskers: Union[str, float, Tuple[float, float]] = "tukey",
    box_edge_color: str = "black",
    box_edge_width: float = 1.0,
    median_color: str = "black",
    median_linewidth: float = 1.0,
    whisker_color: str = "black",
    whisker_linewidth: float = 1.0,
    cap_linewidth: float = 1.0,
    outlier_marker: str = "o",
    outlier_markersize: float = 3.0,
    show_emm_line: bool = True,
    show_emm_ci: bool = True,
    show_raw_mean_line: bool = True,
    emm_line_width: float = 1.0,
    emm_line_color: str = "black",
    emm_line_style: str = "-",
    raw_mean_line_width: float = 1.0,
    raw_mean_line_color: str = "#404040",
    raw_mean_line_style: str = "--",
    error_bar_linewidth: float = 1.0,
    error_bar_cap: float = 3.0,
    error_bar_color: str = "black",
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 16,
    label_top_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    legend_loc: str = "none",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (0.0, 0.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    show_brackets: bool = True,
    hide_ns: bool = True,
    y_start: float = 0.70,
    y_end: float = 0.95,
    y_step: Optional[float] = 0.08,
    bracket_height_frac: float = 0.018,
    bracket_color: str = "black",
    bracket_linewidth: float = 1.2,
    bracket_text_size: int = 12,
    transparent: bool = False,
) -> plt.Figure:
    """Double interaction scalar plot (panel only)."""
    return _plot_interaction_scalar_grid(
        df=df,
        emm=emm,
        tuk=tuk,
        value_col=value_col,
        x_var=x_var,
        col_var=panel_var,
        row_var=None,
        col_levels=panel_levels,
        row_levels=None,
        jitter_var=jitter_var,
        fill_var=fill_var,
        outline_var=outline_var,
        x_levels=x_levels,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        xtick_rotation=xtick_rotation,
        ytick_rotation=ytick_rotation,
        title=title,
        font_family=font_family,
        jitter_palette=jitter_palette,
        fill_palette=fill_palette,
        outline_palette=outline_palette,
        jitter_width=jitter_width,
        jitter_alpha=jitter_alpha,
        jitter_size=jitter_size,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        seed=seed,
        show_top_right_axes=show_top_right_axes,
        show_box=show_box,
        box_width=box_width,
        fill_alpha=fill_alpha,
        whiskers=whiskers,
        box_edge_color=box_edge_color,
        box_edge_width=box_edge_width,
        median_color=median_color,
        median_linewidth=median_linewidth,
        whisker_color=whisker_color,
        whisker_linewidth=whisker_linewidth,
        cap_linewidth=cap_linewidth,
        outlier_marker=outlier_marker,
        outlier_markersize=outlier_markersize,
        show_emm_line=show_emm_line,
        show_emm_ci=show_emm_ci,
        show_raw_mean_line=show_raw_mean_line,
        emm_line_width=emm_line_width,
        emm_line_color=emm_line_color,
        emm_line_style=emm_line_style,
        raw_mean_line_width=raw_mean_line_width,
        raw_mean_line_color=raw_mean_line_color,
        raw_mean_line_style=raw_mean_line_style,
        error_bar_linewidth=error_bar_linewidth,
        error_bar_cap=error_bar_cap,
        error_bar_color=error_bar_color,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_top_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=0.0,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        show_brackets=show_brackets,
        hide_ns=hide_ns,
        y_start=y_start,
        y_end=y_end,
        y_step=y_step,
        bracket_height_frac=bracket_height_frac,
        bracket_color=bracket_color,
        bracket_linewidth=bracket_linewidth,
        bracket_text_size=bracket_text_size,
        transparent=transparent,
    )

def _auto_limits_series(
    dfw: pd.DataFrame,
    *,
    value_col: str,
    y_limits: Optional[Tuple[float, float]],
    x_limits: Optional[Tuple[float, float]],
    line_var: str,
    col_var: Optional[str] = None,
    row_var: Optional[str] = None,
    line_levels: Optional[List] = None,
    col_levels: Optional[List] = None,
    row_levels: Optional[List] = None,
    ribbon: str = "sem",
    include_ribbon: bool = True,
    n_grid: int = 400,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Auto-compute x/y limits for series plots based on what is actually rendered.

    The limits are computed from:
      - the mean curve (per `line_var`, per facet cell)
      - the ribbon envelope (low/high; e.g., SEM / SD / 95% CI) when `include_ribbon` is True

    This avoids a single outlier trace inside `dfw[value_col]` blowing up the axis range
    when the visualization is meant to emphasize the group mean ± uncertainty.
    """
    if x_limits is not None and y_limits is not None:
        return x_limits, y_limits

    # Infer levels if not provided (must match plotting logic).
    if line_levels is None:
        if line_var in dfw.columns:
            line_levels = _ordered_levels(dfw[line_var], None)
        else:
            line_levels = []

    if col_var is None:
        col_levels_use: List = [None]
    else:
        col_levels_use = list(col_levels) if col_levels is not None else _ordered_levels(dfw[col_var], None)

    if row_var is None:
        row_levels_use: List = [None]
    else:
        row_levels_use = list(row_levels) if row_levels is not None else _ordered_levels(dfw[row_var], None)

    xmins: List[float] = []
    xmaxs: List[float] = []
    ymins: List[float] = []
    ymaxs: List[float] = []

    # Prefer limits based on the rendered mean/ribbon curves.
    for row_lv in row_levels_use:
        for col_lv in col_levels_use:
            cell = dfw
            if col_var is not None and col_lv is not None and col_var in cell.columns:
                cell = cell[cell[col_var] == col_lv]
            if row_var is not None and row_lv is not None and row_var in cell.columns:
                cell = cell[cell[row_var] == row_lv]
            if cell.empty:
                continue

            for lv in line_levels:
                if line_var in cell.columns:
                    sub = cell[cell[line_var] == lv]
                else:
                    sub = cell

                series_list = [s for s in sub[value_col].tolist() if isinstance(s, pd.Series) and not s.empty]
                xg, mean, low, high = _series_grid_stats(series_list, n_grid=n_grid, ribbon=ribbon)
                if xg.size == 0:
                    continue

                xg_f = xg[np.isfinite(xg)]
                if xg_f.size:
                    xmins.append(float(np.min(xg_f)))
                    xmaxs.append(float(np.max(xg_f)))

                if include_ribbon:
                    y_cat = np.concatenate([mean, low, high])
                else:
                    y_cat = mean
                y_f = y_cat[np.isfinite(y_cat)]
                if y_f.size:
                    ymins.append(float(np.min(y_f)))
                    ymaxs.append(float(np.max(y_f)))

    # Fallback: if nothing was rendered (e.g., all series empty), fall back to raw series.
    if (not xmins or not xmaxs) or (not ymins or not ymaxs):
        for s in dfw[value_col]:
            if not isinstance(s, pd.Series) or s.empty:
                continue
            x = _to_numeric_index(s.index)
            y = s.to_numpy(dtype=float)
            y = y[np.isfinite(y)]
            x = x[np.isfinite(x)]
            if x.size:
                xmins.append(float(np.min(x)))
                xmaxs.append(float(np.max(x)))
            if y.size:
                ymins.append(float(np.min(y)))
                ymaxs.append(float(np.max(y)))

    if x_limits is None:
        if xmins and xmaxs:
            x0, x1 = float(np.min(xmins)), float(np.max(xmaxs))
        else:
            x0, x1 = 0.0, 1.0
        xr = x1 - x0
        if (not np.isfinite(xr)) or xr == 0:
            xr = 1.0
        x_limits = (x0 - 0.02 * xr, x1 + 0.02 * xr)

    if y_limits is None:
        if ymins and ymaxs:
            y0, y1 = float(np.min(ymins)), float(np.max(ymaxs))
        else:
            y0, y1 = 0.0, 1.0
        yr = y1 - y0
        if (not np.isfinite(yr)) or yr == 0:
            yr = 1.0
        y_limits = (y0 - 0.05 * yr, y1 + 0.05 * yr)

    return x_limits, y_limits


def _plot_interaction_series_grid(
    *,
    df: pd.DataFrame,
    value_col: str,
    line_var: str,
    col_var: Optional[str],
    row_var: Optional[str],
    line_levels: Optional[List],
    col_levels: Optional[List],
    row_levels: Optional[List],
    x_label: Optional[str],
    single_x_label: bool,
    y_label: Optional[str],
    single_y_label: bool,
    x_limits: Optional[Tuple[float, float]],
    y_limits: Optional[Tuple[float, float]],
    y_ticks: Optional[Union[List[float], np.ndarray]],
    y_tick_labels: Optional[List[str]],
    x_log: bool,
    y_log: bool,
    title: Optional[str],
    font_family: str,
    line_palette: Union[str, List, Dict],
    line_width: float,
    line_alpha: float,
    show_ribbon: bool,
    ribbon_alpha: float,
    ribbon: str,
    grid: bool,
    grid_alpha: float,
    dpi: int,
    show_top_right_axes: bool,
    vertical_lines: Optional[Union[List, np.ndarray]],
    vline_color: str,
    vline_style: str,
    vline_width: float,
    vline_alpha: float,
    horizontal_lines: Optional[Union[List, np.ndarray]],
    hline_color: str,
    hline_style: str,
    hline_width: float,
    hline_alpha: float,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]],
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]],
    # strips
    label_fontsize: int,
    label_top_bg_color: str,
    label_right_bg_color: str,
    label_text_color: str,
    label_fontweight: str,
    strip_top_height_mm: float,
    strip_right_width_mm: float,
    strip_pad_mm: float,
    # text sizes
    title_fontsize: int,
    axis_label_fontsize: int,
    tick_label_fontsize: int,
    # legend
    legend_loc: str,
    legend_ncol: Optional[int],
    legend_framealpha: float,
    legend_fontsize: int,
    # geometry
    boxsize: Tuple[float, float],
    panel_gap: Tuple[float, float],
    include_global_label_margins: bool,
    x_label_offset_mm: float,
    y_label_offset_mm: float,
    transparent: bool,
) -> plt.Figure:
    """Core series grid plotter."""
    plt.rcParams["font.family"] = font_family
    dfw = df.copy()

    # Levels
    if line_levels is None:
        line_levels = _ordered_levels(dfw[line_var], None)
    if col_var is not None:
        if col_levels is None:
            col_levels = _ordered_levels(dfw[col_var], None)
    else:
        col_levels = [None]
    if row_var is not None:
        if row_levels is None:
            row_levels = _ordered_levels(dfw[row_var], None)
    else:
        row_levels = [None]

    if line_var in dfw.columns:
        dfw[line_var] = pd.Categorical(dfw[line_var], categories=line_levels, ordered=True)
    if col_var and col_var in dfw.columns:
        dfw[col_var] = pd.Categorical(dfw[col_var], categories=col_levels, ordered=True)
    if row_var and row_var in dfw.columns:
        dfw[row_var] = pd.Categorical(dfw[row_var], categories=row_levels, ordered=True)

    line_cmap = _build_color_map(line_levels, line_palette)

    x_limits_use, y_limits_use = _auto_limits_series(
        dfw,
        value_col=value_col,
        x_limits=x_limits,
        y_limits=y_limits,
        line_var=line_var,
        col_var=col_var,
        row_var=row_var,
        line_levels=line_levels,
        col_levels=col_levels,
        row_levels=row_levels,
        ribbon=ribbon,
        include_ribbon=show_ribbon,
    )

    nrows = len(row_levels)
    ncols = len(col_levels)

    top_h = strip_top_height_mm if (col_var is not None and ncols > 1) else 0.0
    right_w = strip_right_width_mm if (row_var is not None and nrows > 1) else 0.0

    fig, axes, layout = _init_box_figure(
        nrows=nrows,
        ncols=ncols,
        boxsize_mm=boxsize,
        panel_gap_mm=panel_gap,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        colorbar_width_mm=0.0,
        colorbar_pad_mm=0.0,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=(x_label or ""),
        y_label_text=(y_label or value_col),
        colorbar_label_text=None,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        dpi=dpi,
        font_family=font_family,
        sharex=True,
        sharey=True,
        transparent=transparent,
    )

    for i, row_lv in enumerate(row_levels):
        for j, col_lv in enumerate(col_levels):
            ax = axes[i, j]

            if vertical_shadows:
                for (x0, x1), c in vertical_shadows.items():
                    _safe_axvspan(ax, x0, x1, color=_rgba_color(c), zorder=0)
            if horizontal_shadows:
                for (y0, y1), c in horizontal_shadows.items():
                    _safe_axhspan(ax, y0, y1, color=_rgba_color(c), zorder=0)

            cell = dfw
            if col_var is not None and col_lv is not None:
                cell = cell[cell[col_var] == col_lv]
            if row_var is not None and row_lv is not None:
                cell = cell[cell[row_var] == row_lv]

            for lv in line_levels:
                sub = cell[cell[line_var] == lv]
                series_list = [s for s in sub[value_col].tolist() if isinstance(s, pd.Series)]
                xg, mean, low, high = _series_grid_stats(series_list, ribbon=ribbon)
                if xg.size == 0:
                    continue

                c = line_cmap.get(lv, "black")
                ax.plot(xg, mean, color=c, lw=line_width, alpha=line_alpha, label=str(lv), zorder=3)

                if show_ribbon:
                    ax.fill_between(xg, low, high, color=c, alpha=ribbon_alpha, linewidth=0, zorder=2)

            if vertical_lines is not None:
                for xv in vertical_lines:
                    ax.axvline(x=xv, color=vline_color, linestyle=vline_style, linewidth=vline_width, alpha=vline_alpha, zorder=4)

            if horizontal_lines is not None:
                for yv in horizontal_lines:
                    ax.axhline(y=yv, color=hline_color, linestyle=hline_style, linewidth=hline_width, alpha=hline_alpha, zorder=4)

            ax.set_xlim(*x_limits_use)
            ax.set_ylim(*y_limits_use)

            if x_log:
                ax.set_xscale("log")
            if y_log:
                ax.set_yscale("log")

            if y_tick_labels is not None:
                labels = [str(v) for v in y_tick_labels]
                if len(labels) > 0:
                    if y_ticks is not None:
                        ticks = np.asarray(y_ticks, dtype=float)
                        if ticks.size != len(labels):
                            raise ValueError("y_ticks and y_tick_labels must have the same length.")
                    else:
                        if len(labels) == 1:
                            ticks = np.array([(y_limits_use[0] + y_limits_use[1]) / 2.0], dtype=float)
                        else:
                            ticks = np.linspace(y_limits_use[0], y_limits_use[1], num=len(labels), dtype=float)
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(labels)

            ax.tick_params(labelsize=tick_label_fontsize)
            if grid:
                ax.grid(True, alpha=grid_alpha, linestyle="--", zorder=0)

            ax.spines["top"].set_visible(show_top_right_axes)
            ax.spines["right"].set_visible(show_top_right_axes)

            if (i == nrows - 1) and (not single_x_label):
                ax.set_xlabel(x_label or "", fontsize=axis_label_fontsize)
            if (j == 0) and (not single_y_label):
                ax.set_ylabel(y_label or value_col, fontsize=axis_label_fontsize)

    # strips
    col_labs = [f"{lv}" for lv in col_levels] if (col_var is not None and ncols > 1) else None
    row_labs = [f"{lv}" for lv in row_levels] if (row_var is not None and nrows > 1) else None
    _ = _add_strips_mm(
        fig,
        axes,
        col_labels=col_labs,
        row_labels=row_labs,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
    )

    if transparent:
        _ensure_strip_background_opaque(fig)

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=x_label,
        y_label=(y_label or value_col),
        axis_label_fontsize=axis_label_fontsize,
        title=title,
        title_fontsize=title_fontsize,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        font_family=font_family,
    )

    if legend_loc != "none":
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if legend_ncol is None:
            legend_ncol = 1

        inside_map = not _is_outside_legend_loc(legend_loc)
        _place_legend(
            fig,
            axes[0, 0],
            handles,
            labels,
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_ncol=legend_ncol,
            legend_framealpha=legend_framealpha,
            inside_map=inside_map,
        )

    return fig


def plot_single_effect_series(
    df: pd.DataFrame,
    value_col: str = "Value",
    x_var: str = "Phase",
    x_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    y_ticks: Optional[Union[List[float], np.ndarray]] = None,
    y_tick_labels: Optional[List[str]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = "Arial",
    line_palette: Union[str, List, Dict] = "viridis",
    line_width: float = 2.0,
    line_alpha: float = 0.95,
    show_ribbon: bool = True,
    ribbon_alpha: float = 0.25,
    ribbon: str = "sem",
    grid: bool = False,
    grid_alpha: float = 0.3,
    dpi: int = 300,
    show_top_right_axes: bool = True,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    legend_loc: str = "right",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 10,
    y_label_offset_mm: float = 18,
    transparent: bool = False,
) -> plt.Figure:
    """Single effect series plot (no faceting)."""
    return _plot_interaction_series_grid(
        df=df,
        value_col=value_col,
        line_var=x_var,
        col_var=None,
        row_var=None,
        line_levels=x_levels,
        col_levels=None,
        row_levels=None,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        y_ticks=y_ticks,
        y_tick_labels=y_tick_labels,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        line_palette=line_palette,
        line_width=line_width,
        line_alpha=line_alpha,
        show_ribbon=show_ribbon,
        ribbon_alpha=ribbon_alpha,
        ribbon=ribbon,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        show_top_right_axes=show_top_right_axes,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=16,
        label_top_bg_color="lightgray",
        label_right_bg_color="lightgray",
        label_text_color="black",
        label_fontweight="normal",
        strip_top_height_mm=0.0,
        strip_right_width_mm=0.0,
        strip_pad_mm=0.0,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=(0.0, 0.0),
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        transparent=transparent,
    )


def plot_triple_interaction_series(
    df: pd.DataFrame,
    value_col: str = "value",
    x_var: str = "phase",
    panel_var: str = "region",
    facet_var: str = "lat",
    x_levels: Optional[List] = None,
    panel_levels: Optional[List] = None,
    facet_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    y_ticks: Optional[Union[List[float], np.ndarray]] = None,
    y_tick_labels: Optional[List[str]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = "Arial",
    line_palette: Union[str, List, Dict] = "viridis",
    line_width: float = 2.0,
    line_alpha: float = 0.95,
    show_ribbon: bool = True,
    ribbon_alpha: float = 0.25,
    ribbon: str = "sem",
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    show_top_right_axes: bool = True,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 16,
    label_top_bg_color: str = "lightgray",
    label_right_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_right_width_mm: float = 3.0,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    legend_loc: str = "right",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (3.0, 3.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    transparent: bool = False,
) -> plt.Figure:
    """Triple interaction series plot (panel × facet)."""
    return _plot_interaction_series_grid(
        df=df,
        value_col=value_col,
        line_var=x_var,
        col_var=panel_var,
        row_var=facet_var,
        line_levels=x_levels,
        col_levels=panel_levels,
        row_levels=facet_levels,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        y_ticks=y_ticks,
        y_tick_labels=y_tick_labels,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        line_palette=line_palette,
        line_width=line_width,
        line_alpha=line_alpha,
        show_ribbon=show_ribbon,
        ribbon_alpha=ribbon_alpha,
        ribbon=ribbon,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        show_top_right_axes=show_top_right_axes,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=strip_right_width_mm,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        transparent=transparent,
    )


def plot_double_interaction_series(
    df: pd.DataFrame,
    value_col: str = "value",
    x_var: str = "phase",
    panel_var: str = "region",
    x_levels: Optional[List] = None,
    panel_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    y_ticks: Optional[Union[List[float], np.ndarray]] = None,
    y_tick_labels: Optional[List[str]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = "Arial",
    line_palette: Union[str, List, Dict] = "viridis",
    line_width: float = 2.0,
    line_alpha: float = 0.95,
    show_ribbon: bool = True,
    ribbon_alpha: float = 0.25,
    ribbon: str = "sem",
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    show_top_right_axes: bool = True,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 16,
    label_top_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    legend_loc: str = "right",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (3.0, 3.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    transparent: bool = False,
) -> plt.Figure:
    """Double interaction series plot (panel only)."""
    return _plot_interaction_series_grid(
        df=df,
        value_col=value_col,
        line_var=x_var,
        col_var=panel_var,
        row_var=None,
        line_levels=x_levels,
        col_levels=panel_levels,
        row_levels=None,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        y_ticks=y_ticks,
        y_tick_labels=y_tick_labels,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        line_palette=line_palette,
        line_width=line_width,
        line_alpha=line_alpha,
        show_ribbon=show_ribbon,
        ribbon_alpha=ribbon_alpha,
        ribbon=ribbon,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        show_top_right_axes=show_top_right_axes,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_top_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=0.0,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        transparent=transparent,
    )


def _cell_aggregate_xyz(df_list: List[pd.DataFrame], mode: str = "mean"):
    """
    Aggregate a list of 2D DataFrames onto a common numeric X/Y grid.

    Each df in df_list is expected to be a DataFrame with:
      - columns representing x coordinates
      - index representing y coordinates
      - values are Z

    Returns:
        X (2D), Y (2D), Z (2D), x_coords (1D), y_coords (1D), y_tick_labels (optional)

    Notes:
        Missing points are left as NaN before aggregation.
    """
    if len(df_list) == 0:
        return None, None, None, None, None, None

    # Preserve first-seen order across cells for stable categorical labels.
    x_all = []
    y_all = []
    for d in df_list:
        if not isinstance(d, pd.DataFrame) or d.empty:
            continue
        x_all.extend(list(d.columns))
        y_all.extend(list(d.index))

    x_labels = list(dict.fromkeys(x_all))
    y_labels = list(dict.fromkeys(y_all))

    if len(x_labels) == 0 or len(y_labels) == 0:
        return None, None, None, None, None, None

    x_labels = sorted(x_labels, key=lambda v: _to_numeric_index([v])[0])
    y_labels = sorted(y_labels, key=lambda v: _to_numeric_index([v])[0])

    x_coords = _to_numeric_index(x_labels)
    y_coords = _to_numeric_index(y_labels)

    X, Y = np.meshgrid(x_coords, y_coords)

    stack = []
    for d in df_list:
        if not isinstance(d, pd.DataFrame) or d.empty:
            continue
        # Reindex to full grid
        dd = d.reindex(index=y_labels, columns=x_labels)
        stack.append(dd.to_numpy(dtype=float))

    if len(stack) == 0:
        return None, None, None, None, None, None

    A = np.stack(stack, axis=0)
    if mode == "median":
        Z = np.nanmedian(A, axis=0)
    else:
        Z = np.nanmean(A, axis=0)

    y_num = pd.to_numeric(np.asarray(y_labels), errors="coerce")
    y_tick_labels = None if np.all(np.isfinite(y_num)) else [str(v) for v in y_labels]

    return X, Y, Z, x_coords, y_coords, y_tick_labels


def _compute_global_vrange(cells: List[np.ndarray], vmin: Optional[float], vmax: Optional[float], vmode: str):
    """Compute vmin/vmax from a list of Z arrays."""
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)

    vals = []
    for z in cells:
        if z is None:
            continue
        f = z[np.isfinite(z)]
        if f.size:
            vals.append(f)
    if not vals:
        vmin0, vmax0 = 0.0, 1.0
    else:
        allv = np.concatenate(vals)
        vmin0, vmax0 = float(np.min(allv)), float(np.max(allv))

    if vmode == "sym":
        m = max(abs(vmin0), abs(vmax0))
        return -m, m

    return vmin0, vmax0


def _plot_interaction_df_grid(
    *,
    df: pd.DataFrame,
    value_col: str,
    mode: str,
    col_var: Optional[str],
    row_var: Optional[str],
    col_levels: Optional[List],
    row_levels: Optional[List],
    x_label: Optional[str],
    single_x_label: bool,
    y_label: Optional[str],
    single_y_label: bool,
    x_limits: Optional[Tuple[float, float]],
    y_limits: Optional[Tuple[float, float]],
    x_log: bool,
    y_log: bool,
    title: Optional[str],
    font_family: str,
    cmap: Union[str, plt.Colormap],
    vmin: Optional[float],
    vmax: Optional[float],
    vmode: str,
    dpi: int,
    grid: bool,
    grid_alpha: float,
    # geometry
    boxsize: Tuple[float, float],
    panel_gap: Tuple[float, float],
    include_global_label_margins: bool,
    x_label_offset_mm: float,
    y_label_offset_mm: float,
    cbar_label_offset_mm: float,
    # refs/shadows
    vertical_lines: Optional[Union[List, np.ndarray]],
    vline_color: str,
    vline_style: str,
    vline_width: float,
    vline_alpha: float,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]],
    horizontal_lines: Optional[Union[List, np.ndarray]],
    hline_color: str,
    hline_style: str,
    hline_width: float,
    hline_alpha: float,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]],
    # strips
    label_fontsize: int,
    label_top_bg_color: str,
    label_right_bg_color: str,
    label_text_color: str,
    label_fontweight: str,
    strip_top_height_mm: float,
    strip_right_width_mm: float,
    strip_pad_mm: float,
    # text sizes
    title_fontsize: int,
    axis_label_fontsize: int,
    tick_label_fontsize: int,
    # colorbar
    colorbar_label: Optional[str],
    colorbar_width_mm: float,
    colorbar_pad_mm: float,
    # annotation
    annotate_input_vars: bool,
    input_vars_box_loc: str,
    input_vars_fontsize: int,
    input_vars_facecolor: str,
    input_vars_text_color: str,
    # transparent
    transparent: bool,
) -> plt.Figure:
    """Core heatmap grid plotter where each row holds a 2D DataFrame."""
    plt.rcParams["font.family"] = font_family
    dfx = df.copy()

    if col_var is not None:
        if col_levels is None:
            col_levels = _ordered_levels(dfx[col_var], None)
    else:
        col_levels = [None]

    if row_var is not None:
        if row_levels is None:
            row_levels = _ordered_levels(dfx[row_var], None)
    else:
        row_levels = [None]

    nrows = len(row_levels)
    ncols = len(col_levels)

    top_h = strip_top_height_mm if (col_var is not None and ncols > 1) else 0.0
    right_w = strip_right_width_mm if (row_var is not None and nrows > 1) else 0.0

    fig, axes, layout = _init_box_figure(
        nrows=nrows,
        ncols=ncols,
        boxsize_mm=boxsize,
        panel_gap_mm=panel_gap,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        colorbar_width_mm=colorbar_width_mm,
        colorbar_pad_mm=colorbar_pad_mm,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=(x_label or "x"),
        y_label_text=(y_label or value_col),
        colorbar_label_text=colorbar_label,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=cbar_label_offset_mm,
        dpi=dpi,
        font_family=font_family,
        sharex=True,
        sharey=True,
        transparent=transparent,
    )

    # Pre-compute Z arrays for vmin/vmax
    Z_cells: List[np.ndarray] = []
    cell_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[List[str]]]] = {}

    for i, row_lv in enumerate(row_levels):
        for j, col_lv in enumerate(col_levels):
            cell = dfx
            if col_var is not None and col_lv is not None:
                cell = cell[cell[col_var] == col_lv]
            if row_var is not None and row_lv is not None:
                cell = cell[cell[row_var] == row_lv]

            df_list = [d for d in cell[value_col].tolist() if isinstance(d, pd.DataFrame)]
            X, Y, Z, _, y_coords, y_tick_labels = _cell_aggregate_xyz(df_list, mode=mode)
            if Z is not None:
                Z_cells.append(Z)
                cell_cache[(i, j)] = (X, Y, Z, y_coords, y_tick_labels)

    vmin_use, vmax_use = _compute_global_vrange(Z_cells, vmin=vmin, vmax=vmax, vmode=vmode)

    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = mpl.colors.Normalize(vmin=vmin_use, vmax=vmax_use)

    last_mappable = None

    for i, row_lv in enumerate(row_levels):
        for j, col_lv in enumerate(col_levels):
            ax = axes[i, j]

            if vertical_shadows:
                for (x0, x1), c in vertical_shadows.items():
                    _safe_axvspan(ax, x0, x1, color=_rgba_color(c), zorder=2)
            if horizontal_shadows:
                for (y0, y1), c in horizontal_shadows.items():
                    _safe_axhspan(ax, y0, y1, color=_rgba_color(c), zorder=2)

            if (i, j) in cell_cache:
                X, Y, Z, y_coords, y_tick_labels = cell_cache[(i, j)]
                m = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cmap_obj, norm=norm, zorder=1)
                last_mappable = m
                if y_tick_labels is not None:
                    ax.set_yticks(y_coords)
                    ax.set_yticklabels(y_tick_labels)
                    ax.tick_params(labelleft=(j == 0))

            if vertical_lines is not None:
                for xv in vertical_lines:
                    ax.axvline(x=xv, color=vline_color, linestyle=vline_style, linewidth=vline_width, alpha=vline_alpha, zorder=3)

            if horizontal_lines is not None:
                for yv in horizontal_lines:
                    ax.axhline(y=yv, color=hline_color, linestyle=hline_style, linewidth=hline_width, alpha=hline_alpha, zorder=3)

            if x_log:
                ax.set_xscale("log")
            if y_log:
                ax.set_yscale("log")
            if x_limits:
                ax.set_xlim(*x_limits)
            if y_limits:
                ax.set_ylim(*y_limits)

            if i == nrows - 1:
                ax.set_xlabel("" if single_x_label else (x_label or "x"), fontsize=axis_label_fontsize)
            if j == 0:
                ax.set_ylabel("" if single_y_label else (y_label or value_col), fontsize=axis_label_fontsize)
                
            ax.tick_params(labelsize=tick_label_fontsize)
            if grid:
                ax.grid(True, alpha=grid_alpha, linestyle="--", zorder=4)

            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

    # strips
    col_labs = [f"{lv}" for lv in col_levels] if (col_var is not None and ncols > 1) else None
    row_labs = [f"{lv}" for lv in row_levels] if (row_var is not None and nrows > 1) else None
    right_edge = _add_strips_mm(
        fig,
        axes,
        col_labels=col_labs,
        row_labels=row_labs,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
    )

    if transparent:
        _ensure_strip_background_opaque(fig)

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=(x_label or "x"),
        y_label=(y_label or value_col),
        axis_label_fontsize=axis_label_fontsize,
        title=title,
        title_fontsize=title_fontsize,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        font_family=font_family,
    )

    if annotate_input_vars:
        mapping_text = f"Heatmap: {value_col}  |  Columns: {col_var or '-'}  |  Rows: {row_var or '-'}  |  agg: {mode}"
        loc_map = {
            "upper left": (0.01, 0.995, "top", "left"),
            "upper right": (0.99, 0.995, "top", "right"),
            "lower left": (0.01, 0.01, "bottom", "left"),
            "lower right": (0.99, 0.01, "bottom", "right"),
        }
        x0, y0, va, ha = loc_map.get(input_vars_box_loc, (0.01, 0.995, "top", "left"))
        fig.text(
            x0,
            y0,
            mapping_text,
            ha=ha,
            va=va,
            fontsize=input_vars_fontsize,
            color=input_vars_text_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=input_vars_facecolor, edgecolor="none", alpha=0.9),
        )

    # colorbar
    if last_mappable is not None and colorbar_width_mm > 0:
        _add_global_colorbar(
            fig,
            layout,
            mappable=last_mappable,
            right_edge=float(right_edge),
            colorbar_label=colorbar_label,
            tick_label_fontsize=tick_label_fontsize,
            axis_label_fontsize=axis_label_fontsize,
            font_family=font_family,
        )

    return fig


def plot_single_effect_df(
    df: pd.DataFrame,
    value_col: str = "Value",
    mode: str = "mean",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    single_x_label: bool = True,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = "Arial",
    cmap: Union[str, plt.Colormap] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmode: str = "auto",
    dpi: int = 300,
    grid: bool = False,
    grid_alpha: float = 0.3,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 10,
    y_label_offset_mm: float = 18,
    cbar_label_offset_mm: float = 18,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "white",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.8,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "white",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.8,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    colorbar_label: Optional[str] = None,
    colorbar_width_mm: float = 6.0,
    colorbar_pad_mm: float = 4.0,
    annotate_input_vars: bool = False,
    input_vars_box_loc: str = "upper left",
    input_vars_fontsize: int = 10,
    input_vars_facecolor: str = "#f0f0f0",
    input_vars_text_color: str = "black",
    transparent: bool = False,
) -> plt.Figure:
    """Single heatmap (no faceting)."""
    return _plot_interaction_df_grid(
        df=df,
        value_col=value_col,
        mode=mode,
        col_var=None,
        row_var=None,
        col_levels=None,
        row_levels=None,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        vmode=vmode,
        dpi=dpi,
        grid=grid,
        grid_alpha=grid_alpha,
        boxsize=boxsize,
        panel_gap=(0.0, 0.0),
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=cbar_label_offset_mm,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=16,
        label_top_bg_color="lightgray",
        label_right_bg_color="lightgray",
        label_text_color="black",
        label_fontweight="normal",
        strip_top_height_mm=0.0,
        strip_right_width_mm=0.0,
        strip_pad_mm=0.0,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        colorbar_label=colorbar_label,
        colorbar_width_mm=colorbar_width_mm,
        colorbar_pad_mm=colorbar_pad_mm,
        annotate_input_vars=annotate_input_vars,
        input_vars_box_loc=input_vars_box_loc,
        input_vars_fontsize=input_vars_fontsize,
        input_vars_facecolor=input_vars_facecolor,
        input_vars_text_color=input_vars_text_color,
        transparent=transparent,
    )


def plot_triple_interaction_df(
    df: pd.DataFrame,
    panel_var: str = "region",
    facet_var: str = "lat",
    panel_levels: Optional[List] = None,
    facet_levels: Optional[List] = None,
    value_col: str = "value",
    mode: str = "mean",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    single_x_label: bool = True,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = "Arial",
    cmap: Union[str, plt.Colormap] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmode: str = "auto",
    dpi: int = 300,
    grid: bool = False,
    grid_alpha: float = 0.3,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (3.0, 3.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    cbar_label_offset_mm: float = 1.5,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "white",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.8,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "white",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.8,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 16,
    label_top_bg_color: str = "lightgray",
    label_right_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_right_width_mm: float = 3.0,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    colorbar_label: Optional[str] = None,
    colorbar_width_mm: float = 3.0,
    colorbar_pad_mm: float = 1.5,
    annotate_input_vars: bool = False,
    input_vars_box_loc: str = "upper left",
    input_vars_fontsize: int = 10,
    input_vars_facecolor: str = "#f0f0f0",
    input_vars_text_color: str = "black",
    transparent: bool = False,
) -> plt.Figure:
    """Grid of heatmaps (panel × facet)."""
    return _plot_interaction_df_grid(
        df=df,
        value_col=value_col,
        mode=mode,
        col_var=panel_var,
        row_var=facet_var,
        col_levels=panel_levels,
        row_levels=facet_levels,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        vmode=vmode,
        dpi=dpi,
        grid=grid,
        grid_alpha=grid_alpha,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=cbar_label_offset_mm,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=strip_right_width_mm,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        colorbar_label=colorbar_label,
        colorbar_width_mm=colorbar_width_mm,
        colorbar_pad_mm=colorbar_pad_mm,
        annotate_input_vars=annotate_input_vars,
        input_vars_box_loc=input_vars_box_loc,
        input_vars_fontsize=input_vars_fontsize,
        input_vars_facecolor=input_vars_facecolor,
        input_vars_text_color=input_vars_text_color,
        transparent=transparent,
    )


def plot_double_interaction_df(
    df: pd.DataFrame,
    panel_var: str = "region",
    panel_levels: Optional[List] = None,
    value_col: str = "value",
    mode: str = "mean",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    single_x_label: bool = True,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = "Arial",
    cmap: Union[str, plt.Colormap] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmode: str = "auto",
    dpi: int = 300,
    grid: bool = False,
    grid_alpha: float = 0.3,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (3.0, 3.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    cbar_label_offset_mm: float = 1.5,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "white",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.8,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "white",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.8,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 16,
    label_top_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    colorbar_label: Optional[str] = None,
    colorbar_width_mm: float = 3.0,
    colorbar_pad_mm: float = 1.5,
    annotate_input_vars: bool = False,
    input_vars_box_loc: str = "upper left",
    input_vars_fontsize: int = 10,
    input_vars_facecolor: str = "#f0f0f0",
    input_vars_text_color: str = "black",
    transparent: bool = False,
) -> plt.Figure:
    """Side-by-side heatmaps (panel only)."""
    return _plot_interaction_df_grid(
        df=df,
        value_col=value_col,
        mode=mode,
        col_var=panel_var,
        row_var=None,
        col_levels=panel_levels,
        row_levels=None,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        vmode=vmode,
        dpi=dpi,
        grid=grid,
        grid_alpha=grid_alpha,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=cbar_label_offset_mm,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_top_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=0.0,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        colorbar_label=colorbar_label,
        colorbar_width_mm=colorbar_width_mm,
        colorbar_pad_mm=colorbar_pad_mm,
        annotate_input_vars=annotate_input_vars,
        input_vars_box_loc=input_vars_box_loc,
        input_vars_fontsize=input_vars_fontsize,
        input_vars_facecolor=input_vars_facecolor,
        input_vars_text_color=input_vars_text_color,
        transparent=transparent,
    )


def _finite(arr):
    """Return finite values from array-like."""
    a = np.asarray(arr)
    return a[np.isfinite(a)]


def _auto_limits(x_arrays, y_arrays, x_pad_frac, y_pad_frac, y_log: bool = False):
    """Compute padded limits from multiple x/y arrays."""
    if x_arrays:
        xs = _finite(np.concatenate([np.asarray(v).ravel() for v in x_arrays if v is not None]))
    else:
        xs = np.array([0.0, 1.0])

    if y_arrays:
        ys = _finite(np.concatenate([np.asarray(v).ravel() for v in y_arrays if v is not None]))
    else:
        ys = np.array([0.0, 1.0])

    if xs.size == 0:
        xs = np.array([0.0, 1.0])
    if ys.size == 0:
        ys = np.array([0.0, 1.0])

    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))

    xr = x1 - x0
    yr = y1 - y0
    x_pad = xr * float(x_pad_frac)
    y_pad = yr * float(y_pad_frac)

    if y_log:
        y0 = max(y0, 1e-12)

    return (x0 - x_pad, x1 + x_pad), (y0 - y_pad, y1 + y_pad)


def _pick_p_column(slope_df: pd.DataFrame, preferred: str) -> Optional[str]:
    """Pick a p-value column name from slope_df with strict priority fallbacks."""
    if slope_df is None or slope_df.empty:
        return None

    candidates: List[str] = []
    if preferred:
        candidates.append(str(preferred))
    candidates.extend(list(_P_VALUE_PRIORITY))
    candidates.extend(list(_P_VALUE_FALLBACK))
    candidates = _unique_preserve_order([c for c in candidates if c])

    for cand in candidates:
        if cand not in slope_df.columns:
            continue
        v = pd.to_numeric(slope_df[cand], errors="coerce").astype(float).to_numpy()
        if np.any(np.isfinite(v)):
            return cand
    return None


def _format_slope_text(
    slope_row: pd.Series,
    *,
    beta_col: str,
    lower_col: str,
    upper_col: str,
    p_col: Optional[str],
    text_fmt: str,
) -> str:
    """Format slope annotation string."""
    beta = float(slope_row.get(beta_col, np.nan))
    lower = float(slope_row.get(lower_col, np.nan))
    upper = float(slope_row.get(upper_col, np.nan))

    pval = _resolve_p_value_from_row(slope_row, preferred=p_col)
    stars = p_to_stars(pval)

    return text_fmt.format(beta=beta, lower=lower, upper=upper, p=pval, q=pval, stars=stars)


def _plot_interaction_fit_grid(
    *,
    df: pd.DataFrame,
    curve: pd.DataFrame,
    slope: pd.DataFrame,
    value_col: str,
    x_var: str,
    col_var: Optional[str],
    row_var: Optional[str],
    color_var: Optional[str],
    col_levels: Optional[List],
    row_levels: Optional[List],
    x_label: Optional[str],
    single_x_label: bool,
    y_label: Optional[str],
    single_y_label: bool,
    x_limits: Optional[Tuple[float, float]],
    y_limits: Optional[Tuple[float, float]],
    x_log: bool,
    y_log: bool,
    title: Optional[str],
    font_family: str,
    palette: Union[str, List, Dict],
    jitter_alpha: float,
    jitter_size: float,
    curve_line_width: float,
    curve_line_color: str,
    ribbon_alpha: float,
    show_ribbon: bool,
    grid: bool,
    grid_alpha: float,
    dpi: int,
    seed: int,
    show_top_right_axes: bool,
    vertical_lines: Optional[Union[List, np.ndarray]],
    vline_color: str,
    vline_style: str,
    vline_width: float,
    vline_alpha: float,
    horizontal_lines: Optional[Union[List, np.ndarray]],
    hline_color: str,
    hline_style: str,
    hline_width: float,
    hline_alpha: float,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]],
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]],
    label_fontsize: int,
    label_top_bg_color: str,
    label_right_bg_color: str,
    label_text_color: str,
    label_fontweight: str,
    strip_top_height_mm: float,
    strip_right_width_mm: float,
    strip_pad_mm: float,
    title_fontsize: int,
    axis_label_fontsize: int,
    tick_label_fontsize: int,
    legend_loc: str,
    legend_ncol: Optional[int],
    legend_framealpha: float,
    legend_fontsize: int,
    boxsize: Tuple[float, float],
    panel_gap: Tuple[float, float],
    include_global_label_margins: bool,
    x_label_offset_mm: float,
    y_label_offset_mm: float,
    slope_beta_col: str,
    slope_lower_col: str,
    slope_upper_col: str,
    slope_p_col: str,
    slope_text_fmt: str,
    slope_text_loc: Union[str, Tuple[float, float]],
    slope_text_coord: str,
    slope_text_offset: Tuple[float, float],
    slope_text_ha: Optional[str],
    slope_text_va: Optional[str],
    slope_text_box_alpha: float,
    transparent: bool,
) -> plt.Figure:
    """Core fit grid plotter."""
    rng = np.random.default_rng(seed)
    plt.rcParams["font.family"] = font_family

    dfw = df.copy()
    curw = _normalize_emm_ci_columns(curve.copy())
    slw = slope.copy()

    if col_var is not None:
        if col_levels is None:
            col_levels = _ordered_levels(dfw[col_var], None)
    else:
        col_levels = [None]
    if row_var is not None:
        if row_levels is None:
            row_levels = _ordered_levels(dfw[row_var], None)
    else:
        row_levels = [None]

    nrows = len(row_levels)
    ncols = len(col_levels)

    top_h = strip_top_height_mm if (col_var is not None and ncols > 1) else 0.0
    right_w = strip_right_width_mm if (row_var is not None and nrows > 1) else 0.0

    if color_var is None or color_var not in dfw.columns:
        color_levels = ["raw"]
        cmap = {"raw": "gray"}
        dfw["_color_level"] = "raw"
    else:
        color_levels = _ordered_levels(dfw[color_var], None)
        cmap = _build_color_map(color_levels, palette)
        dfw["_color_level"] = dfw[color_var]

    if x_limits is None or y_limits is None:
        x_arrays = []
        y_arrays = []
        if x_var in dfw.columns:
            x_arrays.append(dfw[x_var].to_numpy(dtype=float))
        if x_var in curw.columns:
            x_arrays.append(curw[x_var].to_numpy(dtype=float))
        if value_col in dfw.columns:
            y_arrays.append(dfw[value_col].to_numpy(dtype=float))
        if "emmean" in curw.columns:
            y_arrays.append(curw["emmean"].to_numpy(dtype=float))
        if "lower.CL" in curw.columns:
            y_arrays.append(curw["lower.CL"].to_numpy(dtype=float))
        if "upper.CL" in curw.columns:
            y_arrays.append(curw["upper.CL"].to_numpy(dtype=float))

        x_lim_auto, y_lim_auto = _auto_limits(
            x_arrays,
            y_arrays,
            x_pad_frac=0.05,
            y_pad_frac=0.05,
            y_log=y_log,
        )
        if x_limits is None:
            x_limits = x_lim_auto
        if y_limits is None:
            y_limits = y_lim_auto

    fig, axes, layout = _init_box_figure(
        nrows=nrows,
        ncols=ncols,
        boxsize_mm=boxsize,
        panel_gap_mm=panel_gap,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        colorbar_width_mm=0.0,
        colorbar_pad_mm=0.0,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=(x_label or x_var),
        y_label_text=(y_label or value_col),
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        dpi=dpi,
        font_family=font_family,
        sharex=True,
        sharey=True,
        transparent=transparent,
    )

    p_col = _pick_p_column(slw, slope_p_col)

    for i, row_lv in enumerate(row_levels):
        for j, col_lv in enumerate(col_levels):
            ax = axes[i, j]

            if vertical_shadows:
                for (x0, x1), c in vertical_shadows.items():
                    _safe_axvspan(ax, x0, x1, color=_rgba_color(c), zorder=0)
            if horizontal_shadows:
                for (y0, y1), c in horizontal_shadows.items():
                    _safe_axhspan(ax, y0, y1, color=_rgba_color(c), zorder=0)

            cell_raw = dfw
            cell_curve = curw
            cell_slope = slw

            if col_var is not None and col_lv is not None and col_var in dfw.columns:
                cell_raw = cell_raw[cell_raw[col_var] == col_lv]
            if row_var is not None and row_lv is not None and row_var in dfw.columns:
                cell_raw = cell_raw[cell_raw[row_var] == row_lv]

            if col_var is not None and col_lv is not None and col_var in curw.columns:
                cell_curve = cell_curve[cell_curve[col_var] == col_lv]
            if row_var is not None and row_lv is not None and row_var in curw.columns:
                cell_curve = cell_curve[cell_curve[row_var] == row_lv]

            if col_var is not None and col_lv is not None and col_var in slw.columns:
                cell_slope = cell_slope[cell_slope[col_var] == col_lv]
            if row_var is not None and row_lv is not None and row_var in slw.columns:
                cell_slope = cell_slope[cell_slope[row_var] == row_lv]

            if not cell_raw.empty:
                xs = cell_raw[x_var].to_numpy(dtype=float)
                ys = cell_raw[value_col].to_numpy(dtype=float)
                cols = [cmap.get(lv, "gray") for lv in cell_raw["_color_level"].tolist()]
                ax.scatter(xs, ys, c=cols, alpha=jitter_alpha, s=jitter_size, linewidths=0, zorder=2)

            if not cell_curve.empty and x_var in cell_curve.columns and "emmean" in cell_curve.columns:
                cell_curve = cell_curve.sort_values(x_var)
                xg = cell_curve[x_var].to_numpy(dtype=float)
                em = cell_curve["emmean"].to_numpy(dtype=float)

                if show_ribbon and {"lower.CL", "upper.CL"}.issubset(cell_curve.columns):
                    lo = cell_curve["lower.CL"].to_numpy(dtype=float)
                    hi = cell_curve["upper.CL"].to_numpy(dtype=float)
                    ax.fill_between(
                        xg,
                        lo,
                        hi,
                        color=curve_line_color,
                        alpha=ribbon_alpha,
                        linewidth=0,
                        zorder=1.5,
                    )

                ax.plot(xg, em, color=curve_line_color, lw=curve_line_width, zorder=3, label="Fit")

            if cell_slope is not None and not cell_slope.empty:
                row0 = cell_slope.iloc[0]
                text = _format_slope_text(
                    row0,
                    beta_col=slope_beta_col,
                    lower_col=slope_lower_col,
                    upper_col=slope_upper_col,
                    p_col=p_col,
                    text_fmt=slope_text_fmt,
                )

                if isinstance(slope_text_loc, tuple):
                    x0, y0 = slope_text_loc
                    coord_alias = str(slope_text_coord).strip().lower()
                    if coord_alias in {"axes", "axes fraction"}:
                        xycoords = "axes fraction"
                    elif coord_alias in {"data"}:
                        xycoords = "data"
                    elif coord_alias in {"figure", "figure fraction"}:
                        xycoords = "figure fraction"
                    else:
                        xycoords = "axes fraction"

                    ha = slope_text_ha or ("left" if x0 <= 0.5 else "right")
                    va = slope_text_va or ("top" if y0 >= 0.5 else "bottom")

                    ax.annotate(
                        text,
                        xy=(x0, y0),
                        xycoords=xycoords,
                        textcoords="offset points",
                        xytext=slope_text_offset,
                        ha=ha,
                        va=va,
                        fontsize=tick_label_fontsize,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="white",
                            edgecolor="none",
                            alpha=slope_text_box_alpha,
                        ),
                        zorder=10,
                        clip_on=False,
                    )
                else:
                    anchors = {
                        "upper left": dict(x=0.05, y=0.95, ha="left", va="top"),
                        "upper right": dict(x=0.95, y=0.95, ha="right", va="top"),
                        "lower left": dict(x=0.05, y=0.05, ha="left", va="bottom"),
                        "lower right": dict(x=0.95, y=0.05, ha="right", va="bottom"),
                    }
                    an = anchors.get(str(slope_text_loc).lower(), anchors["upper left"])
                    ax.text(
                        an["x"],
                        an["y"],
                        text,
                        transform=ax.transAxes,
                        ha=an["ha"],
                        va=an["va"],
                        fontsize=tick_label_fontsize,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="white",
                            edgecolor="none",
                            alpha=slope_text_box_alpha,
                        ),
                        zorder=10,
                        clip_on=False,
                    )

            if vertical_lines is not None:
                for xv in vertical_lines:
                    ax.axvline(
                        x=xv,
                        color=vline_color,
                        linestyle=vline_style,
                        linewidth=vline_width,
                        alpha=vline_alpha,
                        zorder=4,
                    )

            if horizontal_lines is not None:
                for yv in horizontal_lines:
                    ax.axhline(
                        y=yv,
                        color=hline_color,
                        linestyle=hline_style,
                        linewidth=hline_width,
                        alpha=hline_alpha,
                        zorder=4,
                    )

            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)

            if x_log:
                ax.set_xscale("log")
            if y_log:
                ax.set_yscale("log")

            ax.tick_params(labelsize=tick_label_fontsize)

            if grid:
                ax.grid(True, alpha=grid_alpha, linestyle="--", zorder=0)

            ax.spines["top"].set_visible(show_top_right_axes)
            ax.spines["right"].set_visible(show_top_right_axes)

            if (i == nrows - 1) and (not single_x_label):
                ax.set_xlabel(x_label or x_var, fontsize=axis_label_fontsize)
            if (j == 0) and (not single_y_label):
                ax.set_ylabel(y_label or value_col, fontsize=axis_label_fontsize)

    col_labs = [f"{lv}" for lv in col_levels] if (col_var is not None and ncols > 1) else None
    row_labs = [f"{lv}" for lv in row_levels] if (row_var is not None and nrows > 1) else None
    _ = _add_strips_mm(
        fig,
        axes,
        col_labels=col_labs,
        row_labels=row_labs,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
    )

    if transparent:
        _ensure_strip_background_opaque(fig)

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=(x_label or x_var),
        y_label=(y_label or value_col),
        axis_label_fontsize=axis_label_fontsize,
        title=title,
        title_fontsize=title_fontsize,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        font_family=font_family,
    )

    if legend_loc != "none":
        handles = [
            Line2D([0], [0], color=curve_line_color, lw=curve_line_width, label="Fit"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="gray",
                alpha=jitter_alpha,
                markersize=float(np.sqrt(jitter_size)),
                label=f"Raw ({color_var or 'raw'})",
            ),
        ]
        if legend_ncol is None:
            legend_ncol = 1

        inside_map = not _is_outside_legend_loc(legend_loc)
        _place_legend(
            fig,
            axes[0, 0],
            handles,
            [h.get_label() for h in handles],
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_ncol=legend_ncol,
            legend_framealpha=legend_framealpha,
            inside_map=inside_map,
        )

    return fig


def plot_triple_interaction_fit(
    df: pd.DataFrame,
    curve: pd.DataFrame,
    slope: pd.DataFrame,
    value_col: str = "Value",
    x_var: str = "LogI_c",
    panel_var: str = "Region",
    facet_var: str = "Stimulus",
    color_var: Optional[str] = "Subject",
    panel_levels: Optional[List] = None,
    facet_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = "Arial",
    palette: Union[str, List, Dict] = "viridis",
    jitter_alpha: float = 0.35,
    jitter_size: float = 10.0,
    curve_line_width: float = 2.0,
    curve_line_color: str = "black",
    ribbon_alpha: float = 0.20,
    show_ribbon: bool = True,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    seed: int = 1,
    show_top_right_axes: bool = True,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 14,
    label_top_bg_color: str = "lightgray",
    label_right_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_right_width_mm: float = 3.0,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 18,
    axis_label_fontsize: int = 14,
    tick_label_fontsize: int = 12,
    legend_loc: str = "none",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (3.0, 3.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    slope_beta_col: str = "slope",
    slope_lower_col: str = "lower",
    slope_upper_col: str = "upper",
    slope_p_col: str = "q",
    slope_text_fmt: str = "β = {beta:.3f}\\nq = {q:.3f} ({stars})",
    slope_text_loc: Union[str, Tuple[float, float]] = "upper left",
    slope_text_coord: str = "axes fraction",
    slope_text_offset: Tuple[float, float] = (0.0, 0.0),
    slope_text_ha: Optional[str] = None,
    slope_text_va: Optional[str] = None,
    slope_text_box_alpha: float = 0.9,
    transparent: bool = False,
) -> plt.Figure:
    """Triple interaction fit plot (panel × facet)."""
    return _plot_interaction_fit_grid(
        df=df,
        curve=curve,
        slope=slope,
        value_col=value_col,
        x_var=x_var,
        col_var=panel_var,
        row_var=facet_var,
        color_var=color_var,
        col_levels=panel_levels,
        row_levels=facet_levels,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        palette=palette,
        jitter_alpha=jitter_alpha,
        jitter_size=jitter_size,
        curve_line_width=curve_line_width,
        curve_line_color=curve_line_color,
        ribbon_alpha=ribbon_alpha,
        show_ribbon=show_ribbon,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        seed=seed,
        show_top_right_axes=show_top_right_axes,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=strip_right_width_mm,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        slope_beta_col=slope_beta_col,
        slope_lower_col=slope_lower_col,
        slope_upper_col=slope_upper_col,
        slope_p_col=slope_p_col,
        slope_text_fmt=slope_text_fmt,
        slope_text_loc=slope_text_loc,
        slope_text_coord=slope_text_coord,
        slope_text_offset=slope_text_offset,
        slope_text_ha=slope_text_ha,
        slope_text_va=slope_text_va,
        slope_text_box_alpha=slope_text_box_alpha,
        transparent=transparent,
    )


def plot_double_interaction_fit(
    df: pd.DataFrame,
    curve: pd.DataFrame,
    slope: pd.DataFrame,
    value_col: str = "Value",
    x_var: str = "LogI_c",
    panel_var: str = "Region",
    color_var: Optional[str] = "Subject",
    panel_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = "Arial",
    palette: Union[str, List, Dict] = "viridis",
    jitter_alpha: float = 0.35,
    jitter_size: float = 10.0,
    curve_line_width: float = 2.0,
    curve_line_color: str = "black",
    ribbon_alpha: float = 0.20,
    show_ribbon: bool = True,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    seed: int = 1,
    show_top_right_axes: bool = True,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 14,
    label_top_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 18,
    axis_label_fontsize: int = 14,
    tick_label_fontsize: int = 12,
    legend_loc: str = "none",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (3.0, 3.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    slope_beta_col: str = "slope",
    slope_lower_col: str = "lower",
    slope_upper_col: str = "upper",
    slope_p_col: str = "q",
    slope_text_fmt: str = "β = {beta:.3f}\\nq = {q:.3f} ({stars})",
    slope_text_loc: Union[str, Tuple[float, float]] = "upper left",
    slope_text_coord: str = "axes fraction",
    slope_text_offset: Tuple[float, float] = (0.0, 0.0),
    slope_text_ha: Optional[str] = None,
    slope_text_va: Optional[str] = None,
    slope_text_box_alpha: float = 0.9,
    transparent: bool = False,
) -> plt.Figure:
    """Double interaction fit plot (panel only)."""
    return _plot_interaction_fit_grid(
        df=df,
        curve=curve,
        slope=slope,
        value_col=value_col,
        x_var=x_var,
        col_var=panel_var,
        row_var=None,
        color_var=color_var,
        col_levels=panel_levels,
        row_levels=None,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        palette=palette,
        jitter_alpha=jitter_alpha,
        jitter_size=jitter_size,
        curve_line_width=curve_line_width,
        curve_line_color=curve_line_color,
        ribbon_alpha=ribbon_alpha,
        show_ribbon=show_ribbon,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        seed=seed,
        show_top_right_axes=show_top_right_axes,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_top_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=0.0,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        slope_beta_col=slope_beta_col,
        slope_lower_col=slope_lower_col,
        slope_upper_col=slope_upper_col,
        slope_p_col=slope_p_col,
        slope_text_fmt=slope_text_fmt,
        slope_text_loc=slope_text_loc,
        slope_text_coord=slope_text_coord,
        slope_text_offset=slope_text_offset,
        slope_text_ha=slope_text_ha,
        slope_text_va=slope_text_va,
        slope_text_box_alpha=slope_text_box_alpha,
        transparent=transparent,
    )


def plot_single_effect_fit(
    df: pd.DataFrame,
    curve: pd.DataFrame,
    slope: pd.DataFrame,
    value_col: str = "Value",
    x_var: str = "LogI_c",
    color_var: Optional[str] = "Subject",
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = "Arial",
    palette: Union[str, List, Dict] = "viridis",
    jitter_alpha: float = 0.35,
    jitter_size: float = 10.0,
    curve_line_width: float = 2.0,
    curve_line_color: str = "black",
    ribbon_alpha: float = 0.20,
    show_ribbon: bool = True,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    seed: int = 1,
    show_top_right_axes: bool = True,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    title_fontsize: int = 18,
    axis_label_fontsize: int = 14,
    tick_label_fontsize: int = 12,
    legend_loc: str = "none",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    slope_beta_col: str = "slope",
    slope_lower_col: str = "lower",
    slope_upper_col: str = "upper",
    slope_p_col: str = "q",
    slope_text_fmt: str = "β = {beta:.3f}\\nq = {q:.3f} ({stars})",
    slope_text_loc: Union[str, Tuple[float, float]] = "upper left",
    slope_text_coord: str = "axes fraction",
    slope_text_offset: Tuple[float, float] = (0.0, 0.0),
    slope_text_ha: Optional[str] = None,
    slope_text_va: Optional[str] = None,
    slope_text_box_alpha: float = 0.9,
    transparent: bool = False,
) -> plt.Figure:
    """Single fit plot (no faceting)."""
    return _plot_interaction_fit_grid(
        df=df,
        curve=curve,
        slope=slope,
        value_col=value_col,
        x_var=x_var,
        col_var=None,
        row_var=None,
        color_var=color_var,
        col_levels=None,
        row_levels=None,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        palette=palette,
        jitter_alpha=jitter_alpha,
        jitter_size=jitter_size,
        curve_line_width=curve_line_width,
        curve_line_color=curve_line_color,
        ribbon_alpha=ribbon_alpha,
        show_ribbon=show_ribbon,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        seed=seed,
        show_top_right_axes=show_top_right_axes,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=14,
        label_top_bg_color="lightgray",
        label_right_bg_color="lightgray",
        label_text_color="black",
        label_fontweight="normal",
        strip_top_height_mm=0.0,
        strip_right_width_mm=0.0,
        strip_pad_mm=0.0,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=(0.0, 0.0),
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        slope_beta_col=slope_beta_col,
        slope_lower_col=slope_lower_col,
        slope_upper_col=slope_upper_col,
        slope_p_col=slope_p_col,
        slope_text_fmt=slope_text_fmt,
        slope_text_loc=slope_text_loc,
        slope_text_coord=slope_text_coord,
        slope_text_offset=slope_text_offset,
        slope_text_ha=slope_text_ha,
        slope_text_va=slope_text_va,
        slope_text_box_alpha=slope_text_box_alpha,
        transparent=transparent,
    )


def plot_prism_boxplot(
    df: pd.DataFrame,
    tuk: Optional[pd.DataFrame] = None,
    *,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    font_family: str = "Arial",
    group_order: Optional[List[str]] = None,
    x_levels: Optional[List[str]] = None,
    jitter_palette: Union[str, List, Dict] = "viridis",
    fill_palette: Union[str, List, Dict] = "viridis",
    outline_palette: Union[str, List, Dict] = "viridis",
    xtick_rotation: float = 0.0,
    ytick_rotation: float = 0.0,
    x_log: bool = False,
    y_log: bool = False,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    y_min_zero: bool = False,
    fill_alpha: float = 0.40,
    show_points: bool = True,
    jitter_alpha: float = 0.35,
    jitter_size: float = 12.0,
    jitter_width: float = 0.18,
    show_box: bool = True,
    box_width: float = 0.55,
    whiskers: Union[str, float, Tuple[float, float]] = "tukey",
    box_edge_color: str = "black",
    box_edge_width: float = 1.0,
    median_color: str = "black",
    median_linewidth: float = 1.0,
    whisker_color: str = "black",
    whisker_linewidth: float = 1.0,
    cap_linewidth: float = 1.0,
    outlier_marker: str = "o",
    outlier_markersize: float = 3.0,
    show_raw_mean_line: bool = True,
    raw_mean_line_width: float = 1.0,
    raw_mean_line_color: str = "#404040",
    raw_mean_line_style: str = "--",
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 150,
    seed: int = 1,
    show_top_right_axes: bool = True,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    show_brackets: bool = True,
    hide_ns: bool = True,
    y_start: float = 0.70,
    y_end: float = 0.95,
    y_step: Optional[float] = 0.08,
    bracket_height_frac: float = 0.018,
    bracket_color: str = "black",
    bracket_linewidth: float = 1.2,
    bracket_text_size: int = 12,
    title_fontsize: int = 18,
    axis_label_fontsize: int = 14,
    tick_label_fontsize: int = 12,
    legend_loc: str = "none",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    transparent: bool = False,
) -> plt.Figure:
    """
    GraphPad/Prism-style one-factor boxplot.

    Parameters
    ----------
    df:
        Wide table; each column is a group, rows are replicates (NaN allowed).
    tuk:
        Prism multiple-comparisons table (optional). Expected:
          - first column contains comparisons like "A vs. B"
          - a column containing star summary ("*", "**", "ns", etc.)
        The function will try to infer which column has stars.
    """
    plt.rcParams["font.family"] = font_family
    rng = np.random.default_rng(seed)

    if df is None or df.empty:
        raise ValueError("df is empty; expected a wide table with group columns.")

    if x_levels is None:
        if group_order is not None:
            x_levels = list(group_order)
        else:
            x_levels = list(df.columns)
    groups = list(x_levels)
    missing = [g for g in groups if g not in df.columns]
    if missing:
        raise ValueError(f"x_levels/group_order contains missing groups: {missing}")

    jitter_cmap = _build_color_map(groups, jitter_palette)
    fill_cmap = _build_color_map(groups, fill_palette)
    outline_cmap = _build_color_map(groups, outline_palette) if outline_palette is not None else {}
    use_outline_for_lines = outline_palette is not None
    x_offset = 1.0 if x_log else 0.0
    x_to_num = {str(g): i + x_offset for i, g in enumerate(groups)}

    fig, axes, layout = _init_box_figure(
        nrows=1,
        ncols=1,
        boxsize_mm=boxsize,
        panel_gap_mm=(0.0, 0.0),
        strip_top_height_mm=0.0,
        strip_right_width_mm=0.0,
        strip_pad_mm=0.0,
        colorbar_width_mm=0.0,
        colorbar_pad_mm=0.0,
        single_x_label=True,
        single_y_label=True,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=(x_label or ""),
        y_label_text=(y_label or ""),
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        dpi=dpi,
        font_family=font_family,
        sharex=False,
        sharey=False,
        transparent=transparent,
    )
    ax = axes[0, 0]

    if vertical_shadows:
        for (x0, x1), c in vertical_shadows.items():
            _safe_axvspan(ax, x0, x1, color=_rgba_color(c), zorder=0)
    if horizontal_shadows:
        for (y0, y1), c in horizontal_shadows.items():
            _safe_axhspan(ax, y0, y1, color=_rgba_color(c), zorder=0)

    data_for_boxes: List[np.ndarray] = []
    pos_for_boxes: List[float] = []
    all_vals = []

    for g in groups:
        vals = df[g].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        all_vals.append(vals)
        if vals.size:
            data_for_boxes.append(vals)
            pos_for_boxes.append(float(x_to_num[str(g)]))

    if show_box and data_for_boxes:
        whis = _parse_whiskers(whiskers)
        bp = ax.boxplot(
            data_for_boxes,
            positions=pos_for_boxes,
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            whis=whis,
            medianprops=dict(color=median_color, linewidth=median_linewidth),
            whiskerprops=dict(color=whisker_color, linewidth=whisker_linewidth),
            capprops=dict(color=whisker_color, linewidth=cap_linewidth),
            flierprops=dict(
                marker=outlier_marker,
                markersize=outlier_markersize,
                markerfacecolor=whisker_color,
                markeredgecolor=whisker_color,
                alpha=0.7,
            ),
        )

        for k, b in enumerate(bp["boxes"]):
            g = groups[k] if k < len(groups) else None
            face = fill_cmap.get(g, "gray")
            outline_c = outline_cmap.get(g, box_edge_color) if use_outline_for_lines else box_edge_color
            b.set_facecolor(mpl.colors.to_rgba(face, fill_alpha))
            b.set_edgecolor(outline_c)
            b.set_linewidth(box_edge_width)

            if use_outline_for_lines:
                if k < len(bp.get("medians", [])):
                    bp["medians"][k].set_color(outline_c)
                    bp["medians"][k].set_linewidth(median_linewidth)

                for idx in (2 * k, 2 * k + 1):
                    if idx < len(bp.get("whiskers", [])):
                        bp["whiskers"][idx].set_color(outline_c)
                        bp["whiskers"][idx].set_linewidth(whisker_linewidth)
                    if idx < len(bp.get("caps", [])):
                        bp["caps"][idx].set_color(outline_c)
                        bp["caps"][idx].set_linewidth(cap_linewidth)

                if k < len(bp.get("fliers", [])):
                    fl = bp["fliers"][k]
                    fl.set_markerfacecolor(outline_c)
                    fl.set_markeredgecolor(outline_c)

    if show_points:
        for g in groups:
            vals = df[g].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if not vals.size:
                continue
            x0 = float(x_to_num[str(g)])
            xs = x0 + (rng.random(vals.size) - 0.5) * jitter_width
            ax.scatter(
                xs,
                vals,
                color=jitter_cmap.get(g, "gray"),
                alpha=jitter_alpha,
                s=jitter_size,
                linewidths=0,
                zorder=3,
            )

    if show_raw_mean_line:
        xs_mean: List[float] = []
        ys_mean: List[float] = []
        for g in groups:
            vals = df[g].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if not vals.size:
                continue
            xs_mean.append(float(x_to_num[str(g)]))
            ys_mean.append(float(np.mean(vals)))
        if ys_mean:
            ax.plot(
                np.array(xs_mean, dtype=float),
                np.array(ys_mean, dtype=float),
                color=raw_mean_line_color,
                lw=raw_mean_line_width,
                ls=raw_mean_line_style,
                zorder=2.9,
                label="Raw mean",
            )

    if vertical_lines is not None:
        for xv in vertical_lines:
            ax.axvline(
                x=xv,
                color=vline_color,
                linestyle=vline_style,
                linewidth=vline_width,
                alpha=vline_alpha,
                zorder=4,
            )
    if horizontal_lines is not None:
        for yv in horizontal_lines:
            ax.axhline(
                y=yv,
                color=hline_color,
                linestyle=hline_style,
                linewidth=hline_width,
                alpha=hline_alpha,
                zorder=4,
            )

    default_x_limits = (-0.5 + x_offset, len(groups) - 0.5 + x_offset)
    x_limits_use = x_limits if x_limits is not None else default_x_limits
    ax.set_xlim(*x_limits_use)
    xtick_pos = [x_to_num[str(g)] for g in groups]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([str(g) for g in groups], fontsize=tick_label_fontsize, rotation=xtick_rotation)
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    for t in ax.get_yticklabels():
        t.set_rotation(ytick_rotation)
    ax.tick_params(labelsize=tick_label_fontsize)

    if grid:
        ax.grid(True, alpha=grid_alpha, linestyle="--", zorder=0)

    ax.spines["top"].set_visible(show_top_right_axes)
    ax.spines["right"].set_visible(show_top_right_axes)

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif all_vals and any(v.size for v in all_vals):
        ymin = float(np.min([np.min(v) for v in all_vals if v.size]))
        if y_min_zero and ymin > 0:
            ymin = 0.0
        ymax = float(np.max([np.max(v) for v in all_vals if v.size]))
        yr = ymax - ymin
        if yr <= 0:
            yr = 1.0

        ymin = ymin - 0.10 * yr
        ymax = ymax + 0.30 * yr

        ax.set_ylim(ymin - 0.10 * yr, ymax + 0.30 * yr)
    y_limits_use = ax.get_ylim()

    if show_brackets and tuk is not None and not tuk.empty:
        cols = list(tuk.columns)
        pair_col = cols[0]
        star_col = next(
            (c for c in reversed(cols) if re.search(r"(summary|p\s*value\s*summary)", str(c), flags=re.I)),
            None,
        )
        if star_col is None and len(cols) >= 2:
            star_col = cols[-2]

        def _norm_label(s) -> str:
            txt = str(s).replace("\xa0", " ")
            txt = txt.replace("–", "-").replace("—", "-")
            return re.sub(r"\s+", " ", txt).strip()

        def _draw_brackets_from_pairs(
            ax,
            pairs: List[Tuple[str, str, str]],
            x_to_num: Dict[str, int],
            *,
            y_limits_local: Optional[Tuple[float, float]],
            y_start: float,
            y_end: float,
            y_step: Optional[float],
            bracket_height_frac: float,
            color: str,
            lw: float,
            text_size: int,
        ):
            if not pairs:
                return
            intervals = []
            for g1, g2, lab in pairs:
                if g1 not in x_to_num or g2 not in x_to_num:
                    continue
                x1, x2 = float(x_to_num[g1]), float(x_to_num[g2])
                if x1 == x2:
                    continue
                lo, hi = (x1, x2) if x1 < x2 else (x2, x1)
                intervals.append({"lo": lo, "hi": hi, "label": lab})
            if not intervals:
                return

            intervals.sort(key=lambda d: (d["lo"], d["hi"]))
            heap: List[Tuple[float, int]] = []
            next_layer_id = 0
            for d in intervals:
                lo, hi = d["lo"], d["hi"]
                if heap and heap[0][0] < lo:
                    _, lid = heapq.heappop(heap)
                else:
                    lid = next_layer_id
                    next_layer_id += 1
                d["layer"] = lid
                heapq.heappush(heap, (hi, lid))
            n_layers = max(d["layer"] for d in intervals) + 1

            if y_limits_local is None:
                y0, y1 = ax.get_ylim()
            else:
                y0, y1 = y_limits_local
            y_range = float(y1 - y0)
            if y_range <= 0:
                return
            tick = y_range * float(bracket_height_frac)
            top = y0 + y_range * float(y_end)
            base = y0 + y_range * float(y_start)
            usable = max(0.0, top - base)
            if y_step is None:
                step = 0.0 if n_layers <= 1 else max(usable / (n_layers - 1), tick * 1.0)
            else:
                step = y_range * abs(float(y_step))
            intervals.sort(key=lambda d: d["layer"])
            for d in intervals:
                lo, hi, lab, layer = d["lo"], d["hi"], d["label"], d["layer"]
                y = max(base, top - layer * step)
                ax.plot([lo, lo, hi, hi], [y, y + tick, y + tick, y], color=color, lw=lw, zorder=5, clip_on=False)
                ax.text(
                    (lo + hi) / 2.0,
                    y + tick - 0.035 * y_range,
                    str(lab),
                    ha="center",
                    va="bottom",
                    fontsize=text_size,
                    color=color,
                    zorder=6,
                    clip_on=False,
                )

        pairs = []
        norm_map = {_norm_label(g): g for g in x_levels}
        for _, row in tuk.iterrows():
            comp = row.get(pair_col, None)
            stars = row.get(star_col, None)
            if comp is None or pd.isna(comp) or stars is None or pd.isna(stars):
                continue
            s = _norm_label(stars)
            if s.lower() == "ns" and hide_ns:
                continue
            txt = _norm_label(comp)
            parts = re.split(r"\s+vs\.?\s+", txt, maxsplit=1, flags=re.I)
            if len(parts) != 2:
                continue
            a, b = _norm_label(parts[0]), _norm_label(parts[1])
            g1 = norm_map.get(a) or (a if a in x_levels else None)
            g2 = norm_map.get(b) or (b if b in x_levels else None)
            if g1 is None or g2 is None:
                continue
            pairs.append((g1, g2, s))

        _draw_brackets_from_pairs(
            ax,
            pairs,
            x_to_num,
            y_limits_local=y_limits_use,
            y_start=y_start,
            y_end=y_end,
            y_step=y_step,
            bracket_height_frac=bracket_height_frac,
            color=bracket_color,
            lw=bracket_linewidth,
            text_size=bracket_text_size,
        )

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=x_label,
        y_label=y_label,
        axis_label_fontsize=axis_label_fontsize,
        title=title,
        title_fontsize=title_fontsize,
        single_x_label=True,
        single_y_label=True,
        font_family=font_family,
    )

    if legend_loc != "none":
        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=mpl.colors.to_rgba("gray", fill_alpha),
                markeredgecolor=box_edge_color,
                label="Box",
            ),
        ]
        if show_raw_mean_line:
            handles.append(
                Line2D([0], [0], color=raw_mean_line_color, lw=raw_mean_line_width, ls=raw_mean_line_style, label="Raw mean")
            )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="gray",
                alpha=jitter_alpha,
                markersize=float(np.sqrt(jitter_size)),
                label="Points",
            )
        )
        if legend_ncol is None:
            legend_ncol = 1

        inside_map = not _is_outside_legend_loc(legend_loc)
        _place_legend(
            fig,
            ax,
            handles,
            [h.get_label() for h in handles],
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_ncol=legend_ncol,
            legend_framealpha=legend_framealpha,
            inside_map=inside_map,
        )

    return fig


def _parse_plate_grid_lines(
    grid_lines: Optional[Union[List, np.ndarray]],
    *,
    ncols: int,
    nrows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse `grid_lines` into (x_lines, y_lines) for plate-style heatmaps."""
    if ncols <= 0 or nrows <= 0:
        raise ValueError(f"ncols and nrows must be positive; got ncols={ncols}, nrows={nrows}.")

    if grid_lines is None:
        x_lines = np.arange(0, ncols + 1, dtype=float)
        y_lines = np.arange(0, nrows + 1, dtype=float)
        return x_lines, y_lines

    if isinstance(grid_lines, (tuple, list)) and len(grid_lines) == 2:
        xg, yg = grid_lines[0], grid_lines[1]
        if isinstance(xg, (list, tuple, np.ndarray)) and isinstance(yg, (list, tuple, np.ndarray)):
            x_lines = np.asarray(xg, dtype=float).ravel()
            y_lines = np.asarray(yg, dtype=float).ravel()
        else:
            arr = np.asarray(grid_lines, dtype=float).ravel()
            x_lines = arr
            y_lines = arr
    else:
        arr = np.asarray(grid_lines, dtype=float).ravel()
        x_lines = arr
        y_lines = arr

    x_lines = np.unique(x_lines[(x_lines >= 0) & (x_lines <= ncols)])
    y_lines = np.unique(y_lines[(y_lines >= 0) & (y_lines <= nrows)])
    return x_lines, y_lines


def plot_well_heatmap_df(
    df_value: pd.DataFrame,
    df_p: Optional[pd.DataFrame] = None,
    *,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    xtick_rotation: float = 0.0,
    ytick_rotation: float = 0.0,
    title: Optional[str] = None,
    font_family: str = "Arial",
    cmap: Union[str, plt.Colormap] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmode: str = "auto",
    dpi: int = 300,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    cbar_label_offset_mm: float = 1.5,
    grid_lines: Optional[Union[List, np.ndarray]] = None,
    gline_color: str = "white",
    gline_width: float = 1.0,
    gline_alpha: float = 0.8,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    colorbar_label: Optional[str] = None,
    colorbar_width_mm: float = 3.0,
    colorbar_pad_mm: float = 1.5,
    show_p: bool = True,
    hide_ns: bool = True,
    p_text_color: str = "black",
    p_text_size: int = 12,
    transparent: bool = False,
) -> plt.Figure:
    """Plot a plate-style (well-grid) heatmap using mm-based `boxsize` geometry."""
    if not isinstance(df_value, pd.DataFrame):
        raise TypeError("df_value must be a pandas DataFrame.")
    if df_value.ndim != 2:
        raise ValueError("df_value must be 2-dimensional.")

    dfv = df_value.copy()
    for c in dfv.columns:
        dfv[c] = pd.to_numeric(dfv[c], errors="coerce")
    Z = dfv.to_numpy(dtype=float)

    nrows, ncols = Z.shape
    if nrows == 0 or ncols == 0:
        raise ValueError("df_value must have at least one row and one column.")

    dfp_aligned: Optional[pd.DataFrame] = None
    if df_p is not None:
        if not isinstance(df_p, pd.DataFrame):
            raise TypeError("df_p must be a pandas DataFrame or None.")

        missing_rows = set(df_value.index) - set(df_p.index)
        missing_cols = set(df_value.columns) - set(df_p.columns)
        if missing_rows or missing_cols:
            raise ValueError(
                "df_p is missing labels present in df_value. "
                f"Missing rows: {sorted(missing_rows)}; missing cols: {sorted(missing_cols)}"
            )

        dfp_aligned = df_p.reindex(index=df_value.index, columns=df_value.columns).copy()
        for c in dfp_aligned.columns:
            dfp_aligned[c] = pd.to_numeric(dfp_aligned[c], errors="coerce")

    single_x_label = bool(x_label)
    single_y_label = bool(y_label)

    fig, axes, layout = _init_box_figure(
        nrows=1,
        ncols=1,
        boxsize_mm=boxsize,
        panel_gap_mm=(0.0, 0.0),
        strip_top_height_mm=0.0,
        strip_right_width_mm=0.0,
        strip_pad_mm=0.0,
        colorbar_width_mm=colorbar_width_mm,
        colorbar_pad_mm=colorbar_pad_mm,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        axis_label_fontsize=float(axis_label_fontsize),
        include_global_label_margins=include_global_label_margins,
        x_label_text=(x_label or ""),
        y_label_text=(y_label or ""),
        colorbar_label_text=colorbar_label,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=cbar_label_offset_mm,
        dpi=dpi,
        font_family=font_family,
        sharex=False,
        sharey=False,
        transparent=transparent,
    )
    ax = axes[0, 0]

    vmin_use, vmax_use = _compute_global_vrange([Z], vmin=vmin, vmax=vmax, vmode=vmode)
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = mpl.colors.Normalize(vmin=vmin_use, vmax=vmax_use)

    Zm = np.ma.masked_invalid(Z)

    x_edges = np.arange(ncols + 1, dtype=float)
    y_edges = np.arange(nrows + 1, dtype=float)

    mappable = ax.pcolormesh(
        x_edges,
        y_edges,
        Zm,
        shading="auto",
        cmap=cmap_obj,
        norm=norm,
        zorder=1,
    )

    x_lines, y_lines = _parse_plate_grid_lines(grid_lines, ncols=ncols, nrows=nrows)
    if x_lines.size or y_lines.size:
        ax.vlines(x_lines, ymin=0, ymax=nrows, colors=gline_color, linewidth=gline_width, alpha=gline_alpha, zorder=2)
        ax.hlines(y_lines, xmin=0, xmax=ncols, colors=gline_color, linewidth=gline_width, alpha=gline_alpha, zorder=2)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()

    x_centers = np.arange(ncols, dtype=float) + 0.5
    y_centers = np.arange(nrows, dtype=float) + 0.5
    ax.set_xticks(x_centers)
    ax.set_yticks(y_centers)

    ax.set_xticklabels([str(c) for c in df_value.columns])
    ax.set_yticklabels([str(r) for r in df_value.index])

    ax.tick_params(labelsize=tick_label_fontsize)
    for lab in ax.get_xticklabels():
        lab.set_rotation(xtick_rotation)
    for lab in ax.get_yticklabels():
        lab.set_rotation(ytick_rotation)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    if show_p and dfp_aligned is not None:
        for i in range(nrows):
            for j in range(ncols):
                if not np.isfinite(Z[i, j]):
                    continue
                stars = p_to_stars(dfp_aligned.iat[i, j])
                if hide_ns and stars == "n.s.":
                    continue
                ax.text(
                    j + 0.5,
                    i + 0.75,
                    stars,
                    ha="center",
                    va="center",
                    color=p_text_color,
                    fontsize=p_text_size,
                    zorder=3,
                )

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=x_label,
        y_label=y_label,
        axis_label_fontsize=float(axis_label_fontsize),
        title=title,
        title_fontsize=float(title_fontsize),
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        font_family=font_family,
    )

    if mappable is not None and colorbar_width_mm > 0:
        right_edge = layout["rect"][2]
        _add_global_colorbar(
            fig,
            layout,
            mappable=mappable,
            right_edge=float(right_edge),
            colorbar_label=colorbar_label,
            tick_label_fontsize=float(tick_label_fontsize),
            axis_label_fontsize=float(axis_label_fontsize),
            font_family=font_family,
        )

    return fig
