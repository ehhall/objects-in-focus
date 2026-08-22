"""Figures: outlines, per-object heatmaps, fixations on scenes.

Everything here returns a matplotlib ``Axes`` or a numpy image, so plots
compose into whatever figure you are building rather than dictating one.
Import is lazy: the package works without matplotlib installed, and only
these functions require it.
"""

from __future__ import annotations

import colorsys
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "distinct_colors",
    "outline_objects",
    "overlay_labels",
    "show_scene",
    "show_objects",
    "show_fixations",
    "show_object_values",
    "scene_grid",
]


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotting needs matplotlib: pip install 'oif[full]'"
        ) from exc
    return plt


def distinct_colors(n: int, as_float: bool = False) -> list:
    """``n`` visually separable RGB colours, spaced by the golden angle."""
    out = []
    for i in range(max(n, 1)):
        hue = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        out.append((r, g, b) if as_float else (int(r * 255), int(g * 255), int(b * 255)))
    return out


def outline_objects(image: np.ndarray, label_map: np.ndarray, thickness: int = 2,
                    ids: Optional[Sequence[int]] = None,
                    color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """Draw each object's boundary onto a copy of ``image`` in RGB."""
    import cv2

    from .masks import region_ids

    img = np.asarray(image)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    result = np.ascontiguousarray(img[..., :3].copy())
    if result.dtype != np.uint8:
        result = (255 * np.clip(result, 0, 1)).astype(np.uint8)

    all_ids = list(ids) if ids is not None else [int(i) for i in region_ids(label_map)]
    colors = distinct_colors(len(all_ids))
    for i, mask_id in enumerate(all_ids):
        binary = (label_map == mask_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color or colors[i], thickness)
    return result


def overlay_labels(ax, label_map: np.ndarray, labels: Mapping[int, str],
                   fontsize: int = 7, color: str = "white") -> None:
    """Write each object's label at its centre of mass."""
    from .masks import region_ids

    for mask_id in region_ids(label_map):
        name = labels.get(int(mask_id))
        if not name:
            continue
        ys, xs = np.nonzero(label_map == mask_id)
        ax.text(xs.mean(), ys.mean(), name, fontsize=fontsize, color=color,
                ha="center", va="center",
                path_effects=_stroke())


def _stroke():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=2, foreground="black")]


def show_scene(image: np.ndarray, ax=None, title: Optional[str] = None):
    plt = _plt()
    ax = ax or plt.subplots(figsize=(8, 6))[1]
    ax.imshow(image)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


def show_objects(image: np.ndarray, label_map: np.ndarray, ax=None,
                 thickness: int = 2, labels: Optional[Mapping[int, str]] = None,
                 title: Optional[str] = None):
    """Scene with object outlines, optionally annotated with labels."""
    ax = show_scene(outline_objects(image, label_map, thickness), ax=ax, title=title)
    if labels:
        overlay_labels(ax, label_map, labels)
    return ax


def show_fixations(image: np.ndarray, fixations, ax=None, size: float = 30,
                   color: str = "#2d7dd2", edge: str = "white", alpha: float = 0.85,
                   connect: bool = False, title: Optional[str] = None):
    """Scatter fixations over a scene, optionally joined into a scanpath."""
    plt = _plt()
    ax = show_scene(image, ax=ax, title=title)
    x = np.asarray(fixations["x"], dtype=float)
    y = np.asarray(fixations["y"], dtype=float)
    if connect:
        ax.plot(x, y, "-", color=color, linewidth=1, alpha=0.6, zorder=2)
    dur = fixations["duration"] if "duration" in getattr(fixations, "columns", []) else None
    if dur is not None and not np.isfinite(np.asarray(dur, float)).any():
        dur = None  # pooled fixation maps carry no durations
    s = size if dur is None else np.clip(np.asarray(dur, float) / 10.0, 5, 400)
    ax.scatter(x, y, s=s, c=color, edgecolors=edge, linewidths=0.8, alpha=alpha, zorder=3)
    _ = plt  # keep the lazy import meaningful for linters
    return ax


def show_object_values(label_map: np.ndarray, values: Mapping[int, float], ax=None,
                       cmap: str = "magma", title: Optional[str] = None,
                       colorbar: bool = True, label: Optional[str] = None):
    """Fill each object with a value and show it as a map."""
    plt = _plt()
    from .mapping import fill_objects

    ax = ax or plt.subplots(figsize=(8, 6))[1]
    filled = fill_objects(label_map, values, default=np.nan)
    im = ax.imshow(np.ma.masked_invalid(filled), cmap=cmap)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        if label:
            cb.set_label(label)
    return ax


def show_density(density: np.ndarray, ax=None, cmap: str = "gist_heat",
                 title: Optional[str] = None):
    """Show a blurred fixation density map."""
    plt = _plt()
    ax = ax or plt.subplots(figsize=(8, 6))[1]
    ax.imshow(density, cmap=cmap, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


def scene_grid(images: Iterable[np.ndarray], rows: int = 2, cols: int = 5,
               titles: Optional[Sequence[str]] = None, figsize: Tuple[float, float] = (15, 5),
               pad: float = 0.01):
    """Tile scenes into a tight grid - the dataset's contact sheet."""
    plt = _plt()
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_axis_off()
    for ax, img, idx in zip(axes, images, range(rows * cols)):
        ax.imshow(img)
        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx], fontsize=8)
    fig.subplots_adjust(wspace=pad, hspace=pad, left=0, right=1, top=1, bottom=0)
    return fig, axes
