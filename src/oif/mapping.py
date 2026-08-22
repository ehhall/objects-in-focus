"""Mapping fixations onto segmented objects.

This is the step the rest of the package exists to support: given where
someone looked and a label map of what is where, decide which object each
fixation landed on, and then summarise looking behaviour per object.

Three assignment rules are available, because the right one depends on how
much you trust the gaze coordinates:

``point``
    The object under the fixation pixel. Fastest, and correct when the
    eye-tracker is well calibrated and objects are large.
``disc``
    The object covering the most area inside a disc of ``radius`` pixels
    around the fixation. Absorbs calibration drift; a fixation just outside a
    small object still lands on it. Ties go to the object closest to the
    fixation centre.
``nearest``
    As ``point``, but a fixation landing on background is reassigned to the
    nearest object within ``radius``. Use when every fixation must be
    attributed to something.

A fixation that stays unassigned gets ``mask_id = 0`` and label
``"background"`` rather than being dropped, so counts always reconcile.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "assign_fixations",
    "map_fixations",
    "object_fixations",
    "fill_objects",
    "object_value_map",
    "rescale_fixations",
    "BACKGROUND_ID",
    "BACKGROUND_LABEL",
]

BACKGROUND_ID = 0
BACKGROUND_LABEL = "background"


def rescale_fixations(df: pd.DataFrame, from_shape: Tuple[int, int],
                      to_shape: Tuple[int, int], mode: str = "stretch") -> pd.DataFrame:
    """Convert fixation coordinates from display space to image space.

    ``mode="stretch"`` scales x and y independently. ``mode="fit"`` preserves
    aspect ratio and accounts for the letterboxing used when a scene is shown
    centred on a larger screen - the case for the 1680x1050 COCO-Freeview
    display.
    """
    out = df.copy()
    fh, fw = from_shape
    th, tw = to_shape
    if mode == "stretch":
        out["x"] = out["x"] * (tw / fw)
        out["y"] = out["y"] * (th / fh)
    elif mode == "fit":
        scale = min(tw / fw, th / fh)
        pad_x = (tw - fw * scale) / 2
        pad_y = (th - fh * scale) / 2
        out["x"] = out["x"] * scale + pad_x
        out["y"] = out["y"] * scale + pad_y
    else:
        raise ValueError("mode must be 'stretch' or 'fit'")
    return out


def _disc_offsets(radius: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = int(radius)
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    d2 = dx ** 2 + dy ** 2
    keep = d2 <= r * r
    return dy[keep], dx[keep], d2[keep]


def assign_fixations(
    x: Sequence[float],
    y: Sequence[float],
    label_map: np.ndarray,
    method: str = "point",
    radius: int = 0,
) -> np.ndarray:
    """Return the object id each ``(x, y)`` fixation belongs to.

    Coordinates are in image pixels with the origin at the top-left, the
    convention used by both the images and the label maps. Fixations off the
    image, or on background, come back as :data:`BACKGROUND_ID`.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    h, w = label_map.shape
    out = np.zeros(xs.shape[0], dtype=label_map.dtype)

    valid = np.isfinite(xs) & np.isfinite(ys)
    xi = np.where(valid, np.round(xs), 0).astype(int)
    yi = np.where(valid, np.round(ys), 0).astype(int)
    inside = valid & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)

    if method == "point" or radius <= 0:
        out[inside] = label_map[yi[inside], xi[inside]]
        if method != "nearest" or radius <= 0:
            return out
    elif method not in {"disc", "nearest"}:
        raise ValueError("method must be 'point', 'disc' or 'nearest'")

    dy, dx, d2 = _disc_offsets(radius)
    order = np.argsort(d2, kind="stable")
    dy, dx, d2 = dy[order], dx[order], d2[order]

    for i in np.where(inside)[0]:
        if method == "nearest" and out[i] != BACKGROUND_ID:
            continue
        yy = yi[i] + dy
        xx = xi[i] + dx
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        vals = label_map[yy[ok], xx[ok]]
        dd = d2[ok]
        nz = vals != 0
        if not nz.any():
            continue
        if method == "nearest":
            out[i] = vals[nz][0]  # offsets are sorted by distance
            continue
        ids, counts = np.unique(vals[nz], return_counts=True)
        best = counts.max()
        tied = ids[counts == best]
        if len(tied) == 1:
            out[i] = tied[0]
        else:  # break ties by whichever object comes closest to the centre
            closest = {int(t): dd[nz][vals[nz] == t].min() for t in tied}
            out[i] = min(closest, key=closest.get)
    return out


def map_fixations(
    fixations: pd.DataFrame,
    label_map: np.ndarray,
    labels: Optional[Mapping[int, str]] = None,
    method: str = "point",
    radius: int = 0,
    keep_background: bool = True,
) -> pd.DataFrame:
    """Attach ``mask_id`` and ``label`` columns to a fixation table.

    Parameters
    ----------
    fixations:
        Canonical fixation table for **one scene** (see :mod:`oif.fixations`).
    label_map:
        Integer label map for that scene.
    labels:
        ``{mask_id: label}``. Without it the ``label`` column is left empty
        except for background.
    method, radius:
        See :func:`assign_fixations`.
    keep_background:
        Keep fixations that landed on no object (default) or drop them.
    """
    out = fixations.copy()
    out["mask_id"] = assign_fixations(out["x"].to_numpy(), out["y"].to_numpy(),
                                      label_map, method=method, radius=radius)
    lookup: Dict[int, str] = {BACKGROUND_ID: BACKGROUND_LABEL}
    if labels:
        lookup.update({int(k): str(v) for k, v in labels.items()})
    out["label"] = out["mask_id"].map(lambda i: lookup.get(int(i), ""))
    if not keep_background:
        out = out[out["mask_id"] != BACKGROUND_ID].reset_index(drop=True)
    return out


def object_fixations(
    mapped: pd.DataFrame,
    label_map: Optional[np.ndarray] = None,
    labels: Optional[Mapping[int, str]] = None,
    scene: Optional[str] = None,
    include_empty: bool = True,
) -> pd.DataFrame:
    """Aggregate mapped fixations to one row per object.

    Objects nobody looked at are included with zero counts when
    ``include_empty`` and a ``label_map`` is given - which matters, because a
    model of how attention is distributed has to explain the objects that got
    none.

    Columns: ``mask_id, label, n_fixations, total_duration, mean_duration,
    n_subjects, first_fix_index``.
    """
    grouped = mapped.groupby("mask_id", dropna=False)
    table = pd.DataFrame({
        "n_fixations": grouped.size(),
        "total_duration": grouped["duration"].sum(min_count=1),
        "mean_duration": grouped["duration"].mean(),
        "n_subjects": grouped["subject"].nunique(),
        "first_fix_index": grouped["fix_index"].min(),
    }).reset_index()

    if include_empty and label_map is not None:
        from .masks import region_ids
        all_ids = pd.DataFrame({"mask_id": region_ids(label_map).astype(table["mask_id"].dtype)})
        table = all_ids.merge(table, on="mask_id", how="left")
        table["n_fixations"] = table["n_fixations"].fillna(0).astype(int)
        table["total_duration"] = table["total_duration"].fillna(0.0)
        table["n_subjects"] = table["n_subjects"].fillna(0).astype(int)

    lookup: Dict[int, str] = {BACKGROUND_ID: BACKGROUND_LABEL}
    if labels:
        lookup.update({int(k): str(v) for k, v in labels.items()})
    table["label"] = table["mask_id"].map(lambda i: lookup.get(int(i), ""))
    if scene is not None:
        table.insert(0, "image", scene)
    return table.sort_values("mask_id").reset_index(drop=True)


def fill_objects(label_map: np.ndarray, values: Mapping[int, float],
                 default: float = 0.0) -> np.ndarray:
    """Paint a per-object value back onto the image.

    Turns a table of numbers per object into a picture you can look at:
    predicted fixations per object, residuals, mean depth, anything keyed by
    ``mask_id``.
    """
    out = np.full(label_map.shape, float(default), dtype=float)
    for mask_id, value in values.items():
        if int(mask_id) == BACKGROUND_ID:
            continue
        out[label_map == int(mask_id)] = float(value)
    return out


def object_value_map(label_map: np.ndarray, pixel_values: np.ndarray,
                     stat: str = "mean") -> Dict[int, float]:
    """Summarise a pixel-level map (salience, depth, fixation density) per object."""
    if pixel_values.shape != label_map.shape:
        raise ValueError(
            f"shape mismatch: values {pixel_values.shape} vs label map {label_map.shape}"
        )
    from .masks import region_ids

    func = {"mean": np.mean, "sum": np.sum, "max": np.max,
            "median": np.median, "min": np.min}[stat]
    out: Dict[int, float] = {}
    for mask_id in region_ids(label_map):
        out[int(mask_id)] = float(func(pixel_values[label_map == mask_id]))
    return out


def fixation_counts_by_object(
    fixations: pd.DataFrame,
    label_maps: Mapping[str, np.ndarray],
    labels: Optional[Mapping[str, Mapping[int, str]]] = None,
    method: str = "point",
    radius: int = 0,
) -> pd.DataFrame:
    """Run the whole mapping over many scenes and stack the results."""
    frames = []
    for scene, frame in fixations.groupby("image"):
        scene = str(scene)
        lm = label_maps.get(scene)
        if lm is None:
            continue
        scene_labels = (labels or {}).get(scene)
        mapped = map_fixations(frame, lm, scene_labels, method=method, radius=radius)
        frames.append(object_fixations(mapped, lm, scene_labels, scene=scene))
    if not frames:
        return pd.DataFrame(columns=["image", "mask_id", "label", "n_fixations"])
    return pd.concat(frames, ignore_index=True)


def iter_object_masks(label_map: np.ndarray) -> Iterable[Tuple[int, np.ndarray]]:
    """Yield ``(mask_id, boolean mask)`` for every object in a label map."""
    from .masks import region_ids
    for mask_id in region_ids(label_map):
        yield int(mask_id), label_map == mask_id
