"""Per-object spatial features.

The object-based attention model in Hall & Loh (2025) predicts how many
fixations an object receives from four properties of that object:

============  =========================================================
feature       definition
============  =========================================================
``size``      visible pixel count
``ecc``       Euclidean distance from the object's centre of mass to the
              centre of the image, in pixels
``depth``     depth sampled at the centre of mass (larger = farther)
``salience``  salience-model output sampled at the centre of mass
============  =========================================================

:func:`object_features` computes all four for every object in a scene and
returns them as a tidy table, ready to join to fixation counts and hand to
:class:`oif.model.ObjectAttentionModel`.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .masks import region_ids

__all__ = [
    "object_features",
    "eccentricity",
    "add_model_terms",
    "FEATURE_COLUMNS",
]

FEATURE_COLUMNS: Tuple[str, ...] = ("size", "ecc", "depth", "salience")


def eccentricity(point: Tuple[float, float], shape: Tuple[int, int]) -> float:
    """Distance in pixels from a ``(y, x)`` point to the centre of the image."""
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    return float(np.hypot(point[0] - cy, point[1] - cx))


def object_features(
    label_map: np.ndarray,
    depth: Optional[np.ndarray] = None,
    salience: Optional[np.ndarray] = None,
    labels: Optional[Mapping[int, str]] = None,
    scene: Optional[str] = None,
    min_fraction: float = 0.0,
    depth_stat: str = "center",
    salience_stat: str = "center",
) -> pd.DataFrame:
    """Measure every object in a label map.

    Parameters
    ----------
    label_map:
        Integer label map for one scene.
    depth, salience:
        Optional pixel maps at the same shape. Missing maps leave their
        columns as NaN rather than failing, so you can build the table before
        you have run a salience model.
    labels:
        ``{mask_id: label}`` to name the objects.
    min_fraction:
        Drop objects covering less than this percentage of the image. The
        published COCO analysis used 0.05.
    depth_stat, salience_stat:
        ``"center"`` samples at the object's centre of mass (what the paper
        did); ``"mean"``, ``"median"``, ``"max"``, ``"min"`` and ``"sum"``
        summarise over the whole object instead.

    Returns
    -------
    DataFrame with one row per object:
    ``image, mask_id, label, size, area_fraction, centroid_x, centroid_y,
    ecc, depth, salience``.
    """
    h, w = label_map.shape
    for name, arr in (("depth", depth), ("salience", salience)):
        if arr is not None and arr.shape != label_map.shape:
            raise ValueError(
                f"{name} map has shape {arr.shape}, label map has {label_map.shape}; "
                "resize one to match (see oif.depth.to_depth / oif.masks.resize_label_map)"
            )

    ids = region_ids(label_map, min_fraction=min_fraction)
    rows = []
    for mask_id in ids:
        mask = label_map == mask_id
        n = int(mask.sum())
        ys, xs = np.nonzero(mask)
        cy, cx = float(ys.mean()), float(xs.mean())
        row: Dict[str, object] = {
            "mask_id": int(mask_id),
            "label": (labels or {}).get(int(mask_id), ""),
            "size": n,
            "area_fraction": n / label_map.size,
            "centroid_x": cx,
            "centroid_y": cy,
            "ecc": eccentricity((cy, cx), (h, w)),
            "depth": np.nan,
            "salience": np.nan,
        }
        for key, arr, stat in (("depth", depth, depth_stat),
                               ("salience", salience, salience_stat)):
            if arr is None:
                continue
            row[key] = _sample(arr, mask, (cy, cx), stat)
        rows.append(row)

    table = pd.DataFrame(rows, columns=["mask_id", "label", "size", "area_fraction",
                                        "centroid_x", "centroid_y", "ecc",
                                        "depth", "salience"])
    if scene is not None:
        table.insert(0, "image", scene)
    return table


def _sample(arr: np.ndarray, mask: np.ndarray, centroid: Tuple[float, float],
            stat: str) -> float:
    if stat == "center":
        y = int(np.clip(round(centroid[0]), 0, arr.shape[0] - 1))
        x = int(np.clip(round(centroid[1]), 0, arr.shape[1] - 1))
        return float(arr[y, x])
    func = {"mean": np.mean, "median": np.median, "max": np.max,
            "min": np.min, "sum": np.sum}[stat]
    return float(func(arr[mask]))


def add_model_terms(
    df: pd.DataFrame,
    outcome: str = "n_fixations",
    size: str = "size",
    depth: str = "depth",
    ecc: str = "ecc",
    salience: str = "salience",
    by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Add the transformed predictors the attention model is fitted on.

    Sizes, depths and fixation counts are all heavily right-skewed, so they
    enter the model as ``log1p``; eccentricity and salience are z-scored.
    Pass ``by=["dataset"]`` to z-score within groups when pooling datasets
    whose salience maps are not on a common scale.
    """
    out = df.copy()
    if outcome in out.columns:
        out["log_sum"] = np.log1p(out[outcome].astype(float))
    out["log_size"] = np.log1p(out[size].astype(float))
    if depth in out.columns:
        out["log_depth"] = np.log1p(out[depth].astype(float))

    def _z(col: str, name: str) -> None:
        if col not in out.columns:
            return
        values = out[col].astype(float)
        if by:
            out[name] = values.groupby([out[b] for b in by]).transform(
                lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) else 0.0)
        else:
            sd = values.std(ddof=0)
            out[name] = (values - values.mean()) / sd if sd else 0.0

    _z(ecc, "z_ecc")
    _z(salience, "z_salience")
    return out
