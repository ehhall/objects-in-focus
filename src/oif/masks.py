"""Turning polygons into label maps, and label maps back into labelled objects.

A *label map* is an integer array the size of the scene in which every pixel
carries the id of the object visible at that pixel, 0 meaning "no annotated
object". Ids are unique within a scene.

Two directions are supported:

``build_label_map``
    polygons + depth -> label map. Objects are painted back-to-front so that
    nearer objects overwrite farther ones, which is what makes a fixation
    land on the thing a viewer actually saw rather than on whatever happens
    to be drawn last.

``recover_labels``
    label map + polygons -> id/label table. The published ``masks/`` arrays
    carry no record of which id is which object; this peels the painting
    order back off to recover it, and reports a match score per region so you
    can see how sure the recovery is.

Rasterisation uses OpenCV's even-odd polygon fill, matching the arrays that
ship with the dataset exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .annotations import BACKGROUND_LABELS, Polygon, SceneAnnotation

__all__ = [
    "rasterize",
    "polygon_masks",
    "build_label_map",
    "recover_labels",
    "RecoveredObject",
    "resize_label_map",
    "split_components",
    "region_ids",
    "region_areas",
    "centroids",
]


def _cv2():
    try:
        import cv2  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenCV is required for polygon rasterisation. "
            "Install it with: pip install 'oif[full]' or pip install opencv-python-headless"
        ) from exc
    return cv2


def rasterize(polygon: Polygon, shape: Tuple[int, int], value: int = 1,
              out: Optional[np.ndarray] = None) -> np.ndarray:
    """Fill one polygon into an integer array of ``shape`` = ``(h, w)``."""
    cv2 = _cv2()
    canvas = np.zeros(shape, dtype=np.int32) if out is None else out
    # Truncate rather than round: this is what produced the published masks,
    # and matching it keeps rebuilt and shipped arrays pixel-identical.
    pts = polygon.closed().astype(np.int32)[None, :, :]
    return cv2.fillPoly(canvas, pts, int(value))


def polygon_masks(polygons: Sequence[Polygon], shape: Tuple[int, int]) -> List[np.ndarray]:
    """Boolean mask per polygon, ignoring occlusion by other objects."""
    return [rasterize(p, shape).astype(bool) for p in polygons]


# --------------------------------------------------------------------------
# polygons -> label map
# --------------------------------------------------------------------------

def depth_order(polygons: Sequence[Polygon], depth: np.ndarray,
                shape: Optional[Tuple[int, int]] = None,
                group_by_occlusion: bool = True) -> List[int]:
    """Indices of ``polygons`` ordered far-to-near, for back-to-front painting.

    ``depth`` is a metric depth map at image resolution (see
    :func:`oif.depth.to_depth`): larger means farther away. Each object is
    scored by its mean depth over its own outline.

    When ``group_by_occlusion`` is true, objects the annotator marked
    ``occluded`` are painted before unoccluded ones regardless of their depth
    score. That reproduces the ordering rule used to build the published
    masks: an object someone flagged as partly hidden must not end up on top
    of the object hiding it, even if the depth estimate disagrees.
    """
    shape = shape or depth.shape[:2]
    scores, occ = [], []
    for p in polygons:
        m = rasterize(p, shape).astype(bool)
        scores.append(float(depth[m].mean()) if m.any() else np.inf)
        occ.append(int(p.occluded))
    order = np.argsort(-np.asarray(scores), kind="stable")  # far -> near
    if not group_by_occlusion:
        return [int(i) for i in order]
    occluded = [int(i) for i in order if occ[i] == 1]
    unoccluded = [int(i) for i in order if occ[i] != 1]
    return occluded + unoccluded


def build_label_map(
    annotation: SceneAnnotation,
    depth: Optional[np.ndarray] = None,
    background: Sequence[str] = BACKGROUND_LABELS,
    include_background: bool = False,
    order: Optional[Sequence[int]] = None,
    keep_hidden: bool = True,
    dtype=np.int32,
) -> Tuple[np.ndarray, List[Polygon]]:
    """Paint a scene's polygons into a label map.

    Parameters
    ----------
    annotation:
        Parsed scene annotation.
    depth:
        Depth map at image resolution used to order the painting. If omitted,
        polygons are painted in annotation order (largest z_order last).
    background / include_background:
        Whether to paint surfaces like ``sky`` and ``floor``.
    order:
        Explicit painting order (indices into the selected polygon list),
        overriding ``depth``.
    keep_hidden:
        If an object ends up with no visible pixels because nearer objects
        cover it completely, repaint it on top so that every object has an id
        in the map. Set false to let fully hidden objects disappear.

    Returns
    -------
    (label_map, polygons)
        ``label_map[y, x]`` holds ``i + 1`` for the i-th entry of the returned
        polygon list, so ``polygons[label_map[y, x] - 1]`` is the object at
        that pixel.
    """
    polys = list(annotation.polygons) if include_background else annotation.objects(background)
    shape = annotation.shape

    if order is not None:
        idx = list(order)
    elif depth is not None:
        idx = depth_order(polys, depth, shape)
    else:
        idx = sorted(range(len(polys)), key=lambda i: polys[i].z_order)

    ordered = [polys[i] for i in idx]
    label_map = np.zeros(shape, dtype=dtype)
    for new_id, poly in enumerate(ordered, start=1):
        m = rasterize(poly, shape).astype(bool)
        label_map[m] = new_id

    if keep_hidden:
        for new_id, poly in enumerate(ordered, start=1):
            if not np.any(label_map == new_id):
                label_map[rasterize(poly, shape).astype(bool)] = new_id

    return label_map, ordered


# --------------------------------------------------------------------------
# label map -> labels
# --------------------------------------------------------------------------

@dataclass
class RecoveredObject:
    """One region of a label map matched back to its source polygon."""

    mask_id: int
    label: str
    occluded: int
    polygon_index: int
    score: float
    area: int
    visible_fraction: float

    @property
    def confident(self) -> bool:
        return self.score >= 0.99


def recover_labels(
    label_map: np.ndarray,
    annotation: SceneAnnotation,
    include_background: bool | str = "auto",
    background: Sequence[str] = BACKGROUND_LABELS,
) -> List[RecoveredObject]:
    """Recover which polygon each id in ``label_map`` came from.

    The published masks were painted back-to-front, so the highest id was
    painted last and its region is its whole polygon. Working down from the
    highest id and marking off the pixels already claimed, each id's region
    should coincide with exactly one polygon minus the area painted over it.
    The routine matches greedily on that basis and returns an IoU-style
    ``score`` per region: 1.0 means the region and the still-unclaimed part
    of the polygon agree pixel for pixel.

    Regions scoring below 0.99 deserve a look before you trust their label -
    filter with ``[r for r in recovered if r.confident]``.

    ``include_background="auto"`` (the default) tries both with and without
    background surfaces as candidates and keeps whichever explains the label
    map better. Some scenes' masks include the sky and the floor as objects
    and some do not, and guessing wrong costs accuracy on every region.
    """
    if include_background == "auto":
        with_bg = _recover(label_map, list(annotation.polygons))
        without_bg = _recover(label_map, annotation.objects(background))
        return max((with_bg, without_bg),
                   key=lambda recs: float(np.mean([r.score for r in recs])) if recs else -1.0)

    polys = list(annotation.polygons) if include_background else annotation.objects(background)
    return _recover(label_map, polys)


def _recover(label_map: np.ndarray, polys: Sequence[Polygon]) -> List[RecoveredObject]:
    """Greedy reverse-paint-order matching of label map regions to polygons."""
    shape = label_map.shape
    filled = [rasterize(p, shape).astype(bool) for p in polys]

    ids = [int(i) for i in np.unique(label_map) if i != 0]
    claimed = np.zeros(shape, dtype=bool)
    used: set[int] = set()
    out: List[RecoveredObject] = []

    for mask_id in sorted(ids, reverse=True):
        region = label_map == mask_id
        best_j, best_score = -1, -1.0
        for j, f in enumerate(filled):
            if j in used:
                continue
            visible = f & ~claimed
            if not visible.any():
                continue
            union = (visible | region).sum()
            score = float((visible & region).sum() / union) if union else 0.0
            if score > best_score:
                best_j, best_score = j, score
        if best_j < 0:
            out.append(RecoveredObject(mask_id, "", -1, -1, 0.0, int(region.sum()), 0.0))
            continue
        used.add(best_j)
        claimed |= region
        poly = polys[best_j]
        full = int(filled[best_j].sum())
        area = int(region.sum())
        out.append(
            RecoveredObject(
                mask_id=mask_id,
                label=poly.label,
                occluded=int(poly.occluded),
                polygon_index=int(poly.index),
                score=best_score,
                area=area,
                visible_fraction=(area / full) if full else 0.0,
            )
        )

    return sorted(out, key=lambda r: r.mask_id)


# --------------------------------------------------------------------------
# label map utilities
# --------------------------------------------------------------------------

def region_ids(label_map: np.ndarray, min_fraction: float = 0.0) -> np.ndarray:
    """Non-zero ids in a label map, optionally dropping tiny regions.

    ``min_fraction`` is a percentage of total image pixels, matching the
    0.05% cut used for the COCO-Freeview objects in the paper.
    """
    ids, counts = np.unique(label_map, return_counts=True)
    keep = ids != 0
    if min_fraction > 0:
        keep &= (counts / label_map.size) * 100 > min_fraction
    return ids[keep]


def region_areas(label_map: np.ndarray) -> Dict[int, int]:
    ids, counts = np.unique(label_map, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts) if i != 0}


def centroids(label_map: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """Centre of mass ``(y, x)`` of every region, in pixels."""
    from scipy.ndimage import center_of_mass

    ids = region_ids(label_map)
    if len(ids) == 0:
        return {}
    coms = center_of_mass(np.ones_like(label_map, dtype=float), labels=label_map, index=ids)
    return {int(i): (float(c[0]), float(c[1])) for i, c in zip(ids, coms)}


def resize_label_map(label_map: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resize a label map with nearest-neighbour so ids stay ids."""
    cv2 = _cv2()
    h, w = shape
    return cv2.resize(label_map.astype(np.int32), (w, h),
                      interpolation=cv2.INTER_NEAREST).astype(label_map.dtype)


def split_components(
    mask: np.ndarray,
    min_size: int = 1,
    connectivity: int = 4,
    downscale: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Split one object's mask into spatially separate blobs.

    Datasets that share a single id across every instance of a category
    (COCO-Stuff does this) need splitting before object-level measures mean
    anything: four chairs under one "chair" id are four objects, not one.

    ``min_size`` blobs are absorbed into the neighbouring blob they touch
    most, so speckle does not become objects. ``downscale`` runs the analysis
    at a coarser ``(h, w)`` first - the paper used 1/30th scale - which both
    speeds it up and stops single-pixel bridges from fusing separate blobs.
    """
    from scipy.ndimage import label as ndi_label
    from scipy.ndimage import zoom

    binary = np.asarray(mask).astype(bool)
    original_shape = binary.shape
    if downscale is not None:
        factors = (downscale[0] / binary.shape[0], downscale[1] / binary.shape[1])
        work = zoom(binary, factors, order=0)
    else:
        work = binary

    structure = (np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) if connectivity == 4
                 else np.ones((3, 3), int))
    labels, _ = ndi_label(work, structure=structure)
    if min_size > 1:
        labels = _absorb_small(labels, min_size)
    if downscale is not None:
        labels = resize_label_map(labels.astype(np.int32), original_shape)
        labels = labels * binary  # never leak outside the original mask
    return labels.astype(np.int32)


def _absorb_small(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Merge blobs under ``min_size`` px into their most common neighbour.

    A small blob touching nothing is speckle - a few stray pixels of a
    category with no body nearby - and is dropped rather than promoted to an
    object of its own.
    """
    from collections import Counter

    sizes = np.bincount(labels.ravel())
    small = {int(i) for i in np.where((sizes < min_size) & (np.arange(len(sizes)) > 0))[0]}
    if not small:
        return labels

    out = labels.copy()
    padded = np.pad(labels, 1)
    for lab in small:
        ys, xs = np.where(labels == lab)
        neighbours: Counter = Counter()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                vals = padded[ys + 1 + dy, xs + 1 + dx]
                for v in vals[(vals > 0)]:
                    v = int(v)
                    if v != lab and v not in small:
                        neighbours[v] += 1
        out[labels == lab] = neighbours.most_common(1)[0][0] if neighbours else 0
    return out
