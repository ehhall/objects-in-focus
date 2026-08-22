"""Monocular depth maps for the OiF scenes.

``depth/target_<scene>_disp.npy`` holds the raw output of MonoDepth2
(Godard et al., 2019) as saved by that model's ``test_simple`` script:
shape ``(1, 1, 320, 1024)``, float32 *disparity*, where larger values mean
**nearer**. Two conversions are needed before the numbers are usable
alongside the masks:

1. squeeze the batch/channel axes and resize 320 -> 768 rows so the map lines
   up with the image and the label map;
2. flip the sense so that larger means **farther**, which is what "depth"
   reads as everywhere else in this package.

:func:`to_depth` does both. It is relative depth, not metres - fine for
ordering objects front-to-back and as a regression predictor, not a claim
about absolute distance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

__all__ = ["load_disparity", "to_depth", "load_depth", "depth_at", "normalize"]

DEFAULT_SHAPE: Tuple[int, int] = (768, 1024)


def load_disparity(path: str | Path) -> np.ndarray:
    """Load a MonoDepth2 ``_disp.npy`` file as a 2-D disparity map."""
    arr = np.load(Path(path))
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D disparity map after squeezing, got {arr.shape}")
    return arr.astype(np.float32)


def _resize(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if arr.shape == tuple(shape):
        return arr
    try:
        import cv2
        h, w = shape
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    except ImportError:  # pragma: no cover - fallback without OpenCV
        from scipy.ndimage import zoom
        return zoom(arr, (shape[0] / arr.shape[0], shape[1] / arr.shape[1]), order=1)


def to_depth(disparity: np.ndarray, shape: Tuple[int, int] = DEFAULT_SHAPE,
             invert: bool = True, rescale: bool = True) -> np.ndarray:
    """Resize a disparity map to image size and flip it to "larger = farther".

    ``rescale`` then maps the result onto ``[0, 1]`` within the scene, which
    is what makes the values safe to ``log1p`` and comparable across scenes.
    MonoDepth2 output is scale-free anyway - only the ordering within a scene
    carries information - so nothing is lost, but the choice is explicit:
    pass ``invert=False, rescale=False`` to get the raw disparity back.
    """
    resized = _resize(np.squeeze(np.asarray(disparity, dtype=np.float32)), shape)
    out = (1.0 - resized) if invert else resized
    return normalize(out) if rescale else out


def load_depth(path: str | Path, shape: Tuple[int, int] = DEFAULT_SHAPE,
               invert: bool = True, rescale: bool = True) -> np.ndarray:
    """``load_disparity`` + :func:`to_depth` in one call."""
    return to_depth(load_disparity(path), shape=shape, invert=invert, rescale=rescale)


def depth_at(depth: np.ndarray, point: Tuple[float, float]) -> float:
    """Sample a depth map at a ``(y, x)`` point, clipped to the array."""
    y, x = point
    y = int(np.clip(round(y), 0, depth.shape[0] - 1))
    x = int(np.clip(round(x), 0, depth.shape[1] - 1))
    return float(depth[y, x])


def normalize(depth: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Rescale a depth map to ``[0, 1]`` over ``mask`` (default: whole map)."""
    vals = depth[mask] if mask is not None else depth
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi == lo:
        return np.zeros_like(depth)
    return (depth - lo) / (hi - lo)
