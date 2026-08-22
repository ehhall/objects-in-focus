"""Salience maps: loading them, summarising them per object.

The attention model takes one salience value per object, read off a
pixel-level salience map. This package does not bundle a salience model -
they are large, they change, and which one you want depends on your claim -
but it makes plugging one in a two-line job.

Where the published numbers came from
-------------------------------------
The OiF analyses used **DeepGaze IIE** (Linardos et al., 2021) with a uniform
centre-bias map, saved one ``.npy`` per scene. The successor, **DeepGaze III**
(Kümmerer, Bethge & Wallis, 2022, https://doi.org/10.1167/jov.22.5.7), also
predicts scanpaths rather than only fixation density, and is the better
starting point for new work; both live at
https://github.com/matthias-k/DeepGaze.

Note for anyone arriving from a web search: the well-known
`mpatacchiola/deepgaze <https://github.com/mpatacchiola/deepgaze>`_ library is
a different project - a friendly OpenCV toolkit for head pose, face and motion
tracking, with its own simple saliency detector. It is a good source of
webcam-side tooling, but it is not the DeepGaze saliency model used here.

Once you have maps, put them anywhere and load them::

    from oif import OiF
    from oif.salience import load_salience_maps

    data = OiF()
    salience = load_salience_maps("salience/", shape=(768, 1024))
    table = data.object_tables(fixations, salience=salience)

If you have no salience maps at all, everything still runs: the ``salience``
column comes back as NaN and the model fits on the remaining predictors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["load_salience_map", "load_salience_maps", "center_bias", "salience_per_object"]


def load_salience_map(path: str | Path, shape: Optional[Tuple[int, int]] = None,
                      normalize: bool = False) -> np.ndarray:
    """Load one salience map (``.npy`` or an image), optionally resized."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.squeeze(np.load(path, allow_pickle=False)).astype(np.float32)
    else:
        from PIL import Image
        with Image.open(path) as im:
            arr = np.array(im.convert("F"), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if shape is not None and arr.shape != tuple(shape):
        import cv2
        h, w = shape
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_CUBIC)
    if normalize:
        total = float(arr.sum())
        if total:
            arr = arr / total
    return arr


def load_salience_maps(folder: str | Path, shape: Optional[Tuple[int, int]] = None,
                       pattern: str = "*.npy", strip: Sequence[str] = (),
                       normalize: bool = False) -> Dict[str, np.ndarray]:
    """Load a folder of salience maps into ``{scene_name: array}``.

    ``strip`` removes suffixes from filenames when deriving the scene name,
    e.g. ``strip=("_deepgaze",)`` maps ``target_bar_deepgaze.npy`` to
    ``target_bar``.
    """
    out: Dict[str, np.ndarray] = {}
    for p in sorted(Path(folder).glob(pattern)):
        name = p.stem
        for suffix in strip:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        out[name] = load_salience_map(p, shape=shape, normalize=normalize)
    return out


def center_bias(shape: Tuple[int, int], sigma: Optional[float] = None) -> np.ndarray:
    """A Gaussian centre-bias map - the baseline every salience model must beat.

    Viewers look at the middle of pictures. A model that only knows this
    already predicts a lot of fixations, so it is the honest floor to compare
    against before claiming a scene-content effect.
    """
    h, w = shape
    sigma = sigma if sigma is not None else 0.25 * max(h, w)
    y = np.arange(h)[:, None] - (h - 1) / 2
    x = np.arange(w)[None, :] - (w - 1) / 2
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def salience_per_object(label_map: np.ndarray, salience: np.ndarray,
                        stat: str = "center") -> Mapping[int, float]:
    """Salience value per object: at its centre of mass, or summarised over it."""
    if stat == "center":
        from .features import object_features
        table = object_features(label_map, salience=salience)
        return dict(zip(table["mask_id"], table["salience"]))
    from .mapping import object_value_map
    return object_value_map(label_map, salience, stat=stat)
