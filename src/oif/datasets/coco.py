"""COCO-Freeview adapter: the same analysis on MS-COCO scenes.

COCO-Freeview (Chen et al., 2022) recorded free-viewing eye movements over
MS-COCO images, and COCO-Stuff (Caesar et al., 2018) supplies dense
thing-and-stuff segmentations for the same pictures. Together they make a
second, much larger testbed for object-based attention - which is how the
OiF model was checked for generalisation.

Two differences from OiF have to be handled explicitly, and this module
exists to handle them:

**One id covers every instance of a category.** COCO-Stuff labels all chairs
"chair". Object-level measures need instances, so each category mask is
split into spatially separate blobs with :func:`oif.masks.split_components`.

**Images were shown letterboxed on a 1680 x 1050 display.** Fixations are in
screen coordinates, segmentations in image coordinates.
:func:`fit_to_display` reproduces the presentation geometry (scale to the
display height, centre, crop the overflow) so the two line up.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import features as feat_mod
from .. import mapping as map_mod
from .. import masks as mask_mod

__all__ = ["COCOFreeview", "COCOScene", "fit_to_display", "STUFF_BACKGROUND_IDS"]

DISPLAY_SHAPE: Tuple[int, int] = (1050, 1680)

#: COCO-Stuff ids treated as background: amorphous stuff regions plus the
#: "other"/unlabelled classes. Everything else counts as an object.
STUFF_BACKGROUND_IDS: Tuple[int, ...] = (
    101, 102, 103, 114, 115, 116, 117, 118, 120, 124, 126, 140, 149, 157,
    171, 172, 173, 174, 175, 176, 177, 178, 183,
)


def fit_to_display(image: np.ndarray, shape: Tuple[int, int] = DISPLAY_SHAPE,
                   interpolation: Optional[int] = None) -> np.ndarray:
    """Place an image on the presentation canvas exactly as viewers saw it.

    Scales to the canvas height preserving aspect ratio, centres horizontally,
    crops anything wider than the canvas and pads anything narrower with
    black. Use nearest-neighbour (the default for integer input) when the
    "image" is a label map, or ids will be interpolated into nonsense.
    """
    import cv2

    arr = np.asarray(image)
    is_labels = np.issubdtype(arr.dtype, np.integer)
    if interpolation is None:
        interpolation = cv2.INTER_NEAREST if is_labels else cv2.INTER_LANCZOS4

    h, w = shape
    scale = h / arr.shape[0]
    new_w = max(int(round(arr.shape[1] * scale)), 1)
    resized = cv2.resize(arr, (new_w, h), interpolation=interpolation)

    if new_w > w:
        excess = new_w - w
        left = excess // 2
        resized = resized[:, left:left + w]
        new_w = w

    if resized.ndim == 3:
        canvas = np.zeros((h, w, resized.shape[2]), dtype=arr.dtype)
    else:
        canvas = np.zeros((h, w), dtype=arr.dtype)
    start = (w - new_w) // 2
    canvas[:, start:start + new_w] = resized
    return canvas


def instance_label_map(
    stuff_map: np.ndarray,
    background_ids: Sequence[int] = STUFF_BACKGROUND_IDS,
    min_fraction: float = 0.05,
    downscale: Tuple[int, int] = (35, 56),
    min_blob: int = 4,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Split a COCO-Stuff category map into per-instance objects.

    Returns the new label map and ``{new_id: original_category_id}`` so the
    category of every instance stays recoverable.

    ``min_fraction`` drops categories covering less than that percentage of
    the image before splitting; ``downscale`` is the working resolution for
    the connected-component pass (the published analysis used 1/30th scale,
    35 x 56 for the 1050 x 1680 canvas), and ``min_blob`` absorbs blobs
    smaller than that many cells at the working resolution.
    """
    bg = set(int(b) for b in background_ids)
    out = np.zeros(stuff_map.shape, dtype=np.int32)
    origin: Dict[int, int] = {}
    next_id = 1

    for cat in mask_mod.region_ids(stuff_map, min_fraction=min_fraction):
        cat = int(cat)
        if cat in bg:
            continue
        mask = stuff_map == cat
        blobs = mask_mod.split_components(mask, min_size=min_blob, connectivity=4,
                                          downscale=downscale)
        for blob in np.unique(blobs):
            if blob == 0:
                continue
            region = blobs == blob
            if not region.any():
                continue
            out[region] = next_id
            origin[next_id] = cat
            next_id += 1
    return out, origin


@dataclass
class COCOScene:
    """One COCO image with its instance label map."""

    name: str
    root: Path
    category_names: Mapping[int, str]
    display_shape: Tuple[int, int] = DISPLAY_SHAPE
    images_dir: str = "images"
    stuff_dir: str = "stuff"
    depth_dir: str = "depth"

    @cached_property
    def image(self) -> np.ndarray:
        from PIL import Image
        for ext in (".jpg", ".png", ".jpeg"):
            p = self.root / self.images_dir / f"{self.name}{ext}"
            if p.exists():
                with Image.open(p) as im:
                    return fit_to_display(np.array(im.convert("RGB")), self.display_shape)
        raise FileNotFoundError(f"no image for {self.name} in {self.root / self.images_dir}")

    @cached_property
    def stuff_map(self) -> np.ndarray:
        """COCO-Stuff category map, placed on the presentation canvas."""
        from PIL import Image
        p_npy = self.root / self.stuff_dir / f"{self.name}.npy"
        if p_npy.exists():
            arr = np.load(p_npy, allow_pickle=False)
        else:
            p_png = self.root / self.stuff_dir / f"{self.name}.png"
            with Image.open(p_png) as im:
                arr = np.array(im)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.shape != tuple(self.display_shape):
            arr = fit_to_display(arr.astype(np.int32), self.display_shape)
        return arr.astype(np.int32)

    @cached_property
    def _instances(self) -> Tuple[np.ndarray, Dict[int, int]]:
        return instance_label_map(self.stuff_map)

    @property
    def label_map(self) -> np.ndarray:
        return self._instances[0]

    @cached_property
    def labels(self) -> Dict[int, str]:
        return {mask_id: str(self.category_names.get(cat, cat))
                for mask_id, cat in self._instances[1].items()}

    @cached_property
    def depth(self) -> Optional[np.ndarray]:
        p = self.root / self.depth_dir / f"{self.name}.npy"
        if not p.exists():
            return None
        arr = np.squeeze(np.load(p, allow_pickle=False))
        if arr.shape != tuple(self.display_shape):
            import cv2
            h, w = self.display_shape
            arr = cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        return arr

    def features(self, salience: Optional[np.ndarray] = None, **kwargs) -> pd.DataFrame:
        return feat_mod.object_features(self.label_map, depth=self.depth,
                                        salience=salience, labels=self.labels,
                                        scene=self.name, **kwargs)

    def map_fixations(self, fixations: pd.DataFrame, method: str = "point",
                      radius: int = 0) -> pd.DataFrame:
        frame = fixations
        if "image" in frame.columns:
            frame = frame[frame["image"].astype(str) == self.name]
        return map_mod.map_fixations(frame, self.label_map, self.labels,
                                     method=method, radius=radius)


class COCOFreeview:
    """A COCO-Freeview data tree laid out like the OiF one.

    Expected layout::

        <root>/images/<image_id>.jpg
        <root>/stuff/<image_id>.png     COCO-Stuff label map
        <root>/depth/<image_id>.npy     optional
        <root>/labels.txt               "id: name" per line (COCO-Stuff labels)

    ``labels.txt`` is the standard COCO-Stuff label file; the loader also
    accepts a CSV with ``id`` and ``label`` columns.
    """

    def __init__(self, root: str | Path, display_shape: Tuple[int, int] = DISPLAY_SHAPE):
        self.root = Path(root)
        self.display_shape = display_shape

    @cached_property
    def category_names(self) -> Dict[int, str]:
        for name in ("labels.txt", "labels.csv"):
            p = self.root / name
            if not p.exists():
                continue
            sep = ":" if name.endswith(".txt") else ","
            df = pd.read_csv(p, sep=sep, header=None if name.endswith(".txt") else "infer",
                             names=["id", "label"] if name.endswith(".txt") else None,
                             skipinitialspace=True, engine="python")
            return {int(r.id): str(r.label).strip() for r in df.itertuples()}
        return {}

    @cached_property
    def image_ids(self) -> List[str]:
        d = self.root / "images"
        return sorted(p.stem for p in d.glob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"})

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, key: str | int) -> COCOScene:
        name = self.image_ids[key] if isinstance(key, int) else str(key)
        return COCOScene(name, self.root, self.category_names, self.display_shape)

    def __iter__(self) -> Iterator[COCOScene]:
        for name in self.image_ids:
            yield self[name]

    def object_tables(self, fixations: Optional[pd.DataFrame] = None,
                      method: str = "point", radius: int = 0,
                      progress: bool = False) -> pd.DataFrame:
        frames = []
        for scene in self:
            if progress:
                print(f"  {scene.name}", flush=True)
            table = scene.features()
            if fixations is not None:
                mapped = scene.map_fixations(fixations, method=method, radius=radius)
                counts = map_mod.object_fixations(mapped, scene.label_map, scene.labels,
                                                  scene=scene.name)
                counts = counts.drop(columns=["label", "image"], errors="ignore")
                table = table.merge(counts, on="mask_id", how="left").fillna(
                    {"n_fixations": 0, "total_duration": 0.0, "n_subjects": 0})
            frames.append(table)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def __repr__(self) -> str:  # pragma: no cover
        return f"COCOFreeview({str(self.root)!r}, {len(self)} images)"
