"""The Objects in Focus dataset: 100 fully segmented real-world scenes.

Fifty indoor and fifty outdoor photographs, each 1024 x 768, with every
object hand-segmented in CVAT - not just the foreground, and not only
categories from a fixed vocabulary. That is what makes the set useful for
object-based attention: when a fixation lands somewhere, there is an
annotated object under it.

Typical use::

    from oif import OiF

    data = OiF("path/to/objects-in-focus")
    scene = data["target_kitchen_IDS01"]

    scene.image        # (768, 1024, 3) uint8
    scene.label_map    # (768, 1024) int32, one id per object
    scene.labels       # {mask_id: "sink", ...}
    scene.depth        # (768, 1024) float, larger = farther

    looked_at = scene.map_fixations(my_fixations, radius=25)
    per_object = scene.object_table(my_fixations)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import annotations as ann_mod
from .. import depth as depth_mod
from .. import features as feat_mod
from .. import fixations as fix_mod
from .. import mapping as map_mod
from .. import masks as mask_mod
from ..paths import DataRoot

__all__ = ["OiF", "Scene"]

LABELS_FILENAME = "labels.csv"
REBUILT_FILENAME = "rebuilt_labels.csv"


@dataclass
class Scene:
    """One scene and everything that lines up with it."""

    name: str
    root: DataRoot

    # -- raw arrays --------------------------------------------------------
    @cached_property
    def image(self) -> np.ndarray:
        from PIL import Image
        with Image.open(self.root.path("images", self.name)) as im:
            return np.array(im.convert("RGB"))

    @cached_property
    def annotation(self) -> ann_mod.SceneAnnotation:
        return ann_mod.read_annotation(self.root.path("annotations", self.name), scene=self.name)

    @cached_property
    def depth(self) -> np.ndarray:
        """Depth at image resolution, larger = farther."""
        return depth_mod.load_depth(self.root.path("depth", self.name, "_disp.npy"),
                                    shape=self.shape)

    @cached_property
    def label_map(self) -> np.ndarray:
        """Object id per pixel.

        Reads ``masks/<scene>.npy`` when it is intact, and otherwise rebuilds
        it from the annotation and depth map - so neither a truncated file in
        the published release nor a partial download (annotations and depth
        only, which is a tenth of the bytes) stops an analysis.
        """
        if self.mask_file_ok:
            return np.load(self.root.path("masks", self.name), allow_pickle=False)
        return self._rebuilt[0]

    @property
    def was_rebuilt(self) -> bool:
        """True if this scene's label map came from ``oif repair``, not the release."""
        if not self.mask_file_ok:
            return True
        registry = self.root.dir("derived") / REBUILT_FILENAME
        if not registry.exists():
            return False
        df = pd.read_csv(registry, usecols=["image"])
        return bool((df["image"].str.lower() == self.name.lower()).any())

    @property
    def mask_file_ok(self) -> bool:
        try:
            return self.root.path("masks", self.name).stat().st_size >= 1024
        except FileNotFoundError:
            return False

    @property
    def shape(self) -> Tuple[int, int]:
        return self.annotation.shape

    # -- labels ------------------------------------------------------------
    @cached_property
    def labels(self) -> Dict[int, str]:
        """``{mask_id: label}`` for this scene.

        Uses ``derived/labels.csv`` if it exists (written by ``oif labels``),
        otherwise recovers the mapping on the fly - see
        :func:`oif.masks.recover_labels`.
        """
        table = self._labels_from_file()
        if table is not None:
            return table
        return {r.mask_id: r.label for r in self.recover_labels()}

    def _labels_from_file(self) -> Optional[Dict[int, str]]:
        path = self.root.dir("derived") / LABELS_FILENAME
        if not path.exists():
            return None
        df = pd.read_csv(path)
        sub = df[df["image"].str.lower() == self.name.lower()]
        if sub.empty:
            return None
        return {int(r.mask_id): str(r.label) for r in sub.itertuples()}

    @cached_property
    def _rebuilt(self) -> Tuple[np.ndarray, List[ann_mod.Polygon]]:
        """Label map built here, with the polygon behind each id."""
        return mask_mod.build_label_map(self.annotation, self.depth)

    def recover_labels(self, **kwargs) -> List[mask_mod.RecoveredObject]:
        """Match every id in the label map back to its annotation polygon.

        When the label map was rebuilt by this package (because the published
        mask file was unreadable), the mapping is known exactly from the
        painting order and is returned directly, with a score of 1.0.
        """
        exact = self._exact_labels()
        if exact is not None:
            return exact
        return mask_mod.recover_labels(self.label_map, self.annotation, **kwargs)

    def _exact_labels(self) -> Optional[List[mask_mod.RecoveredObject]]:
        """Labels known by construction, for a label map this package built.

        Either the mask file is unreadable and we are rebuilding it right now,
        or ``oif repair`` rebuilt it earlier and recorded what it painted in
        ``derived/rebuilt_labels.csv``. Both cases give the id-to-label
        mapping outright, so there is nothing to infer.
        """
        if not self.mask_file_ok:
            label_map, ordered = self._rebuilt
            return self._records_from_order(label_map, ordered)

        registry = self.root.dir("derived") / REBUILT_FILENAME
        if not registry.exists():
            return None
        df = pd.read_csv(registry)
        sub = df[df["image"].str.lower() == self.name.lower()]
        if sub.empty:
            return None
        areas = mask_mod.region_areas(self.label_map)
        return [
            mask_mod.RecoveredObject(
                mask_id=int(r.mask_id), label=str(r.label), occluded=int(r.occluded),
                polygon_index=int(r.polygon_index), score=1.0,
                area=int(areas.get(int(r.mask_id), 0)),
                visible_fraction=float(getattr(r, "visible_fraction", float("nan"))),
            )
            for r in sub.itertuples()
        ]

    @staticmethod
    def _records_from_order(label_map: np.ndarray, ordered: List[ann_mod.Polygon]
                            ) -> List[mask_mod.RecoveredObject]:
        areas = mask_mod.region_areas(label_map)
        out = []
        for mask_id, poly in enumerate(ordered, start=1):
            area = areas.get(mask_id, 0)
            full = int(mask_mod.rasterize(poly, label_map.shape).astype(bool).sum())
            out.append(mask_mod.RecoveredObject(
                mask_id=mask_id, label=poly.label, occluded=int(poly.occluded),
                polygon_index=int(poly.index), score=1.0, area=area,
                visible_fraction=(area / full) if full else 0.0,
            ))
        return out

    def rebuild_records(self) -> List[mask_mod.RecoveredObject]:
        """Rebuild the label map and return its exact id-to-label records."""
        label_map, ordered = self._rebuilt
        return self._records_from_order(label_map, ordered)

    def build_label_map(self, **kwargs) -> np.ndarray:
        """Rebuild the label map from polygons + depth (back-to-front)."""
        if not kwargs:
            return self._rebuilt[0]
        label_map, _ = mask_mod.build_label_map(self.annotation, self.depth, **kwargs)
        return label_map

    # -- objects -----------------------------------------------------------
    @property
    def object_ids(self) -> np.ndarray:
        return mask_mod.region_ids(self.label_map)

    def object_mask(self, mask_id: int) -> np.ndarray:
        return self.label_map == int(mask_id)

    def features(self, salience: Optional[np.ndarray] = None, **kwargs) -> pd.DataFrame:
        """Size, eccentricity, depth and salience for every object."""
        return feat_mod.object_features(self.label_map, depth=self.depth,
                                        salience=salience, labels=self.labels,
                                        scene=self.name, **kwargs)

    # -- fixations ---------------------------------------------------------
    def fixation_map(self, task: str = "memorize") -> np.ndarray:
        """The released binary fixation map for this scene.

        ``raw/<scene>_<task>.npy``: an image-shaped array with 1 at every
        pixel where a fixation landed, pooled over the study's viewers.
        """
        return fix_mod.load_fixation_map(
            self.root.path("raw", self.name, f"_{task}.npy"))

    def fixations(self, task: str = "memorize") -> pd.DataFrame:
        """This scene's released fixations as a canonical table.

        One row per fixated pixel in the binary map. Pooled maps carry no
        subject, order or duration, so those columns are empty.
        """
        return fix_mod.map_to_fixations(self.fixation_map(task), self.name, task)

    def map_fixations(self, fixations: pd.DataFrame, method: str = "point",
                      radius: int = 0, **kwargs) -> pd.DataFrame:
        """Label each fixation with the object it landed on."""
        frame = fixations
        if "image" in frame.columns:
            frame = frame[frame["image"].str.lower() == self.name.lower()]
        return map_mod.map_fixations(frame, self.label_map, self.labels,
                                     method=method, radius=radius, **kwargs)

    def object_table(self, fixations: Optional[pd.DataFrame] = None,
                     salience: Optional[np.ndarray] = None,
                     method: str = "point", radius: int = 0) -> pd.DataFrame:
        """One row per object: features, and fixation counts if given fixations."""
        table = self.features(salience=salience)
        if fixations is None:
            return table
        mapped = self.map_fixations(fixations, method=method, radius=radius)
        counts = map_mod.object_fixations(mapped, self.label_map, self.labels,
                                          scene=self.name)
        counts = counts.drop(columns=["label", "image"], errors="ignore")
        return table.merge(counts, on="mask_id", how="left").fillna(
            {"n_fixations": 0, "total_duration": 0.0, "n_subjects": 0})

    def density(self, fixations: pd.DataFrame, fc: float = 6.0,
                weight: str = "count") -> np.ndarray:
        """Blurred fixation density map for this scene."""
        frame = fixations
        if "image" in frame.columns:
            frame = frame[frame["image"].str.lower() == self.name.lower()]
        return fix_mod.density_map(frame, shape=self.shape, weight=weight, fc=fc)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Scene({self.name!r}, {len(self.annotation)} polygons)"


class OiF:
    """The dataset as a whole: a sequence of :class:`Scene` objects.

    Parameters
    ----------
    root:
        Path to the data tree. Defaults to searching the working directory
        and its parents, then ``$OIF_DATA_ROOT``.
    """

    def __init__(self, root: Optional[str | Path] = None):
        self.data = DataRoot(root)

    # -- access ------------------------------------------------------------
    @property
    def root(self) -> Path:
        return self.data.root

    @cached_property
    def scene_names(self) -> List[str]:
        return self.data.scenes()

    def __len__(self) -> int:
        return len(self.scene_names)

    def __iter__(self) -> Iterator[Scene]:
        for name in self.scene_names:
            yield Scene(name, self.data)

    def __getitem__(self, key: str | int) -> Scene:
        if isinstance(key, int):
            return Scene(self.scene_names[key], self.data)
        lower = {n.lower(): n for n in self.scene_names}
        if key.lower() not in lower:
            raise KeyError(f"no scene named {key!r}; try data.scene_names")
        return Scene(lower[key.lower()], self.data)

    def __contains__(self, key: str) -> bool:
        return key.lower() in {n.lower() for n in self.scene_names}

    # -- dataset-level -----------------------------------------------------
    def check(self):
        """Integrity report for the data tree."""
        return self.data.check()

    def fixations(self, folder: str = "raw", **kwargs) -> pd.DataFrame:
        """Load every fixation file in ``<root>/raw`` in canonical form.

        Handles both tabular reports (csv/tsv/xlsx/parquet) and the released
        binary ``.npy`` fixation maps; see :mod:`oif.fixations`.
        """
        return fix_mod.load_fixations(self.data.dir(folder), **kwargs)

    def label_table(self, progress: bool = False) -> pd.DataFrame:
        """Recover ``image, mask_id, label, ...`` for every scene.

        This is the lookup the published release is missing: without it the
        ids in ``masks/`` are anonymous integers.
        """
        rows = []
        for scene in self:
            if progress:
                print(f"  {scene.name}", flush=True)
            for rec in scene.recover_labels():
                rows.append({
                    "image": scene.name,
                    "mask_id": rec.mask_id,
                    "label": rec.label,
                    "occluded": rec.occluded,
                    "polygon_index": rec.polygon_index,
                    "area": rec.area,
                    "visible_fraction": round(rec.visible_fraction, 6),
                    "match_score": round(rec.score, 6),
                    "rebuilt": scene.was_rebuilt,
                })
        return pd.DataFrame(rows)

    def object_tables(self, fixations: Optional[pd.DataFrame] = None,
                      salience: Optional[Mapping[str, np.ndarray]] = None,
                      method: str = "point", radius: int = 0,
                      scenes: Optional[Sequence[str]] = None,
                      progress: bool = False) -> pd.DataFrame:
        """Object feature + fixation table for the whole dataset."""
        frames = []
        names = scenes if scenes is not None else self.scene_names
        for name in names:
            scene = self[name]
            if progress:
                print(f"  {name}", flush=True)
            sal = (salience or {}).get(name)
            frames.append(scene.object_table(fixations, salience=sal,
                                             method=method, radius=radius))
        return pd.concat(frames, ignore_index=True)

    def summary(self) -> pd.DataFrame:
        """Per-scene counts of polygons, objects and background surfaces."""
        rows = []
        for scene in self:
            a = scene.annotation
            rows.append({
                "image": scene.name,
                "n_polygons": len(a),
                "n_objects": len(a.objects()),
                "n_background": len(a.background()),
                "n_unique_labels": len(set(a.labels)),
                "mask_ok": scene.mask_file_ok,
            })
        return pd.DataFrame(rows)

    def stats(self) -> Dict[str, object]:
        """Headline dataset numbers, the ones a paper's Stimuli section needs."""
        summary = self.summary()
        all_labels: List[str] = []
        for scene in self:
            all_labels.extend(scene.annotation.labels)
        return {
            "n_scenes": len(self),
            "n_polygons": int(summary["n_polygons"].sum()),
            "n_objects": int(summary["n_objects"].sum()),
            "objects_per_scene": float(summary["n_objects"].mean()),
            "n_unique_labels": len(set(all_labels)),
            "image_shape": self[0].shape,
        }

    def write_labels(self, path: Optional[str | Path] = None,
                     progress: bool = False) -> Path:
        """Recover and save the label lookup to ``derived/labels.csv``."""
        table = self.label_table(progress=progress)
        out = Path(path) if path else self.data.dir("derived", create=True) / LABELS_FILENAME
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        meta = out.with_suffix(".json")
        meta.write_text(json.dumps({
            "generated_by": "oif.datasets.oif.OiF.write_labels",
            "n_rows": int(len(table)),
            "n_scenes": int(table["image"].nunique()),
            "min_match_score": float(table["match_score"].min()),
            "n_low_confidence": int((table["match_score"] < 0.99).sum()),
            "n_rebuilt_scenes": int(table.loc[table["rebuilt"], "image"].nunique()),
        }, indent=2))
        return out

    def __repr__(self) -> str:  # pragma: no cover
        return f"OiF({str(self.root)!r}, {len(self)} scenes)"
