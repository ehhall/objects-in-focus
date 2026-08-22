"""Reading the CVAT polygon annotations that define OiF objects.

Every scene was hand-segmented in CVAT and exported as XML 1.1. One
``<polygon>`` element is one annotated object::

    <polygon label="coral" points="126.09,524.34;118.59,537.23;..."
             occluded="0" z_order="23"/>

The original analysis notebooks pulled these out with a regular expression
over prettified XML, which silently dropped any label containing a space or
hyphen (the pattern was ``label="(\\w*)"``). This module parses the XML
properly, so multi-word labels survive, and reports what the regex would
have missed via :func:`labels_lost_to_regex`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Polygon",
    "SceneAnnotation",
    "BACKGROUND_LABELS",
    "read_annotation",
    "labels_lost_to_regex",
]

#: Labels treated as scene background (surfaces and extended stuff regions)
#: rather than objects. This is the list used to build the published masks;
#: pass your own to any function that takes ``background=``.
BACKGROUND_LABELS: Tuple[str, ...] = (
    "floor", "sky", "wall", "grass", "carpet", "ground",
    "ceiling", "trees", "water", "road", "pavement",
)

_LEGACY_REGEX = re.compile(r'label="(?P<object>\w*)".*points="(?P<coordinates>[\d\.,;]*)"')


@dataclass(frozen=True)
class Polygon:
    """One annotated object outline.

    Attributes
    ----------
    label:
        Category name as typed by the annotator, e.g. ``"coral"``.
    points:
        ``(n, 2)`` float array of ``(x, y)`` vertices in image pixels. The
        ring is *not* closed; rasterisers close it themselves.
    occluded:
        Annotator flag: 1 if the object is partly hidden by another object.
    z_order:
        CVAT drawing order. Preserved but not used for depth sorting - see
        :mod:`oif.masks`, which sorts by estimated depth instead.
    index:
        Position of this polygon in the XML file, so a row in a derived table
        can always be traced back to its source element.
    """

    label: str
    points: np.ndarray
    occluded: int = 0
    z_order: int = 0
    index: int = -1

    @property
    def n_vertices(self) -> int:
        return int(self.points.shape[0])

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)`` of the outline."""
        p = self.points
        return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())

    def closed(self) -> np.ndarray:
        """Vertices with the first point repeated at the end."""
        return np.vstack([self.points, self.points[:1]])


@dataclass
class SceneAnnotation:
    """All polygons for one scene, plus the image size they were drawn at."""

    scene: str
    width: int
    height: int
    polygons: List[Polygon]
    source: Optional[Path] = None

    @property
    def shape(self) -> Tuple[int, int]:
        """Array shape ``(height, width)`` - the numpy convention."""
        return self.height, self.width

    @property
    def labels(self) -> List[str]:
        return [p.label for p in self.polygons]

    def objects(self, background: Sequence[str] = BACKGROUND_LABELS) -> List[Polygon]:
        """Polygons excluding background surfaces."""
        bg = {b.lower() for b in background}
        return [p for p in self.polygons if p.label.lower() not in bg]

    def background(self, background: Sequence[str] = BACKGROUND_LABELS) -> List[Polygon]:
        bg = {b.lower() for b in background}
        return [p for p in self.polygons if p.label.lower() in bg]

    def __len__(self) -> int:
        return len(self.polygons)

    def __iter__(self):
        return iter(self.polygons)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"SceneAnnotation({self.scene!r}, {len(self.polygons)} polygons, "
                f"{self.width}x{self.height})")


def _parse_points(text: str) -> np.ndarray:
    pairs = [pt for pt in text.strip().split(";") if pt]
    return np.array([[float(v) for v in pt.split(",")] for pt in pairs], dtype=float)


def read_annotation(path: str | Path, scene: Optional[str] = None) -> SceneAnnotation:
    """Parse one CVAT XML file into a :class:`SceneAnnotation`."""
    path = Path(path)
    root = ET.parse(path).getroot()

    image_el = root.find("image")
    if image_el is None:
        raise ValueError(f"{path} has no <image> element - not a CVAT 1.1 export?")

    width = int(float(image_el.get("width", 0)))
    height = int(float(image_el.get("height", 0)))
    name = scene or Path(image_el.get("name", path.stem)).stem

    polygons: List[Polygon] = []
    for i, el in enumerate(image_el.findall("polygon")):
        pts = _parse_points(el.get("points", ""))
        if pts.shape[0] < 3:
            continue  # degenerate outline, cannot be filled
        polygons.append(
            Polygon(
                label=(el.get("label") or "").strip(),
                points=pts,
                occluded=int(el.get("occluded", 0) or 0),
                z_order=int(el.get("z_order", 0) or 0),
                index=i,
            )
        )
    return SceneAnnotation(scene=name, width=width, height=height,
                           polygons=polygons, source=path)


def labels_lost_to_regex(path: str | Path) -> List[str]:
    """Labels this parser finds that the notebooks' regex would have dropped.

    Useful as a one-line audit when comparing new results against numbers
    published from the original notebooks.
    """
    ann = read_annotation(path)
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    kept = {m.group("object") for m in _LEGACY_REGEX.finditer(text)}
    return sorted({p.label for p in ann.polygons if p.label not in kept})


def read_annotations(paths: Iterable[str | Path]) -> List[SceneAnnotation]:
    return [read_annotation(p) for p in paths]
