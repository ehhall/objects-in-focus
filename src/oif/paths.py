"""Locating and validating an Objects in Focus data tree.

The dataset is a directory of parallel folders keyed by scene name::

    <root>/
        images/       target_<scene>.png          1024 x 768 RGB
        annotations/  target_<scene>.xml          CVAT 1.1 polygon export
        masks/        target_<scene>.npy          int label map, 768 x 1024
        depth/        target_<scene>_disp.npy     MonoDepth2 disparity
        raw/          target_<scene>_<task>.npy   binary fixation maps (and/or *.csv reports)
        derived/                                  anything this package writes

Two things about the published tree make naive path building fail, so all
lookups go through :class:`DataRoot`:

* Two scenes are spelled with different capitalisation in ``images/`` and
  ``annotations/`` (``target_AtticStorage`` vs ``target_atticstorage``), so
  file resolution is case-insensitive.
* Nine mask files were uploaded truncated (2 bytes). They are detected up
  front rather than blowing up mid-analysis; ``oif repair`` rebuilds them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

__all__ = ["DataRoot", "IntegrityReport", "find_data_root"]

#: Folder name -> (file suffix, whether the scene name is transformed)
LAYOUT = {
    "images": ".png",
    "annotations": ".xml",
    "masks": ".npy",
    "depth": "_disp.npy",
}

#: Smallest plausible size (bytes) per folder. The truncated mask uploads in
#: the published release are 2 bytes; a .npy header alone is 128, and even a
#: one-polygon CVAT export runs to a few hundred.
MIN_FILE_BYTES = 1024
MIN_BYTES_BY_FOLDER = {"annotations": 64}


def _min_bytes(folder: str) -> int:
    return MIN_BYTES_BY_FOLDER.get(folder, MIN_FILE_BYTES)

_ENV_VAR = "OIF_DATA_ROOT"


def find_data_root(start: Optional[os.PathLike | str] = None) -> Path:
    """Return the dataset root, searching in a predictable order.

    Order: explicit ``start`` argument, the ``OIF_DATA_ROOT`` environment
    variable, the current directory, then each parent of the current
    directory. A directory qualifies if it contains ``images/`` and
    ``annotations/``.
    """
    # An explicit path is a statement of intent: if it is not a dataset, say
    # so instead of quietly analysing whatever happens to be in the cwd.
    if start is not None:
        cand = Path(start)
        if (cand / "images").is_dir() and (cand / "annotations").is_dir():
            return cand.resolve()
        raise FileNotFoundError(
            f"{cand} is not an Objects in Focus data root: it needs an images/ "
            "folder and an annotations/ folder next to each other."
        )

    candidates: List[Path] = []
    env = os.environ.get(_ENV_VAR)
    if env:
        candidates.append(Path(env))
    here = Path.cwd().resolve()
    candidates.append(here)
    candidates.extend(here.parents)

    for cand in candidates:
        if (cand / "images").is_dir() and (cand / "annotations").is_dir():
            return cand.resolve()

    raise FileNotFoundError(
        "Could not find an Objects in Focus data root. Pass one explicitly "
        f"(OiF('/path/to/objects-in-focus')) or set {_ENV_VAR}."
    )


@dataclass
class IntegrityReport:
    """Result of :meth:`DataRoot.check`."""

    root: Path
    n_scenes: int
    missing: Dict[str, List[str]] = field(default_factory=dict)
    truncated: Dict[str, List[str]] = field(default_factory=dict)
    case_mismatched: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.truncated)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        lines = [f"Objects in Focus data root: {self.root}",
                 f"  scenes: {self.n_scenes}"]
        for kind, mapping in (("missing", self.missing), ("truncated", self.truncated)):
            for folder, scenes in sorted(mapping.items()):
                shown = ", ".join(scenes[:6]) + (" ..." if len(scenes) > 6 else "")
                lines.append(f"  {kind} in {folder}/: {len(scenes)} ({shown})")
        if self.case_mismatched:
            lines.append(
                f"  case-mismatched filenames: {len(self.case_mismatched)} "
                f"({', '.join(self.case_mismatched)}) - resolved automatically"
            )
        if self.ok:
            lines.append("  status: OK")
        else:
            lines.append("  status: incomplete - see `oif repair --help`")
        return "\n".join(lines)


class DataRoot:
    """Case-insensitive, integrity-aware view of a dataset directory."""

    def __init__(self, root: Optional[os.PathLike | str] = None):
        self.root = find_data_root(root)
        self._index: Dict[str, Dict[str, Path]] = {}

    # -- internals ---------------------------------------------------------
    def _folder_index(self, folder: str) -> Dict[str, Path]:
        """Map lowercased filename -> path for one folder (cached)."""
        if folder not in self._index:
            d = self.root / folder
            self._index[folder] = (
                {p.name.lower(): p for p in sorted(d.iterdir()) if p.is_file()}
                if d.is_dir()
                else {}
            )
        return self._index[folder]

    def refresh(self) -> None:
        """Forget cached directory listings (call after writing new files)."""
        self._index.clear()

    # -- public ------------------------------------------------------------
    def path(self, folder: str, scene: str, suffix: Optional[str] = None) -> Path:
        """Resolve ``<root>/<folder>/<scene><suffix>`` ignoring case.

        Raises :class:`FileNotFoundError` naming the scene and folder when the
        file is absent, which is far easier to act on than a bare path error.
        """
        if suffix is None:
            suffix = LAYOUT.get(folder, "")
        wanted = f"{scene}{suffix}"
        idx = self._folder_index(folder)
        hit = idx.get(wanted.lower())
        if hit is None:
            raise FileNotFoundError(
                f"no {folder}/ file for scene {scene!r} (looked for {wanted})"
            )
        return hit

    def has(self, folder: str, scene: str, suffix: Optional[str] = None) -> bool:
        try:
            p = self.path(folder, scene, suffix)
        except FileNotFoundError:
            return False
        return p.stat().st_size >= _min_bytes(folder)

    def scenes(self) -> List[str]:
        """Scene names, taken from ``images/`` and sorted case-insensitively."""
        idx = self._folder_index("images")
        names = [p.stem for p in idx.values()]
        return sorted(names, key=str.lower)

    def dir(self, folder: str, create: bool = False) -> Path:
        d = self.root / folder
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def check(self, scenes: Optional[Sequence[str]] = None) -> IntegrityReport:
        """Report missing, truncated and case-mismatched files."""
        scenes = list(scenes) if scenes is not None else self.scenes()
        rep = IntegrityReport(root=self.root, n_scenes=len(scenes))
        for folder, suffix in LAYOUT.items():
            if folder == "images":
                continue
            for scene in scenes:
                try:
                    p = self.path(folder, scene, suffix)
                except FileNotFoundError:
                    rep.missing.setdefault(folder, []).append(scene)
                    continue
                if p.stat().st_size < _min_bytes(folder):
                    rep.truncated.setdefault(folder, []).append(scene)
                if p.stem != f"{scene}{suffix}".removesuffix(p.suffix):
                    rep.case_mismatched.append(f"{folder}/{p.name}")
        return rep

    def __iter__(self) -> Iterator[str]:
        return iter(self.scenes())

    def __len__(self) -> int:
        return len(self.scenes())

    def __repr__(self) -> str:  # pragma: no cover
        return f"DataRoot({str(self.root)!r}, {len(self)} scenes)"
