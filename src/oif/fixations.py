"""Reading fixation reports and turning them into arrays.

Eye-movement data arrives in whatever column names the recording software
produced. Everything downstream in this package expects one tidy table:

=============  ====================================================
column         meaning
=============  ====================================================
``subject``    viewer id
``image``      scene name, matching ``images/`` without extension
``fix_index``  1-based order of the fixation within a trial
``x``, ``y``   fixation position in image pixels, origin top-left
``duration``   fixation duration in milliseconds
``task``       viewing task, if the study had more than one
=============  ====================================================

:func:`load_fixations` reads every file in ``raw/`` and normalises it to
that schema. Two source formats are recognised out of the box:

* **Tables** (csv/tsv/xlsx/parquet) - SR Research DataViewer exports
  (``CURRENT_FIX_X`` ...) and the lab's older ``locs_1``/``locs_2``/``durs``
  layout. Anything else: pass ``columns={...}`` to say which of your columns
  is which.
* **Binary fixation maps** (``.npy``) - the released OiF fixations,
  ``raw/target_<scene>_<task>.npy``: an image-shaped array with a 1 at every
  pixel where a fixation landed, pooled over viewers. These carry no
  subject, order or duration, so those columns come back empty; counts per
  object still work exactly the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "CANONICAL_COLUMNS",
    "SCHEMAS",
    "load_fixations",
    "read_fixation_file",
    "load_fixation_map",
    "map_to_fixations",
    "parse_map_filename",
    "normalize_columns",
    "filter_fixations",
    "fixation_array",
    "density_map",
    "gaussian_blur",
]

CANONICAL_COLUMNS: Tuple[str, ...] = (
    "subject", "image", "fix_index", "x", "y", "duration", "task",
)

#: Known source layouts, tried in order. Keys are canonical names.
SCHEMAS: Dict[str, Dict[str, str]] = {
    "dataviewer": {
        "subject": "RECORDING_SESSION_LABEL",
        "image": "image_name",
        "fix_index": "CURRENT_FIX_INDEX",
        "x": "CURRENT_FIX_X",
        "y": "CURRENT_FIX_Y",
        "duration": "CURRENT_FIX_DURATION",
        "task": "task",
    },
    "legacy_locs": {
        "subject": "subj",
        "image": "image",
        "fix_index": "fixN",
        "x": "locs_1",
        "y": "locs_2",
        "duration": "durs",
        "task": "task",
    },
    "canonical": {c: c for c in CANONICAL_COLUMNS},
}


@dataclass
class FixationSchema:
    """Which source column supplies each canonical column."""

    name: str
    mapping: Dict[str, str]
    missing: List[str]


def detect_schema(df: pd.DataFrame) -> FixationSchema:
    """Pick the source layout that covers the most required columns."""
    required = ("image", "x", "y")
    best: Optional[FixationSchema] = None
    for name, mapping in SCHEMAS.items():
        present = {k: v for k, v in mapping.items() if v in df.columns}
        if not all(k in present for k in required):
            continue
        missing = [c for c in CANONICAL_COLUMNS if c not in present]
        cand = FixationSchema(name, present, missing)
        if best is None or len(cand.mapping) > len(best.mapping):
            best = cand
    if best is None:
        raise ValueError(
            "Could not recognise the fixation columns. Columns found: "
            f"{list(df.columns)[:20]}. Pass columns={{'x': ..., 'y': ..., "
            "'image': ...}} to name them explicitly."
        )
    return best


def normalize_columns(df: pd.DataFrame, columns: Optional[Dict[str, str]] = None,
                      strip_extension: bool = True) -> pd.DataFrame:
    """Rename a raw fixation table to the canonical schema."""
    mapping = dict(detect_schema(df).mapping)
    if columns:
        mapping.update({k: v for k, v in columns.items() if v in df.columns})

    out = pd.DataFrame(index=df.index)
    for canon, src in mapping.items():
        out[canon] = df[src]
    for canon in CANONICAL_COLUMNS:
        if canon not in out.columns:
            out[canon] = np.nan

    if strip_extension:
        out["image"] = (out["image"].astype(str)
                        .str.replace(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", "", regex=True,
                                     case=False))
    for col in ("x", "y", "duration", "fix_index"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[list(CANONICAL_COLUMNS)]


def parse_map_filename(name: str) -> Tuple[str, Optional[str]]:
    """Split a fixation-map filename into ``(image, task)``.

    ``target_bakery_memorize.npy`` -> ``("target_bakery", "memorize")``.
    The task is the part after the last underscore; scene names that end in
    an IDS suffix (``target_boating_IDS01_memorize``) still split correctly
    because the task comes last. A name with no underscore has no task.
    """
    stem = Path(name).stem
    if "_" not in stem:
        return stem, None
    image, task = stem.rsplit("_", 1)
    return image, task


def load_fixation_map(path: str | Path) -> np.ndarray:
    """Load one binary fixation map (``raw/*.npy``) as an image-shaped array.

    The released maps are 768 x 1024 float arrays holding 1 at every pixel
    where a fixation landed, pooled over viewers.
    """
    arr = np.load(Path(path), allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"{Path(path).name}: expected a 2-D fixation map, "
                         f"got shape {arr.shape}")
    return arr


def map_to_fixations(arr: np.ndarray, image: str,
                     task: Optional[str] = None) -> pd.DataFrame:
    """Turn a binary fixation map into a canonical fixation table.

    One row per nonzero pixel. Subject, order and duration are unknown for
    pooled maps, so those columns are empty (NaN); every cleaning and
    mapping step downstream treats empty as "no constraint".
    """
    ys, xs = np.nonzero(arr)
    out = pd.DataFrame({
        "subject": np.nan, "image": image, "fix_index": np.nan,
        "x": xs.astype(float), "y": ys.astype(float),
        "duration": np.nan, "task": task if task is not None else np.nan,
    })
    return out[list(CANONICAL_COLUMNS)]


def read_fixation_file(path: str | Path, columns: Optional[Dict[str, str]] = None,
                       **read_kwargs) -> pd.DataFrame:
    """Read one fixation file (csv/tsv/txt/xlsx/parquet/npy) into canonical form."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        image, task = parse_map_filename(path.name)
        out = map_to_fixations(load_fixation_map(path), image, task)
        out["source_file"] = path.name
        return out
    if suffix in {".csv", ".txt", ".tsv"}:
        sep = read_kwargs.pop("sep", "\t" if suffix == ".tsv" else None)
        df = pd.read_csv(path, sep=sep, engine="python", **read_kwargs)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, **read_kwargs)
    elif suffix == ".parquet":
        df = pd.read_parquet(path, **read_kwargs)
    else:
        raise ValueError(f"unsupported fixation file type: {path.name}")
    out = normalize_columns(df, columns)
    out["source_file"] = path.name
    return out


#: File types load_fixations picks up when scanning a folder.
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".npy"}


def load_fixations(
    path: str | Path,
    columns: Optional[Dict[str, str]] = None,
    pattern: str = "*",
    **read_kwargs,
) -> pd.DataFrame:
    """Load a fixation table from a file, or every matching file in a folder.

    Picks up both tabular reports and binary ``.npy`` fixation maps. Files
    that cannot be parsed are skipped with a warning rather than aborting
    the whole load, so one stray export does not block an analysis.
    """
    import warnings

    path = Path(path)
    if path.is_file():
        return read_fixation_file(path, columns, **read_kwargs)

    files = sorted(p for p in path.glob(pattern)
                   if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} in {path}")

    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            frames.append(read_fixation_file(f, columns, **read_kwargs))
        except Exception as exc:  # noqa: BLE001 - report and continue
            warnings.warn(f"skipping {f.name}: {exc}", stacklevel=2)
    if not frames:
        raise ValueError(f"none of the {len(files)} files in {path} could be parsed")
    return pd.concat(frames, ignore_index=True)


def filter_fixations(
    df: pd.DataFrame,
    shape: Tuple[int, int] = (768, 1024),
    min_duration: Optional[float] = 50,
    max_duration: Optional[float] = 1500,
    drop_first: bool = True,
    max_index: Optional[int] = None,
    on_image: bool = True,
) -> pd.DataFrame:
    """Apply the standard scene-viewing cleaning steps.

    The defaults are the ones used in the OiF analyses: drop fixations
    shorter than 50 ms or longer than 1500 ms, drop the first fixation of
    each trial (it reflects where the viewer was told to start, not where the
    scene sent them), and drop anything falling outside the image.
    """
    out = df
    if min_duration is not None:
        out = out[out["duration"].isna() | (out["duration"] > min_duration)]
    if max_duration is not None:
        out = out[out["duration"].isna() | (out["duration"] < max_duration)]
    if drop_first:
        out = out[out["fix_index"].isna() | (out["fix_index"] != 1)]
    if max_index is not None:
        out = out[out["fix_index"].isna() | (out["fix_index"] <= max_index)]
    if on_image:
        h, w = shape
        out = out[(out["x"] >= 0) & (out["x"] < w) & (out["y"] >= 0) & (out["y"] < h)]
    return out.reset_index(drop=True)


def fixation_array(df: pd.DataFrame, shape: Tuple[int, int] = (768, 1024),
                   weight: str = "count") -> np.ndarray:
    """Accumulate fixations into an image-shaped array.

    ``weight="count"`` adds 1 per fixation; ``weight="duration"`` adds the
    fixation's duration in ms; any other column name adds that column.
    """
    arr = np.zeros(shape, dtype=float)
    if len(df) == 0:
        return arr
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    xi = x[ok].astype(int)
    yi = y[ok].astype(int)
    inside = (xi >= 0) & (xi < shape[1]) & (yi >= 0) & (yi < shape[0])
    xi, yi = xi[inside], yi[inside]
    if weight == "count":
        w = np.ones(xi.shape[0])
    else:
        col = "duration" if weight == "duration" else weight
        w = df[col].to_numpy()[ok][inside].astype(float)
    np.add.at(arr, (yi, xi), w)
    return arr


def gaussian_blur(img: np.ndarray, fc: float = 6.0) -> np.ndarray:
    """Low-pass filter in the Fourier domain, cut-off ``fc`` (-6 dB).

    Port of Antonio Torralba's ``antonioGaussian`` MATLAB filter, the blur
    used to turn discrete fixations into the density maps reported in the
    saliency literature. Keeping the same filter keeps numbers comparable
    with that literature.
    """
    import math

    img = np.asarray(img, dtype=float)
    if img.ndim > 2:
        sn, sm, c = img.shape
    else:
        (sn, sm), c = img.shape, 0

    n = max(sn, sm)
    n = n + n % 2
    n = 2 ** (math.ceil(math.log2(n)))

    fx, fy = np.mgrid[0:n, 0:n]
    fx = fx - n / 2
    fy = fy - n / 2
    sigma = fc / math.sqrt(math.log(2))
    gf = np.fft.fftshift(np.exp(-(fx ** 2 + fy ** 2) / (sigma ** 2)))

    if c > 0:
        out = np.zeros((n, n, c))
        for i in range(c):
            out[:, :, i] = np.fft.ifft2(np.fft.fft2(img[:, :, i], s=(n, n)) * gf).real
        return out[:sn, :sm, :]
    out = np.fft.ifft2(np.fft.fft2(img, s=(n, n)) * gf).real
    return out[:sn, :sm]


def density_map(df: pd.DataFrame, shape: Tuple[int, int] = (768, 1024),
                weight: str = "count", fc: float = 6.0) -> np.ndarray:
    """Blurred fixation density map for a set of fixations."""
    return gaussian_blur(fixation_array(df, shape, weight), fc=fc)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-scene counts of fixations, viewers and tasks - a quick sanity check."""
    g = df.groupby("image")
    return pd.DataFrame({
        "n_fixations": g.size(),
        "n_subjects": g["subject"].nunique(),
        "n_tasks": g["task"].nunique(),
        "mean_duration": g["duration"].mean(),
    }).reset_index()


def iter_scene_frames(df: pd.DataFrame, scenes: Optional[Sequence[str]] = None
                      ) -> Iterable[Tuple[str, pd.DataFrame]]:
    """Yield ``(scene, frame)`` pairs, optionally restricted to ``scenes``."""
    if scenes is not None:
        wanted = set(scenes)
        df = df[df["image"].isin(wanted)]
    for scene, frame in df.groupby("image"):
        yield str(scene), frame
