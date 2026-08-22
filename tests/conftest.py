"""Test fixtures.

Most tests run on a synthetic three-object scene built in a temp folder, so
the suite passes in CI without the (large) published data. Tests that need
the real thing are marked ``real_data`` and skip unless a data root is
available - set ``OIF_DATA_ROOT`` or run pytest from the repository root.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SHAPE = (120, 160)  # (h, w)


def _polygon_xml(label: str, points: str, occluded: int = 0, z: int = 0) -> str:
    return (f'    <polygon label="{label}" occluded="{occluded}" '
            f'points="{points}" z_order="{z}"></polygon>\n')


@pytest.fixture
def tiny_root(tmp_path: Path) -> Path:
    """A miniature dataset: 1 scene, 3 objects and a background surface."""
    root = tmp_path / "data"
    for folder in ("images", "annotations", "masks", "depth", "raw"):
        (root / folder).mkdir(parents=True)

    h, w = SHAPE
    scene = "target_test"

    xml = ['<?xml version="1.0" encoding="utf-8"?>\n<annotations>\n'
           '  <version>1.1</version>\n'
           f'  <image id="0" name="{scene}.png" width="{w}" height="{h}">\n']
    # background surface covering the top half
    xml.append(_polygon_xml("sky", f"0,0;{w},0;{w},60;0,60"))
    # three objects, deliberately overlapping so occlusion matters
    xml.append(_polygon_xml("chair", "10,70;60,70;60,110;10,110"))
    xml.append(_polygon_xml("table", "40,80;120,80;120,115;40,115", occluded=1))
    xml.append(_polygon_xml("lamp", "80,20;110,20;110,55;80,55"))
    xml.append("  </image>\n</annotations>\n")
    (root / "annotations" / f"{scene}.xml").write_text("".join(xml))

    rng = np.random.default_rng(0)
    image = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(image).save(root / "images" / f"{scene}.png")

    # MonoDepth2-shaped disparity: near the bottom of the frame = nearer
    disp = np.linspace(0.2, 1.4, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    np.save(root / "depth" / f"{scene}_disp.npy", disp[None, None, :, :])

    fixations = pd.DataFrame({
        "RECORDING_SESSION_LABEL": ["s1"] * 6 + ["s2"] * 4,
        "image_name": [f"{scene}.png"] * 10,
        "CURRENT_FIX_INDEX": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
        "CURRENT_FIX_X": [80, 30, 35, 100, 95, 300, 80, 50, 90, 20],
        "CURRENT_FIX_Y": [60, 90, 95, 100, 40, 60, 60, 90, 30, 100],
        "CURRENT_FIX_DURATION": [200, 250, 300, 180, 420, 200, 210, 260, 30, 2000],
        "task": ["memorize"] * 10,
    })
    fixations.to_csv(root / "raw" / "fixations.csv", index=False)
    return root


@pytest.fixture
def tiny_masks(tiny_root: Path) -> Path:
    """``tiny_root`` with a mask file written, as the published tree has."""
    from oif import OiF
    data = OiF(tiny_root)
    scene = data[0]
    np.save(tiny_root / "masks" / f"{scene.name}.npy",
            scene.build_label_map().astype(np.int32))
    return tiny_root


def real_root():
    """The published data tree, if this checkout has one."""
    import os
    env = os.environ.get("OIF_DATA_ROOT")
    candidates = [Path(env)] if env else []
    candidates.append(Path(__file__).resolve().parents[1])
    for c in candidates:
        if (c / "images").is_dir() and (c / "annotations").is_dir():
            if any((c / "images").glob("*.png")):
                return c
    return None


needs_real_data = pytest.mark.skipif(real_root() is None,
                                     reason="published OiF data not available")
