"""Tests that exercise the dataset wrapper, the CLI, and the published data."""

from __future__ import annotations

import numpy as np
import pytest

from oif import OiF
from oif.cli import main

from .conftest import needs_real_data, real_root

# -- data root handling ----------------------------------------------------

def test_dataset_indexing(tiny_root):
    data = OiF(tiny_root)
    assert len(data) == 1
    assert "target_test" in data
    assert data["TARGET_TEST"].name == "target_test"   # case-insensitive
    assert data[0].name == "target_test"
    with pytest.raises(KeyError, match="no scene named"):
        data["nope"]


def test_missing_root_message(tmp_path):
    """An explicit bad path errors rather than falling back to the cwd."""
    with pytest.raises(FileNotFoundError, match="is not an Objects in Focus data root"):
        OiF(tmp_path / "not-a-dataset")


def test_check_reports_truncated_masks(tiny_root):
    (tiny_root / "masks" / "target_test.npy").write_bytes(b"\x00\x00")
    report = OiF(tiny_root).check()
    assert not report.ok
    assert report.truncated["masks"] == ["target_test"]


def test_label_map_falls_back_when_mask_is_truncated(tiny_root):
    (tiny_root / "masks" / "target_test.npy").write_bytes(b"\x00\x00")
    scene = OiF(tiny_root)[0]
    assert not scene.mask_file_ok
    assert scene.label_map.shape == scene.shape
    assert set(scene.labels.values()) == {"chair", "table", "lamp"}


def test_case_insensitive_file_lookup(tiny_root):
    src = tiny_root / "annotations" / "target_test.xml"
    src.rename(tiny_root / "annotations" / "TARGET_test.xml")
    scene = OiF(tiny_root)[0]
    assert len(scene.annotation) == 4


# -- end to end ------------------------------------------------------------

def test_object_table_end_to_end(tiny_masks):
    data = OiF(tiny_masks)
    scene = data[0]
    table = scene.object_table(data.fixations())
    assert set(table["label"]) == {"chair", "table", "lamp"}
    assert table["n_fixations"].sum() > 0
    assert (table["n_fixations"] >= 0).all()


def test_density_map_shape(tiny_masks):
    data = OiF(tiny_masks)
    scene = data[0]
    density = scene.density(data.fixations())
    assert density.shape == scene.shape
    assert np.isfinite(density).all()


def test_write_labels(tiny_masks):
    data = OiF(tiny_masks)
    out = data.write_labels()
    assert out.exists()
    assert out.with_suffix(".json").exists()
    table = data.label_table()
    assert len(table) == 3
    assert (table["match_score"] > 0.99).all()


def test_summary_and_stats(tiny_masks):
    data = OiF(tiny_masks)
    summary = data.summary()
    assert summary.loc[0, "n_objects"] == 3
    assert summary.loc[0, "n_background"] == 1
    stats = data.stats()
    assert stats["n_scenes"] == 1
    assert stats["image_shape"] == (120, 160)


# -- CLI -------------------------------------------------------------------

def test_cli_check(tiny_masks, capsys):
    assert main(["check", "--root", str(tiny_masks)]) == 0
    assert "scenes: 1" in capsys.readouterr().out


def test_cli_root_before_subcommand(tiny_masks, capsys):
    assert main(["--root", str(tiny_masks), "stats"]) == 0
    assert "n_scenes" in capsys.readouterr().out


def test_cli_repair_writes_registry(tiny_root, capsys):
    (tiny_root / "masks" / "target_test.npy").write_bytes(b"\x00\x00")
    assert main(["repair", "--root", str(tiny_root)]) == 0
    assert (tiny_root / "derived" / "rebuilt_labels.csv").exists()
    assert OiF(tiny_root).check().ok
    # the rebuilt scene now carries exact labels, not inferred ones
    assert all(r.score == 1.0 for r in OiF(tiny_root)[0].recover_labels())


def test_cli_objects_table(tiny_masks, tmp_path):
    out = tmp_path / "objects.csv"
    code = main(["objects", "--root", str(tiny_masks),
                 "--fixations", str(tiny_masks / "raw"), "--out", str(out)])
    assert code == 0
    import pandas as pd
    table = pd.read_csv(out)
    assert {"log_size", "z_ecc", "n_fixations"} <= set(table.columns)


# -- published data --------------------------------------------------------

@needs_real_data
def test_published_dataset_shape():
    data = OiF(real_root())
    assert len(data) == 100
    scene = data[0]
    assert scene.image.shape == (768, 1024, 3)
    assert scene.label_map.shape == (768, 1024)
    assert scene.depth.shape == (768, 1024)


@needs_real_data
def test_published_labels_are_recoverable():
    """Every mask id in a sample of scenes matches a polygon convincingly."""
    data = OiF(real_root())
    for name in data.scene_names[:5]:
        recovered = data[name].recover_labels()
        assert recovered
        assert min(r.score for r in recovered) > 0.95
        assert all(r.label for r in recovered)
