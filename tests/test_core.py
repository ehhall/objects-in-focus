"""Unit tests for annotations, masks, depth, fixations, mapping and features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oif import (
    OiF,
    add_model_terms,
    assign_fixations,
    build_label_map,
    fill_objects,
    filter_fixations,
    load_fixations,
    object_features,
    object_fixations,
    read_annotation,
    recover_labels,
    region_ids,
    split_components,
)
from oif.depth import load_depth, to_depth
from oif.mapping import rescale_fixations
from oif.model import ObjectAttentionModel
from .conftest import SHAPE

# -- annotations -----------------------------------------------------------

def test_read_annotation(tiny_root):
    ann = read_annotation(tiny_root / "annotations" / "target_test.xml")
    assert (ann.width, ann.height) == (160, 120)
    assert ann.shape == (120, 160)
    assert len(ann) == 4
    assert [p.label for p in ann.objects()] == ["chair", "table", "lamp"]
    assert [p.label for p in ann.background()] == ["sky"]
    assert ann.polygons[2].occluded == 1


def test_polygon_geometry(tiny_root):
    ann = read_annotation(tiny_root / "annotations" / "target_test.xml")
    chair = ann.objects()[0]
    assert chair.n_vertices == 4
    assert chair.closed().shape == (5, 2)
    assert chair.bbox == (10.0, 70.0, 60.0, 110.0)


# -- masks -----------------------------------------------------------------

def test_build_label_map_paints_every_object(tiny_root):
    data = OiF(tiny_root)
    scene = data[0]
    label_map, ordered = build_label_map(scene.annotation, scene.depth)
    assert label_map.shape == (120, 160)
    assert len(ordered) == 3
    assert set(region_ids(label_map)) == {1, 2, 3}


def test_occlusion_flag_beats_depth_in_overlaps(tiny_root):
    """An object the annotator flagged as occluded is painted under the rest.

    Here the table sits lower in the frame, so the depth map calls it nearer
    than the chair - but it is flagged ``occluded=1``, and the chair is the
    thing occluding it. The chair must own the overlap.
    """
    data = OiF(tiny_root)
    scene = data[0]
    label_map, ordered = build_label_map(scene.annotation, scene.depth)
    names = {i + 1: p.label for i, p in enumerate(ordered)}
    assert names[int(label_map[95, 50])] == "chair"      # overlapping region
    assert names[int(label_map[100, 110])] == "table"    # table alone


def test_depth_ordering_without_occlusion_flags(tiny_root):
    """With the flags ignored, the nearer object wins instead."""
    data = OiF(tiny_root)
    scene = data[0]
    from oif.masks import depth_order
    polys = scene.annotation.objects()
    order = depth_order(polys, scene.depth, scene.shape, group_by_occlusion=False)
    label_map, ordered = build_label_map(scene.annotation, scene.depth, order=order)
    names = {i + 1: p.label for i, p in enumerate(ordered)}
    assert names[int(label_map[95, 50])] == "table"


def test_recover_labels_round_trip(tiny_root):
    data = OiF(tiny_root)
    scene = data[0]
    label_map, ordered = build_label_map(scene.annotation, scene.depth)
    recovered = recover_labels(label_map, scene.annotation)
    assert len(recovered) == len(ordered)
    assert all(r.confident for r in recovered)
    assert ([r.label for r in sorted(recovered, key=lambda r: r.mask_id)]
            == [p.label for p in ordered])


def test_recover_labels_reports_low_confidence(tiny_root):
    """A label map that does not come from these polygons should say so."""
    data = OiF(tiny_root)
    scene = data[0]
    noise = np.zeros(scene.shape, dtype=np.int32)
    noise[10:30, 10:30] = 1  # a square nothing was annotated at
    recovered = recover_labels(noise, scene.annotation)
    assert recovered and not all(r.confident for r in recovered)


def test_split_components_separates_blobs():
    mask = np.zeros((40, 40), bool)
    mask[2:10, 2:10] = True
    mask[25:35, 25:35] = True
    labels = split_components(mask, connectivity=4)
    assert len({int(v) for v in np.unique(labels)} - {0}) == 2


def test_split_components_absorbs_speckle():
    mask = np.zeros((40, 40), bool)
    mask[2:20, 2:20] = True
    mask[30, 30] = True  # one stray pixel
    labels = split_components(mask, min_size=4, connectivity=4)
    assert len({int(v) for v in np.unique(labels)} - {0}) == 1


def test_region_ids_min_fraction():
    label_map = np.zeros((100, 100), np.int32)
    label_map[:50, :50] = 1     # 25% of the image
    label_map[0, 0:2] = 2       # 0.02%
    assert set(region_ids(label_map)) == {1, 2}
    assert set(region_ids(label_map, min_fraction=0.05)) == {1}


# -- depth -----------------------------------------------------------------

def test_depth_resizes_and_inverts(tiny_root):
    disp = np.load(tiny_root / "depth" / "target_test_disp.npy")
    depth = to_depth(disp, shape=(120, 160))
    assert depth.shape == (120, 160)
    assert 0.0 <= depth.min() and depth.max() <= 1.0
    # disparity grows downward, so depth (farther) must shrink downward
    assert depth[5, 80] > depth[110, 80]


def test_depth_raw_option(tiny_root):
    path = tiny_root / "depth" / "target_test_disp.npy"
    raw = load_depth(path, shape=(120, 160), invert=False, rescale=False)
    assert raw.max() > 1.0  # untouched disparity values


# -- fixations -------------------------------------------------------------

def test_load_fixations_normalises_dataviewer(tiny_root):
    df = load_fixations(tiny_root / "raw")
    assert list(df.columns[:7]) == ["subject", "image", "fix_index", "x", "y",
                                    "duration", "task"]
    assert df["image"].unique().tolist() == ["target_test"]
    assert len(df) == 10


def test_load_fixations_legacy_schema(tmp_path):
    legacy = pd.DataFrame({"subj": [1, 1], "image": ["a", "a"], "fixN": [1, 2],
                           "locs_1": [10, 20], "locs_2": [30, 40], "durs": [100, 200]})
    path = tmp_path / "legacy.csv"
    legacy.to_csv(path, index=False)
    df = load_fixations(path)
    assert df["x"].tolist() == [10, 20]
    assert df["y"].tolist() == [30, 40]


def test_filter_fixations_defaults(tiny_root):
    df = load_fixations(tiny_root / "raw")
    clean = filter_fixations(df, shape=(120, 160))
    assert (clean["fix_index"] != 1).all()          # first fixation dropped
    assert clean["duration"].between(50, 1500).all()  # duration window
    assert (clean["x"] < 160).all()                  # off-image dropped


def test_unknown_columns_raise_helpfully(tmp_path):
    path = tmp_path / "odd.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Could not recognise"):
        load_fixations(path)


def test_parse_map_filename():
    from oif.fixations import parse_map_filename
    assert parse_map_filename("target_bakery_memorize.npy") == \
        ("target_bakery", "memorize")
    assert parse_map_filename("target_boating_IDS01_memorize.npy") == \
        ("target_boating_IDS01", "memorize")
    assert parse_map_filename("plain.npy") == ("plain", None)


def test_load_fixation_map_roundtrip(tmp_path):
    from oif import load_fixation_map, map_to_fixations
    arr = np.zeros((120, 160))
    arr[30, 10] = 1.0
    arr[40, 20] = 1.0
    path = tmp_path / "target_test_memorize.npy"
    np.save(path, arr)

    loaded = load_fixation_map(path)
    assert loaded.shape == (120, 160) and loaded.sum() == 2

    df = map_to_fixations(loaded, "target_test", "memorize")
    assert sorted(zip(df["x"], df["y"])) == [(10.0, 30.0), (20.0, 40.0)]
    assert (df["image"] == "target_test").all()
    assert (df["task"] == "memorize").all()
    assert df["subject"].isna().all() and df["duration"].isna().all()


def test_load_fixations_reads_npy_maps(tmp_path):
    arr = np.zeros((120, 160))
    arr[5, 5] = 1.0
    np.save(tmp_path / "target_a_memorize.npy", arr)
    np.save(tmp_path / "target_b_memorize.npy", arr)
    df = load_fixations(tmp_path)
    assert len(df) == 2
    assert sorted(df["image"]) == ["target_a", "target_b"]


def test_map_fixations_survive_standard_cleaning(tmp_path):
    from oif import load_fixation_map, map_to_fixations
    arr = np.zeros((120, 160))
    arr[60, 80] = 1.0
    df = map_to_fixations(arr, "target_test", "memorize")
    clean = filter_fixations(df, shape=(120, 160))
    assert len(clean) == 1  # NaN duration/order must not be dropped


def test_scene_fixation_map_and_fixations(tiny_root):
    arr = np.zeros(SHAPE)
    arr[10, 12] = 1.0
    np.save(tiny_root / "raw" / "target_test_memorize.npy", arr)
    data = OiF(tiny_root)
    scene = data["target_test"]
    fm = scene.fixation_map()
    assert fm.shape == SHAPE and fm.sum() == 1
    fx = scene.fixations()
    assert len(fx) == 1
    assert fx.loc[0, "x"] == 12.0 and fx.loc[0, "y"] == 10.0
    mapped = scene.map_fixations(fx)
    assert "label" in mapped.columns


# -- mapping ---------------------------------------------------------------

def test_assign_point():
    label_map = np.zeros((50, 50), np.int32)
    label_map[10:20, 10:20] = 7
    ids = assign_fixations([15, 40, -5], [15, 40, 5], label_map)
    assert ids.tolist() == [7, 0, 0]


def test_assign_disc_absorbs_near_misses():
    label_map = np.zeros((50, 50), np.int32)
    label_map[10:20, 10:20] = 7
    assert assign_fixations([22], [15], label_map, method="point").tolist() == [0]
    assert assign_fixations([22], [15], label_map, method="disc", radius=6).tolist() == [7]


def test_assign_nearest_only_fills_background():
    label_map = np.zeros((50, 50), np.int32)
    label_map[10:20, 10:20] = 7
    label_map[30:40, 30:40] = 9
    got = assign_fixations([15, 25], [15, 25], label_map, method="nearest", radius=10)
    assert got[0] == 7          # already on an object, untouched
    assert got[1] in (7, 9)     # background, filled from a neighbour


def test_map_and_aggregate(tiny_masks):
    data = OiF(tiny_masks)
    scene = data[0]
    fixations = filter_fixations(data.fixations(), shape=scene.shape)
    mapped = scene.map_fixations(fixations)
    assert {"mask_id", "label"} <= set(mapped.columns)

    per_object = object_fixations(mapped, scene.label_map, scene.labels, scene=scene.name)
    assert per_object["n_fixations"].sum() == len(mapped)
    # every object appears, including any nobody looked at
    assert set(per_object["mask_id"]) >= set(region_ids(scene.label_map))


def test_fill_objects_round_trip():
    label_map = np.zeros((20, 20), np.int32)
    label_map[0:5, 0:5] = 1
    label_map[10:15, 10:15] = 2
    filled = fill_objects(label_map, {1: 3.5, 2: -1.0})
    assert filled[2, 2] == 3.5
    assert filled[12, 12] == -1.0
    assert filled[18, 18] == 0.0


def test_rescale_fixations_fit_mode():
    df = pd.DataFrame({"x": [0.0, 640.0], "y": [0.0, 480.0]})
    out = rescale_fixations(df, from_shape=(480, 640), to_shape=(1050, 1680))
    assert out["y"].tolist() == [0.0, 1050.0]
    fitted = rescale_fixations(df, (480, 640), (1050, 1680), mode="fit")
    assert fitted["x"].iloc[0] > 0  # letterboxed, so x is padded inward


# -- features and model ----------------------------------------------------

def test_object_features_shapes(tiny_masks):
    data = OiF(tiny_masks)
    scene = data[0]
    table = scene.features()
    assert len(table) == len(region_ids(scene.label_map))
    assert (table["size"] > 0).all()
    assert (table["ecc"] >= 0).all()
    assert table["depth"].notna().all()


def test_features_reject_mismatched_maps(tiny_masks):
    data = OiF(tiny_masks)
    scene = data[0]
    with pytest.raises(ValueError, match="shape"):
        object_features(scene.label_map, salience=np.zeros((10, 10)))


def test_model_recovers_a_known_relationship():
    rng = np.random.default_rng(3)
    n = 400
    log_size = rng.normal(8, 1.5, n)
    z_ecc = rng.normal(0, 1, n)
    log_sum = 0.6 * log_size - 0.4 * z_ecc + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"log_size": log_size, "z_ecc": z_ecc, "log_sum": log_sum})
    model = ObjectAttentionModel(terms=["log_size", "z_ecc"]).fit(df)
    assert model.coefficients["log_size"] == pytest.approx(0.6, abs=0.05)
    assert model.coefficients["z_ecc"] == pytest.approx(-0.4, abs=0.05)
    assert model.score(df)["r2"] > 0.95


def test_model_drops_all_missing_terms():
    df = pd.DataFrame({"log_size": [1.0, 2.0, 3.0, 4.0],
                       "z_salience": [np.nan] * 4,
                       "log_sum": [1.0, 2.0, 3.0, 4.0]})
    with pytest.warns(UserWarning, match="dropping model term"):
        model = ObjectAttentionModel(terms=["log_size", "z_salience"]).fit(df)
    assert model.used_terms_ == ["log_size"]
    assert model.n_obs_ == 4


def test_add_model_terms(tiny_masks):
    data = OiF(tiny_masks)
    scene = data[0]
    table = scene.object_table(data.fixations())
    out = add_model_terms(table)
    assert {"log_size", "log_sum", "log_depth", "z_ecc"} <= set(out.columns)
    assert np.isfinite(out["log_size"]).all()
