# API reference

Grouped by what you are trying to do. Everything listed as `oif.thing` is
importable straight from the top level: `from oif import thing`.

## Opening data

```python
OiF(root=None)                 # the dataset; root defaults to a search upward
data.scene_names               # list of scene names
data[i] / data["target_bar"]   # a Scene (case-insensitive)
data.check()                   # integrity report
data.stats()                   # headline numbers
data.summary()                 # per-scene polygon and object counts
data.fixations(folder="raw")   # load every fixation file in raw/
```

### `Scene`

| attribute | what it is |
|---|---|
| `.image` | (768, 1024, 3) uint8 |
| `.label_map` | (768, 1024) int — object id per pixel |
| `.labels` | `{mask_id: label}` |
| `.depth` | (768, 1024) float — 0 near, 1 far |
| `.annotation` | parsed CVAT polygons |
| `.object_ids` | non-zero ids present |
| `.shape` | (768, 1024) |
| `.mask_file_ok` | is the published mask file intact? |
| `.was_rebuilt` | did this label map come from `oif repair`? |

| method | what it does |
|---|---|
| `.fixation_map(task="memorize")` | the released binary fixation map, `raw/<scene>_<task>.npy` |
| `.fixations(task="memorize")` | those fixations as a canonical table |
| `.map_fixations(df, method, radius)` | fixation table + `mask_id`, `label` |
| `.object_table(df, salience, ...)` | one row per object: features + counts |
| `.features(salience)` | one row per object: features only |
| `.density(df, fc=6)` | blurred fixation density map |
| `.object_mask(id)` | boolean mask for one object |
| `.recover_labels()` | match ids back to polygons, with scores |
| `.build_label_map()` | repaint the label map from polygons + depth |

## Annotations

```python
read_annotation(path)          # -> SceneAnnotation
ann.polygons                   # every polygon
ann.objects(background=...)    # polygons minus background surfaces
ann.background(...)            # the surfaces
BACKGROUND_LABELS              # the eleven default surface labels
```

`Polygon` carries `.label`, `.points` (n×2 float), `.occluded`, `.z_order`,
`.index`, `.bbox`, `.closed()`.

## Masks and label maps

```python
build_label_map(annotation, depth, ...)   # -> (label_map, ordered polygons)
recover_labels(label_map, annotation)     # -> [RecoveredObject]
region_ids(label_map, min_fraction=0)     # ids present, optionally size-filtered
resize_label_map(label_map, (h, w))       # nearest-neighbour resize
split_components(mask, min_size, ...)     # one mask -> separate instances
rasterize(polygon, shape, value)          # fill one polygon
```

`oif.masks` also has `region_areas`, `centroids`, `polygon_masks`,
`depth_order`.

## Depth

```python
load_depth(path, shape, invert=True, rescale=True)
to_depth(disparity, shape, invert=True, rescale=True)
load_disparity(path)           # raw MonoDepth2 array, squeezed
depth_at(depth, (y, x))
normalize(depth, mask=None)
```

## Fixations

```python
load_fixations(path, columns=None)     # file or folder -> canonical table
load_fixation_map(path)                # one released raw/*.npy binary map
map_to_fixations(arr, image, task)     # binary map -> canonical table
filter_fixations(df, shape, min_duration=50, max_duration=1500, drop_first=True)
fixation_array(df, shape, weight="count"|"duration")
gaussian_blur(img, fc=6)               # Torralba low-pass filter
density_map(df, shape, weight, fc)
oif.fixations.summarize(df)            # per-scene counts
```

## Mapping fixations to objects

```python
map_fixations(df, label_map, labels, method="point", radius=0)
assign_fixations(x, y, label_map, method, radius)     # the array-level version
object_fixations(mapped, label_map, labels, scene)    # aggregate to objects
fill_objects(label_map, {id: value})                  # value per object -> image
object_value_map(label_map, pixel_values, stat)       # image -> value per object
oif.mapping.rescale_fixations(df, from_shape, to_shape, mode="stretch"|"fit")
```

`method` is `"point"`, `"disc"` or `"nearest"` — see the README section on
assignment rules.

## Features and model

```python
object_features(label_map, depth, salience, labels, scene, min_fraction=0)
add_model_terms(table, outcome="n_fixations", by=None)

model = ObjectAttentionModel(terms=["log_size", "log_depth", "z_ecc", "z_salience"])
model.fit(table)
model.predict(table, scale="log"|"count")
model.score(table)                     # r2, mae_log, mae_count
model.residuals(table, scale=...)
model.coefficients                     # pandas Series
model.summary(table)                   # statsmodels OLS summary
oif.model.cross_validate_by_scene(table, n_folds=5)
PUBLISHED_FIT                          # the paper's numbers, for comparison
```

## Salience

```python
oif.salience.load_salience_maps(folder, shape, strip=("_deepgaze",))
oif.salience.load_salience_map(path, shape)
oif.salience.center_bias(shape, sigma=None)     # the baseline to beat
oif.salience.salience_per_object(label_map, salience, stat="center")
```

## Plots

```python
from oif.viz import (show_scene, show_objects, show_fixations,
                     show_object_values, show_density, scene_grid,
                     outline_objects, overlay_labels, distinct_colors)
```

Each returns a matplotlib `Axes` (or a numpy image, for `outline_objects`), so
they drop into a figure you are already building.

## COCO-Freeview

```python
coco = COCOFreeview(root)
coco.image_ids
coco[i] / coco["000000081074"]         # a COCOScene
coco.object_tables(fixations)
oif.datasets.coco.fit_to_display(image, (1050, 1680))
oif.datasets.coco.instance_label_map(stuff_map, ...)
```

## Command line

```
oif check | labels | repair | objects | fixations | demo | stats
```

`oif <command> --help` for the options. `--root` works before or after the
command.
