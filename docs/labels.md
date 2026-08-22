# Recovering what each mask id is

The published `masks/` arrays hold integer ids and nothing else. Nothing in
the release said that id 7 in the bakery is the bread. This page explains how
`oif labels` recovers that mapping, and why you can trust it.

Short version:

```bash
oif labels
```

writes `derived/labels.csv` with a row per object and a `match_score` for
each. Across all 100 scenes and 2,427 objects, the lowest score is **0.993**
and none falls below 0.99.

## The problem

A label map is the result of painting polygons into an array one after
another, each with its own id. Given only the finished array, the id-to-object
mapping has been thrown away. You cannot simply ask which polygon contains a
region, because a small visible sliver of a chair also sits inside the wall
polygon behind it, inside the floor polygon under it, and so on. On this
dataset, matching purely by containment picks a different-labelled polygon
that fits just as well for **65% of regions** — worse than useless.

## The idea

Painting order is recoverable because it is baked into the result.

The object painted **last** has the highest id, and nothing was painted over
it, so its region is exactly its polygon. Remove those pixels from
consideration and the same is true of the next-highest id: its region is its
polygon minus the part just claimed. Work down from the top and every id has
exactly one polygon whose *still-unclaimed* area coincides with the region.

So: for each id from highest to lowest, score every unused polygon by the
intersection-over-union between the region and the polygon's unclaimed part,
take the best, mark the pixels claimed, and move on.

```python
from oif import recover_labels
recovered = recover_labels(scene.label_map, scene.annotation)

for r in recovered[:3]:
    print(r.mask_id, r.label, round(r.score, 4), r.occluded)
```

```
2 rock 1.0 1
3 water 0.9959 1
4 sand 1.0 0
```

## The score

`score` is that IoU. 1.0 means the region and the unclaimed part of the
polygon agree pixel for pixel — the match is not a guess, it is an identity.
Values slightly below 1.0 come from rasterisation edges, a pixel here and
there along a boundary.

`RecoveredObject.confident` is `score >= 0.99`. Filter on it if you are doing
something where a wrong label would be expensive:

```python
labels = {r.mask_id: r.label for r in recovered if r.confident}
```

## Two details that matter

**Background surfaces.** Some scenes' masks include the sky and the floor as
ids; some do not. Guessing wrong costs accuracy on every region in the scene,
so `include_background="auto"` (the default) runs the match both ways and
keeps whichever explains the label map better.

**Rasterisation.** Polygon vertices are floats and have to become pixels. The
published masks were made by truncating coordinates, not rounding them, so the
package truncates too. Rounding instead drops the mean match score from 1.000
to 0.966, with three quarters of regions falling below the 0.99 confidence
line — the difference between "this is the same object" and "this is probably
the same object".

## Scenes that were rebuilt

For the nine scenes whose mask files were uploaded truncated, there is nothing
to recover from: `oif repair` paints a new label map and writes down what it
painted into `derived/rebuilt_labels.csv`. Those labels are exact by
construction and carry `score = 1.0`, and the `rebuilt` column in `labels.csv`
marks them so you can tell the two provenances apart.

## Output columns

| column | meaning |
|---|---|
| `image` | scene name |
| `mask_id` | id in the label map |
| `label` | object category |
| `occluded` | annotator's occlusion flag |
| `polygon_index` | position of the source polygon in the XML |
| `area` | visible pixels |
| `visible_fraction` | visible pixels ÷ full polygon area |
| `match_score` | IoU of the match, 1.0 = exact |
| `rebuilt` | whether this scene's label map was rebuilt by the package |

`visible_fraction` is worth a look in its own right: it is how much of an
object survived occlusion, and across the dataset it averages 0.86.

## Checking it yourself

The recovery is deterministic and takes about half a second per scene, so
verify rather than trust:

```python
from oif import OiF
data = OiF()
table = data.label_table()
print(table["match_score"].describe())
print(table.loc[table["match_score"] < 0.99])
```

Or visually — draw the recovered names onto the scene and see whether the
sofa says sofa:

```python
from oif.viz import show_objects
show_objects(scene.image, scene.label_map, labels=scene.labels)
```
