# Quickstart

Ten minutes, from nothing to a table of what people looked at.

If you would rather not install anything, do this in the browser instead:
[the Colab tutorial](https://colab.research.google.com/github/ehhall/objects-in-focus/blob/main/notebooks/objects_in_focus_tutorial.ipynb).

## 1. Install

```bash
git clone https://github.com/ehhall/objects-in-focus.git
cd objects-in-focus
pip install -e ".[full]"
```

`[full]` adds matplotlib and statsmodels, for pictures and model summaries.
Plain `pip install -e .` works if you only want the arrays and tables.

Check it worked:

```bash
oif check
```

You should see 100 scenes. If nine mask files are reported as truncated, run
`oif repair` — that is a known problem with the published upload, and the
command rebuilds them from the annotations.

## 2. Open a scene

```python
from oif import OiF

data = OiF()                          # or OiF("/path/to/objects-in-focus")
print(len(data), "scenes")
print(data.scene_names[:5])

scene = data["target_bakery"]
scene.image        # (768, 1024, 3) uint8 — the photograph
scene.label_map    # (768, 1024) int   — which object owns each pixel
scene.labels       # {3: 'bread', 4: 'counter', ...}
scene.depth        # (768, 1024) float — 0 near, 1 far
```

Nothing is loaded until you ask for it, and each piece is cached after the
first access, so opening all 100 scenes to look at one field is cheap.

## 3. Look at it

```python
import matplotlib.pyplot as plt
from oif.viz import show_objects

show_objects(scene.image, scene.label_map, labels=scene.labels)
plt.show()
```

## 4. Load the fixations

The released fixations live in `raw/`, one binary map per scene. Your own
eye-tracker exports can go in the same folder. Either way:

```python
fixations = data.fixations()
print(fixations.head())
```

The columns come back as `subject, image, fix_index, x, y, duration, task`
whatever they were called in the file. `image` must match the scene names —
`target_bakery`, with or without `.png`.

Clean them the standard way before analysis:

```python
from oif import filter_fixations
fixations = filter_fixations(fixations)     # 50–1500 ms, drop first, on-image only
```

## 5. Map them to objects

```python
looked_at = scene.map_fixations(fixations, method="disc", radius=25)
looked_at[["subject", "x", "y", "duration", "label"]].head()
```

Every fixation now carries the object it landed on. `method="disc"` allows 25
pixels of slop, which is about right for a well-calibrated desktop tracker;
use `method="point"` if you want the strict pixel-under-the-fixation answer.

## 6. Turn it into one row per object

```python
table = scene.object_table(fixations, method="disc", radius=25)
table.sort_values("n_fixations", ascending=False).head()
```

Or for the whole dataset at once:

```python
all_objects = data.object_tables(fixations, method="disc", radius=25)
all_objects.to_csv("objects.csv", index=False)
```

The same thing from the command line, if you would rather not write code:

```bash
oif objects --fixations raw/ --method disc --radius 25 --out objects.csv
```

## 7. Fit the attention model

```python
from oif import add_model_terms, ObjectAttentionModel

table = add_model_terms(all_objects)
model = ObjectAttentionModel().fit(table)

print(model.coefficients)
print(model.score(table))
```

Bigger objects, nearer objects and central objects get more fixations. If
your coefficients come out with those signs, the pipeline is working.

## Where to go next

- [`dataset.md`](dataset.md) — what every file is and how the arrays are laid out
- [`labels.md`](labels.md) — how mask ids were matched back to object names
- [`api.md`](api.md) — the full function list
- [`coco.md`](coco.md) — running the same analysis on COCO-Freeview

## Common problems

**`Could not find an Objects in Focus data root`** — you are not in the data
folder. Pass the path: `OiF("/path/to/objects-in-focus")`, or set
`OIF_DATA_ROOT`.

**Every object shows zero fixations** — the `image` column does not match the
scene names. Check `fixations["image"].unique()` against `data.scene_names`.

**`Could not recognise the fixation columns`** — name them yourself:
`load_fixations(path, columns={"x": ..., "y": ..., "image": ...})`.

**Coordinates look off by a scale factor** — your fixations are in screen
coordinates, not image coordinates. Convert with
`oif.mapping.rescale_fixations(df, from_shape=(1050, 1680), to_shape=(768, 1024))`.
