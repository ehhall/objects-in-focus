# COCO-Freeview

A hundred scenes is enough to fit a model and not enough to know it
generalises. The COCO-Freeview adapter runs the same analysis over MS-COCO
images, which is how the object-based attention model was tested outside its
home dataset — and how it was found to travel poorly (R² 0.82 → 0.66, and
0.49 when the two sets are pooled without a dataset term).

## What you need

Three public pieces, which this repository does not redistribute:

1. **COCO-Freeview fixations** — free-viewing eye movements over MS-COCO
   images, from [Chen et al., 2022](https://openaccess.thecvf.com/content/CVPR2022W/EPIC/html/Chen_Characterizing_Target-Absent_Human_Attention_CVPRW_2022_paper.html).
2. **The images** — MS-COCO train2017/val2017, https://cocodataset.org
3. **COCO-Stuff annotations** — `stuffthingmaps_trainval2017.zip`, from
   [Caesar et al., 2018](https://github.com/nightrome/cocostuff)

Arrange them like this:

```
coco-freeview/
    images/000000081074.jpg
    stuff/000000081074.png        the COCO-Stuff label map
    depth/000000081074.npy        optional, MonoDepth2 output
    labels.txt                    "id: name" per line, from COCO-Stuff
```

## Using it

```python
from oif import COCOFreeview, load_fixations, filter_fixations

coco = COCOFreeview("path/to/coco-freeview")
fixations = filter_fixations(load_fixations("coco-freeview-fixations.csv"),
                             shape=(1050, 1680))

scene = coco["000000081074"]
scene.image        # placed on the 1680 x 1050 presentation canvas
scene.label_map    # instance ids, not category ids
scene.labels       # {instance_id: 'chair', ...}

table = coco.object_tables(fixations)
```

## The two things that make COCO different

### One id for every instance of a category

COCO-Stuff labels all the chairs in a room `chair`. Four chairs under one id
is one object as far as any per-object measure is concerned, which is wrong:
their sizes add up, their centroid lands between them, and the fixation counts
pool. So each category mask is split into spatially separate blobs:

```python
from oif.masks import split_components
blobs = split_components(mask, min_size=4, connectivity=4, downscale=(35, 56))
```

The published analysis ran connected components at 1/30th scale — small blobs
merge, single-pixel bridges between genuinely separate objects break — then
upscaled and measured at full resolution. `instance_label_map` does the whole
sequence, including dropping categories that cover less than 0.05% of the
image and the amorphous stuff classes (`STUFF_BACKGROUND_IDS`).

Note what this cannot do: two chairs that visually touch stay one blob. It is
a heuristic for instance separation, not instance segmentation.

### Images were shown letterboxed

Scenes were scaled to a 1050-pixel height on a 1680-pixel-wide display and
centred, with anything wider cropped. Fixations are in screen coordinates and
the segmentations are in image coordinates, so one has to be moved onto the
other. `fit_to_display` reproduces the presentation geometry:

```python
from oif.datasets.coco import fit_to_display
on_screen = fit_to_display(image, (1050, 1680))
```

It is applied automatically to images, stuff maps and depth maps. Label maps
are resized nearest-neighbour, never interpolated — averaging two ids gives a
third, unrelated id.

If your fixations are in some other display geometry, convert them first:

```python
from oif.mapping import rescale_fixations
fixations = rescale_fixations(fixations, from_shape=(768, 1024),
                              to_shape=(1050, 1680), mode="fit")
```

## Comparing the two datasets

```python
import pandas as pd
from oif import add_model_terms, ObjectAttentionModel

oif_table = add_model_terms(data.object_tables(oif_fixations))
coco_table = add_model_terms(coco.object_tables(coco_fixations))
oif_table["dataset"] = "oif"
coco_table["dataset"] = "coco"

both = pd.concat([oif_table, coco_table], ignore_index=True)
```

Two things to be careful about when pooling. Salience values from different
model runs are not on a common scale, so z-score within dataset:
`add_model_terms(both, by=["dataset"])`. And viewing conditions differ —
viewers per image, scene perspective, retinal object size — which is why the
pooled model needs a dataset term to recover its fit. Adding one is not a
patch on a nuisance; it is the finding.
