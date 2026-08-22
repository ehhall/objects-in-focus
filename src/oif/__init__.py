"""Objects in Focus - datasets and tools for mapping fixations to objects.

A suite for the question "what was this person looking at?" when the answer
has to be an object in a cluttered, real-world scene rather than a point on
a screen.

    from oif import OiF

    data = OiF("path/to/objects-in-focus")
    scene = data["target_kitchen_IDS01"]

    fixations = data.fixations()                 # everything in raw/
    looked_at = scene.map_fixations(fixations, method="disc", radius=25)
    per_object = scene.object_table(fixations)   # features + fixation counts

The pieces, if you want them separately:

``oif.paths``        find and validate a data tree
``oif.annotations``  CVAT polygon XML -> objects
``oif.masks``        polygons <-> label maps, label recovery, instance splitting
``oif.depth``        MonoDepth2 disparity -> usable depth maps
``oif.fixations``    read and clean eye-movement reports
``oif.mapping``      fixations -> objects
``oif.features``     size, eccentricity, depth, salience per object
``oif.model``        the object-based attention model
``oif.viz``          outlines, heatmaps, scanpaths
``oif.datasets``     OiF and COCO-Freeview

Citation and licence: see CITATION.cff and README.md in the repository.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .annotations import (
    BACKGROUND_LABELS,
    Polygon,
    SceneAnnotation,
    read_annotation,
)
from .datasets.coco import COCOFreeview, COCOScene
from .datasets.oif import OiF, Scene
from .depth import load_depth, to_depth
from .features import add_model_terms, object_features
from .fixations import (
    density_map,
    filter_fixations,
    fixation_array,
    gaussian_blur,
    load_fixation_map,
    load_fixations,
    map_to_fixations,
)
from .mapping import (
    assign_fixations,
    fill_objects,
    map_fixations,
    object_fixations,
    object_value_map,
)
from .masks import (
    build_label_map,
    recover_labels,
    region_ids,
    resize_label_map,
    split_components,
)
from .model import PUBLISHED_FIT, ObjectAttentionModel
from .paths import DataRoot, find_data_root

__all__ = [
    "__version__",
    # datasets
    "OiF", "Scene", "COCOFreeview", "COCOScene",
    # annotations
    "read_annotation", "SceneAnnotation", "Polygon", "BACKGROUND_LABELS",
    # masks
    "build_label_map", "recover_labels", "region_ids", "resize_label_map",
    "split_components",
    # depth
    "load_depth", "to_depth",
    # fixations
    "load_fixations", "load_fixation_map", "map_to_fixations",
    "filter_fixations", "fixation_array", "density_map", "gaussian_blur",
    # mapping
    "map_fixations", "assign_fixations", "object_fixations", "fill_objects",
    "object_value_map",
    # features + model
    "object_features", "add_model_terms", "ObjectAttentionModel", "PUBLISHED_FIT",
    # paths
    "DataRoot", "find_data_root",
]
