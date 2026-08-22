"""Dataset wrappers: Objects in Focus, and COCO-Freeview for comparison."""

from .coco import COCOFreeview, COCOScene, fit_to_display
from .oif import OiF, Scene

__all__ = ["OiF", "Scene", "COCOFreeview", "COCOScene", "fit_to_display"]
