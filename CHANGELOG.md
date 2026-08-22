# Changelog

## 0.1.0 — 2025-08-21

First packaged release. The dataset was already public; this turns it into
something installable, documented and testable.

### Added

- **`oif` Python package** (`pip install -e .`) with modules for annotations,
  masks, depth, fixations, fixation-to-object mapping, per-object features,
  the object-based attention model, salience map loading, and plots.
- **The released fixations** in `raw/`: one 768 × 1024 binary map per scene
  (`target_<scene>_memorize.npy`), a 1 at every pixel a fixation landed on
  during the memorization task, pooled over viewers. `scene.fixation_map()`
  loads the array, `scene.fixations()` / `data.fixations()` turn it into the
  canonical fixation table, and the tutorial and README examples run on this
  real data rather than simulations.
- **`oif` command line**: `check`, `labels`, `repair`, `objects`,
  `fixations`, `demo`, `stats`.
- **Label recovery.** `oif labels` writes `derived/labels.csv`, mapping every
  mask id to its object. All 2,427 objects recover with a match score above
  0.99. The published release shipped no such lookup.
- **Mask repair.** `oif repair` rebuilds the nine mask files that were
  uploaded truncated (campsite, canal, canyon, casino, castle, cemetery,
  church, city, classroom) and records what it painted into each id.
- **COCO-Freeview adapter** with instance splitting and display-geometry
  handling.
- **Tutorial notebook** (`notebooks/objects_in_focus_tutorial.ipynb`), runnable
  in Colab with no local setup.
- **Documentation** under `docs/`, and a project website under GitHub Pages.
- **Test suite** (47 tests) and CI on Python 3.9–3.12.
- Licences: MIT for code, CC BY 4.0 for data. `CITATION.cff`.

### Fixed in the data

- Regenerated `figures/image_grid1.png`, which was also a truncated upload.
- File lookups are case-insensitive, working around two scenes spelled
  differently in `images/` and `annotations/` (`target_AtticStorage`,
  `target_atticRoom`).

### Notes for anyone comparing against the original notebooks

- Annotations are parsed as XML rather than by regular expression. The old
  pattern would have dropped any label containing a space or hyphen; on this
  dataset there are none, so no numbers change.
- Depth maps are rescaled to [0, 1] by default, so `log1p` is safe. Pass
  `rescale=False, invert=False` for raw MonoDepth2 disparity.
- Polygon rasterisation truncates vertex coordinates, matching how the
  published masks were made.
