"""``oif`` command line: check, repair and export without writing any code.

    oif check                 is my copy of the data complete?
    oif labels                write derived/labels.csv (what each mask id is)
    oif repair                rebuild mask files that are missing or truncated
    oif objects               write a per-object table (features + fixations)
    oif fixations             read raw/ and report what is in it
    oif demo SCENE            save a picture of one scene with its objects

Every command takes ``--root PATH`` if the data is not in the current folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import __version__


def _dataset(args):
    from .datasets.oif import OiF
    return OiF(args.root)


# -- commands --------------------------------------------------------------

def cmd_check(args) -> int:
    data = _dataset(args)
    report = data.check()
    print(report)
    if not report.ok:
        print("\nRun `oif repair` to rebuild the mask files listed above "
              "from the annotations and depth maps.")
    return 0


def cmd_labels(args) -> int:
    data = _dataset(args)
    print(f"Recovering labels for {len(data)} scenes (about a second each)...")
    out = data.write_labels(args.out, progress=args.verbose)
    import pandas as pd
    table = pd.read_csv(out)
    low = int((table["match_score"] < 0.99).sum())
    print(f"\nWrote {len(table)} objects across {table['image'].nunique()} scenes")
    print(f"  -> {out}")
    print(f"  match score: min {table['match_score'].min():.4f}, "
          f"{low} row(s) below 0.99" + ("" if low else " (all confident)"))
    return 0


def cmd_repair(args) -> int:
    data = _dataset(args)
    report = data.check()
    broken = sorted({s for scenes in report.truncated.values() for s in scenes} |
                    {s for folder, scenes in report.missing.items() if folder == "masks"
                     for s in scenes})
    if not broken:
        print("Nothing to repair - every mask file is present and readable.")
        return 0

    import pandas as pd

    out_dir = Path(args.out) if args.out else data.data.dir("masks")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rebuilding {len(broken)} mask file(s) into {out_dir}")

    rows = []
    for name in broken:
        scene = data[name]
        path = out_dir / f"{name}.npy"
        if path.exists() and path.stat().st_size >= 1024 and not args.force:
            print(f"  skip {name} (already looks fine)")
            continue
        records = scene.rebuild_records()
        label_map = scene.build_label_map()
        np.save(path, label_map.astype(np.int32))
        print(f"  {name}: {len(records)} objects -> {path.name}")
        rows.extend({"image": name, "mask_id": r.mask_id, "label": r.label,
                     "occluded": r.occluded, "polygon_index": r.polygon_index,
                     "visible_fraction": round(r.visible_fraction, 6)} for r in records)

    if rows:
        # Record what was painted into which id. Without this the rebuilt
        # arrays would be anonymous integers all over again.
        registry = data.data.dir("derived", create=True) / "rebuilt_labels.csv"
        table = pd.DataFrame(rows)
        if registry.exists():
            old = pd.read_csv(registry)
            table = pd.concat([old[~old["image"].isin(table["image"])], table],
                              ignore_index=True)
        table.to_csv(registry, index=False)
        print(f"\nRecorded id -> label for the rebuilt scenes -> {registry}")

    data.data.refresh()
    print("Rebuilt masks come from the annotations plus the depth map, so their "
          "ids are internally consistent but need not match any older copy of "
          "these files.")
    return 0


def cmd_objects(args) -> int:
    data = _dataset(args)
    fixations = None
    if args.fixations:
        from .fixations import filter_fixations, load_fixations
        fixations = load_fixations(args.fixations)
        if not args.no_filter:
            fixations = filter_fixations(fixations)
        print(f"Loaded {len(fixations)} fixations over "
              f"{fixations['image'].nunique()} scenes")

    salience = None
    if args.salience:
        from .salience import load_salience_maps
        salience = load_salience_maps(args.salience, shape=(768, 1024))
        print(f"Loaded {len(salience)} salience maps")

    print(f"Measuring objects in {len(data)} scenes...")
    table = data.object_tables(fixations, salience=salience,
                               method=args.method, radius=args.radius,
                               progress=args.verbose)
    if fixations is not None:
        from .features import add_model_terms
        table = add_model_terms(table)

    out = Path(args.out) if args.out else data.data.dir("derived", create=True) / "objects.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nWrote {len(table)} objects -> {out}")
    if fixations is not None:
        looked = int((table["n_fixations"] > 0).sum())
        print(f"  {looked} of {len(table)} objects received at least one fixation")
    return 0


def cmd_fixations(args) -> int:
    from .fixations import filter_fixations, load_fixations, summarize
    data = _dataset(args)
    src = args.path or data.data.dir("raw")
    if not Path(src).exists():
        print(f"No fixation folder at {src}.\n"
              "Put your fixation data (CSV reports or the released .npy maps) "
              "in a folder called raw/ next to images/, or pass a path: "
              "oif fixations --path my_data.csv")
        return 1
    df = load_fixations(src)
    print(f"Read {len(df)} fixations from {src}")
    print(f"  columns recognised: {[c for c in df.columns if df[c].notna().any()]}")
    clean = filter_fixations(df)
    print(f"  {len(clean)} remain after the standard cleaning steps "
          f"({len(df) - len(clean)} dropped)")
    print(f"  {clean['image'].nunique()} scenes, {clean['subject'].nunique()} viewers")
    print()
    print(summarize(clean).head(15).to_string(index=False))
    if args.out:
        clean.to_csv(args.out, index=False)
        print(f"\nWrote cleaned fixations -> {args.out}")
    return 0


def cmd_demo(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .viz import show_objects
    data = _dataset(args)
    scene = data[args.scene] if args.scene else data[0]
    fig, ax = plt.subplots(figsize=(10, 7.5))
    show_objects(scene.image, scene.label_map, ax=ax,
                 labels=scene.labels if args.labels else None,
                 title=scene.name)
    out = Path(args.out or f"{scene.name}_objects.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{scene.name}: {len(scene.object_ids)} objects -> {out}")
    return 0


def cmd_stats(args) -> int:
    data = _dataset(args)
    for key, value in data.stats().items():
        print(f"{key:>20}: {value}")
    return 0


# -- parser ----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="oif",
        description="Objects in Focus - map fixations to segmented objects.",
        epilog="Start with `oif check`. Docs: https://github.com/ehhall/objects-in-focus",
    )
    p.add_argument("--version", action="version", version=f"oif {__version__}")

    # Shared options, accepted either before or after the subcommand so that
    # `oif --root X check` and `oif check --root X` both work. The
    # subcommand copies use SUPPRESS so that leaving them off does not
    # overwrite a value given before the subcommand.
    p.add_argument("--root", default=None,
                   help="dataset folder (default: search from the current directory)")
    p.add_argument("-v", "--verbose", action="store_true", help="print each scene")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=argparse.SUPPRESS,
                        help="dataset folder (default: search from the current directory)")
    common.add_argument("-v", "--verbose", action="store_true",
                        default=argparse.SUPPRESS, help="print each scene")

    class _Sub(argparse.ArgumentParser):
        def __init__(self, **kwargs):
            kwargs.setdefault("parents", [common])
            super().__init__(**kwargs)

    sub = p.add_subparsers(dest="command", parser_class=_Sub)

    s = sub.add_parser("check", help="report missing or damaged data files")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("labels", help="write derived/labels.csv (mask id -> object)")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_labels)

    s = sub.add_parser("repair", help="rebuild missing or truncated mask files")
    s.add_argument("--out", default=None, help="write rebuilt masks here instead")
    s.add_argument("--force", action="store_true", help="overwrite intact files too")
    s.set_defaults(func=cmd_repair)

    s = sub.add_parser("objects", help="write a per-object feature/fixation table")
    s.add_argument("--fixations", default=None, help="fixation file or folder")
    s.add_argument("--salience", default=None, help="folder of salience maps")
    s.add_argument("--method", default="point", choices=["point", "disc", "nearest"])
    s.add_argument("--radius", type=int, default=0,
                   help="pixels of gaze uncertainty for disc/nearest")
    s.add_argument("--no-filter", action="store_true",
                   help="skip the standard fixation cleaning")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_objects)

    s = sub.add_parser("fixations", help="read raw/ and describe what is there")
    s.add_argument("--path", default=None)
    s.add_argument("--out", default=None, help="save the cleaned table here")
    s.set_defaults(func=cmd_fixations)

    s = sub.add_parser("demo", help="save a picture of one scene with its objects")
    s.add_argument("scene", nargs="?", default=None)
    s.add_argument("--labels", action="store_true", help="write object names on it")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("stats", help="headline dataset numbers")
    s.set_defaults(func=cmd_stats)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("\nIf the data is somewhere else, pass --root /path/to/objects-in-focus",
              file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
