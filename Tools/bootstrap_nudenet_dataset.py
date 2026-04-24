#!/usr/bin/env python3
"""
bootstrap_nudenet_dataset.py — Extract photos from the macOS Photos
Library and pre-label them with NudeNet for custom-detector fine-
tuning.

=======================================================================
SAFETY GUARANTEE (read this before running):

  This script is READ-ONLY on the Photos Library.

  Concretely:
    * `osxphotos` opens the Photos SQLite database in read-only mode.
      There is no API path from osxphotos that modifies the library.
    * The only file-system operations this script performs are:
        - READS from the library (via osxphotos' export, which COPIES)
        - WRITES to a user-specified `--output` directory
    * `--output` is validated to live OUTSIDE the Photos Library
      directory. If the paths overlap in either direction, the script
      aborts before touching anything.
    * No osxphotos method that could modify the library (e.g. `add`,
      `remove`, `update_metadata`) is imported or called.
    * The NudeNet detector runs entirely on the copied files in the
      output directory. It never sees the library's internal paths.

  If you pass `--dry-run`, the script only enumerates candidates and
  prints their UUIDs. No I/O to anywhere.

=======================================================================
INSTALL

    pip install osxphotos nudenet pillow pillow-heif tqdm

    # If you want the 640m NudeNet variant instead of the bundled 320n:
    curl -L -O https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx

=======================================================================
USAGE

    # Dry-run against an album — lists what WOULD be processed, no I/O
    python3 Tools/bootstrap_nudenet_dataset.py \\
        --output ~/Desktop/nude-dataset \\
        --album "Training" \\
        --dry-run

    # Export 50 photos from an album as a smoke test, with NudeNet
    # pre-labels at confidence ≥ 0.25
    python3 Tools/bootstrap_nudenet_dataset.py \\
        --output ~/Desktop/nude-dataset \\
        --album "Training" \\
        --limit 50 \\
        --confidence 0.25 \\
        --model-path ~/Downloads/640m.onnx

    # Full-library run (big!) — requires explicit --no-album-filter
    python3 Tools/bootstrap_nudenet_dataset.py \\
        --output ~/Desktop/nude-dataset \\
        --no-album-filter

=======================================================================
OUTPUT LAYOUT

    <output>/
    ├── images/           — exported JPEGs (HEIC/RAW auto-converted)
    ├── labels/           — YOLO-format .txt sidecars
    ├── classes.txt       — one NudeNet label per line, index = class_id
    └── data.yaml         — Ultralytics dataset config; wire `path:` to
                            the train+val splits you carve out post-review

The labels are *candidate* annotations from a known-biased model.
Manual review (CVAT, Label Studio, LabelImg) is REQUIRED before
using this dataset to train anything. See `README-review.md` in the
output for the checklist.
"""

import argparse
import sys
from pathlib import Path

# NudeNet v3.4 default label ordering. Must match the training export.
NUDENET_LABELS = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]
LABEL_TO_ID = {name: i for i, name in enumerate(NUDENET_LABELS)}


def resolve_safely(path_str: str) -> Path:
    """Expand ~ and resolve symlinks; abort if resolution fails."""
    try:
        return Path(path_str).expanduser().resolve(strict=False)
    except Exception as exc:
        sys.exit(f"ERROR: could not resolve path {path_str!r}: {exc}")


def resolve_library_arg(raw: str) -> Path:
    """
    Normalize a user-supplied `--library` path and sanity-check it
    before handing to osxphotos. Accepts:

      * a `.photoslibrary` bundle directory (the common case — that's
        what Finder shows and what the user Cmd-C's from the
        sidebar).
      * a direct path to `database/Photos.sqlite` inside a bundle
        (rare, but osxphotos supports it so we do too).

    Exits with a specific error message if the path doesn't exist
    or clearly isn't a Photos Library, so the failure comes from us
    (with context) rather than from deep inside osxphotos.
    """
    path = resolve_safely(raw)
    if not path.exists():
        sys.exit(f"ERROR: --library path does not exist: {path}")

    # Bundle directory case.
    if path.is_dir():
        if path.suffix.lower() != ".photoslibrary":
            sys.exit(
                f"ERROR: --library directory {path} isn't a "
                "`.photoslibrary` bundle. Pass the bundle itself "
                "(the folder Finder shows as a single Photos Library "
                "with a badge), or the internal Photos.sqlite path."
            )
        return path

    # SQLite file case.
    if path.is_file():
        if path.name.lower() != "photos.sqlite":
            sys.exit(
                f"ERROR: --library file {path} isn't a Photos "
                "SQLite database. Expected `Photos.sqlite` inside a "
                "library's `database/` subfolder, or pass the "
                "`.photoslibrary` bundle instead."
            )
        return path

    sys.exit(
        f"ERROR: --library {path} is neither a `.photoslibrary` "
        "bundle nor a Photos.sqlite file."
    )


def assert_disjoint(library: Path, output: Path) -> None:
    """
    Hard safety check: the output directory must not live inside the
    Photos Library, nor vice versa. Failing this aborts *before any
    file-system write*.
    """
    try:
        output.relative_to(library)
        sys.exit(
            f"ERROR: output {output} is inside the Photos Library "
            f"({library}). Refusing to write anywhere under the "
            "library path."
        )
    except ValueError:
        pass

    try:
        library.relative_to(output)
        sys.exit(
            f"ERROR: the Photos Library {library} is inside the output "
            f"directory {output}. Refusing to write a sibling output "
            "tree that could shadow or overlap the library."
        )
    except ValueError:
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract + pre-label photos from the macOS Photos "
                    "Library for NudeNet fine-tuning. READ-ONLY on the "
                    "library.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", required=True,
                   help="Output directory (must NOT be inside the library).")
    p.add_argument("--library", default=None,
                   help="Path to a non-system Photos Library. Accepts "
                        "either a `.photoslibrary` bundle (e.g. "
                        "`/Volumes/Backup/Archive.photoslibrary`) or a "
                        "direct path to its internal `Photos.sqlite`. "
                        "Omit to use the system library. `~` is "
                        "expanded.")

    filter_group = p.add_argument_group("photo selection")
    mutex = filter_group.add_mutually_exclusive_group(required=True)
    mutex.add_argument("--album", help="Album name to export from.")
    mutex.add_argument("--uuid-file",
                       help="Plain-text file, one photo UUID per line.")
    mutex.add_argument("--no-album-filter", action="store_true",
                       help="Process the entire library (big!).")
    filter_group.add_argument("--limit", type=int, default=None,
                              help="Cap the number of photos processed.")

    label_group = p.add_argument_group("labeling")
    label_group.add_argument("--skip-labeling", action="store_true",
                             help="Export only, skip NudeNet inference.")
    label_group.add_argument("--model-path", default=None,
                             help="Path to NudeNet .onnx. Omit to use the "
                                  "bundled 320n model.")
    label_group.add_argument("--inference-resolution", type=int, default=320,
                             help="NudeNet input size (320 bundled, 640 "
                                  "for the 640m variant).")
    label_group.add_argument("--confidence", type=float, default=0.25,
                             help="Minimum confidence for pre-labels.")

    run_group = p.add_argument_group("run control")
    run_group.add_argument("--dry-run", action="store_true",
                           help="Enumerate photos, print UUIDs, do no I/O.")
    run_group.add_argument("--convert-jpeg", action="store_true",
                           default=True,
                           help="Convert HEIC/RAW to JPEG on export. "
                                "On by default.")
    run_group.add_argument("--keep-originals", action="store_true",
                           help="Disable JPEG conversion. HEIC/RAW require "
                                "pillow-heif / rawpy to label afterward.")

    return p


def main() -> None:
    args = build_parser().parse_args()

    # Resolve + validate --library *before* importing osxphotos so
    # path-argument mistakes (wrong drive, typo) report with a
    # specific error even if osxphotos isn't installed yet. Accepts
    # either a `.photoslibrary` bundle or the internal Photos.sqlite.
    library_arg: Path | None = None
    if args.library:
        library_arg = resolve_library_arg(args.library)

    # Import osxphotos late so --help + library-path validation work
    # without the dependency.
    try:
        import osxphotos
    except ImportError:
        sys.exit("ERROR: osxphotos not installed. Run: pip install osxphotos")

    # Open the library (read-only on Photos.sqlite by osxphotos design).
    try:
        if library_arg is not None:
            library = osxphotos.PhotosDB(dbfile=str(library_arg))
        else:
            library = osxphotos.PhotosDB()
    except Exception as exc:
        sys.exit(f"ERROR: could not open Photos Library: {exc}")

    library_path = resolve_safely(library.library_path)
    output_path = resolve_safely(args.output)

    # Hard safety check. Aborts before any write if paths overlap.
    assert_disjoint(library_path, output_path)

    print(f"Photos Library (read-only): {library_path}")
    print(f"Output directory:           {output_path}")
    print()

    # Enumerate candidate photos.
    if args.uuid_file:
        uuids = [u.strip() for u in open(args.uuid_file).read().splitlines()
                 if u.strip()]
        photos = library.photos(uuid=uuids)
    elif args.album:
        photos = library.photos(albums=[args.album])
    else:
        photos = library.photos()

    # Drop trashed / hidden / missing photos — osxphotos can hand
    # these back but we don't want to touch them. (`intrash`, not
    # `trashed`, is the current osxphotos attribute name.)
    photos = [p for p in photos if not p.intrash and not p.hidden
              and p.path is not None]

    if args.limit:
        photos = photos[:args.limit]

    print(f"Candidates: {len(photos)} photo(s)")
    if not photos:
        print("Nothing to do.")
        return

    if args.dry_run:
        preview = photos[:20]
        for p in preview:
            print(f"  [dry-run] {p.uuid} {p.original_filename}")
        if len(photos) > 20:
            print(f"  ... and {len(photos) - 20} more")
        print("\nDry-run complete. No files written.")
        return

    # Create output directories.
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Initialize NudeNet if we're labeling.
    detector = None
    if not args.skip_labeling:
        try:
            from nudenet import NudeDetector
        except ImportError:
            sys.exit("ERROR: nudenet not installed. Run: pip install nudenet")
        try:
            detector = NudeDetector(
                model_path=args.model_path,
                inference_resolution=args.inference_resolution,
            )
        except Exception as exc:
            sys.exit(f"ERROR: NudeNet init failed: {exc}")

    # Lazy imports after we know we're going to run.
    from PIL import Image
    try:
        import pillow_heif  # noqa: F401  register HEIC opener with PIL
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **_): return it  # noqa: E731

    convert = args.convert_jpeg and not args.keep_originals

    exported_count = 0
    labeled_count = 0
    skipped_export = 0
    skipped_label = 0

    for photo in tqdm(photos, desc="Processing"):
        # osxphotos.export() COPIES the original out. It does not
        # modify the library. The boolean kwargs below only affect
        # the copied output.
        try:
            paths = photo.export(
                str(images_dir),
                edited=False,
                convert_to_jpeg=convert,
                overwrite=False,     # don't clobber an earlier run
            )
        except Exception as exc:
            print(f"  export failed for {photo.uuid}: {exc}", file=sys.stderr)
            skipped_export += 1
            continue

        if not paths:
            skipped_export += 1
            continue
        exported = Path(paths[0])
        exported_count += 1

        if detector is None:
            continue

        # Pre-label with NudeNet.
        try:
            detections = detector.detect(str(exported))
        except Exception as exc:
            print(f"  NudeNet failed on {exported.name}: {exc}",
                  file=sys.stderr)
            skipped_label += 1
            continue

        try:
            with Image.open(exported) as im:
                img_w, img_h = im.size
        except Exception as exc:
            print(f"  PIL failed on {exported.name}: {exc}", file=sys.stderr)
            skipped_label += 1
            continue

        label_path = labels_dir / (exported.stem + ".txt")
        wrote_any = False
        # Write atomically via a .tmp then rename so an interrupted
        # run doesn't leave partial labels.
        tmp_path = label_path.with_suffix(".txt.tmp")
        with open(tmp_path, "w") as out:
            for det in detections or []:
                score = float(det.get("score", 0))
                if score < args.confidence:
                    continue
                name = det.get("class", "")
                class_id = LABEL_TO_ID.get(name)
                if class_id is None:
                    continue
                box = det.get("box", [])
                if len(box) != 4:
                    continue
                x1, y1, x2, y2 = map(float, box)
                # NudeNet returns pixel coords on the source image.
                # YOLO wants normalized [cx, cy, w, h].
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                out.write(f"{class_id} {cx:.6f} {cy:.6f} "
                          f"{bw:.6f} {bh:.6f}\n")
                wrote_any = True
        tmp_path.replace(label_path)
        if wrote_any:
            labeled_count += 1

    # classes.txt — one label per line, index = class id.
    (output_path / "classes.txt").write_text(
        "\n".join(NUDENET_LABELS) + "\n"
    )

    # data.yaml — Ultralytics dataset config. `path:` intentionally
    # left for the user to fill in after they carve train/val splits.
    (output_path / "data.yaml").write_text(
        "# Ultralytics YOLOv8 dataset config for the NudeNet bootstrap.\n"
        "# Fill in `path:` once you've split images/labels into\n"
        "# train/val subsets.\n"
        "\n"
        f"path: {output_path}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "\n"
        f"nc: {len(NUDENET_LABELS)}\n"
        "names:\n"
        + "\n".join(f"  {i}: {n}" for i, n in enumerate(NUDENET_LABELS))
        + "\n"
    )

    # README-review — checklist for the human-review step.
    (output_path / "README-review.md").write_text(
        "# NudeNet bootstrap dataset — review checklist\n\n"
        "The labels in `labels/` are *pre-labels* from a known-biased\n"
        "model. Before using them to train anything, review every\n"
        "image in an annotation tool (CVAT / Label Studio / LabelImg)\n"
        "and correct:\n\n"
        "- [ ] Missed detections, especially `MALE_GENITALIA_EXPOSED`\n"
        "      and `FACE_MALE` (NudeNet's known weak spots).\n"
        "- [ ] Wrong-class detections (FACE_MALE vs FACE_FEMALE flips).\n"
        "- [ ] Tight box fit — NudeNet's boxes can be loose.\n"
        "- [ ] Duplicate boxes on the same subject.\n\n"
        "After review, carve `images/` + `labels/` into `train/` and\n"
        "`val/` subdirectories (~80/20), update `data.yaml`'s `path:`,\n"
        "and fine-tune with Ultralytics YOLOv8m using the upstream\n"
        "640m.onnx's `.pt` checkpoint as starting weights.\n"
    )

    print()
    print(f"Exported: {exported_count} image(s) "
          f"(skipped {skipped_export})")
    if detector is not None:
        print(f"Labeled:  {labeled_count} image(s) with ≥1 detection "
              f"(skipped {skipped_label})")
    print(f"Output:   {output_path}")
    print()
    print("NEXT STEPS")
    print("  1. Review labels in an annotation tool (see "
          "README-review.md).")
    print("  2. Split images + labels into train/val subsets.")
    print("  3. Fine-tune per data.yaml.")


if __name__ == "__main__":
    main()
