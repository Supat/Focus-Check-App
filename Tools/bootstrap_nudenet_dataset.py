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

# NudeNet v3.4 default label ordering for indices 0..17 — these are
# the classes NudeNet's 640m detector emits, so keeping the indices
# exact lets the pre-labeler decode raw detections without a map.
#
# Indices 18..20 extend the schema with a three-class split of the
# old `MALE_GENITALIA_EXPOSED` (index 14). NudeNet itself can't
# predict the sub-classes; the review step is where boxes with
# class_id=14 get reclassified into 18 / 19 / 20 per RUBRIC.md.
# Any class-14 boxes remaining at training time should be dropped.
NUDENET_LABELS = [
    "FEMALE_GENITALIA_COVERED",       # 0
    "FACE_FEMALE",                    # 1
    "BUTTOCKS_EXPOSED",               # 2
    "FEMALE_BREAST_EXPOSED",          # 3
    "FEMALE_GENITALIA_EXPOSED",       # 4
    "MALE_BREAST_EXPOSED",            # 5
    "ANUS_EXPOSED",                   # 6
    "FEET_EXPOSED",                   # 7
    "BELLY_COVERED",                  # 8
    "FEET_COVERED",                   # 9
    "ARMPITS_COVERED",                # 10
    "ARMPITS_EXPOSED",                # 11
    "FACE_MALE",                      # 12
    "BELLY_EXPOSED",                  # 13
    "MALE_GENITALIA_EXPOSED",         # 14  — review-pending only
    "ANUS_COVERED",                   # 15
    "FEMALE_BREAST_COVERED",          # 16
    "BUTTOCKS_COVERED",               # 17
    # Review-added sub-classes of the old 14:
    "MALE_GENITALIA_FLACCID",         # 18
    "MALE_GENITALIA_AROUSAL",         # 19
    "MALE_GENITALIA_ORGASM",          # 20
]
LABEL_TO_ID = {name: i for i, name in enumerate(NUDENET_LABELS)}

# Classes that NudeNet itself can't predict; present in the schema
# only because the review step assigns them. If a user trains
# directly without review, these classes will have zero positives
# and the detection head will never learn them — intentional.
REVIEW_ONLY_CLASSES = {
    "MALE_GENITALIA_FLACCID",
    "MALE_GENITALIA_AROUSAL",
    "MALE_GENITALIA_ORGASM",
}

# Pixel padding applied around each detection's box when writing the
# crop thumbnail. Fractional margin on the shorter side; gives the
# reviewer enough context (hair / skin tone / surroundings) to
# disambiguate cases that a tight crop can't resolve.
CROP_PADDING_FRAC = 0.20

# Annotation rubric written to the output directory as RUBRIC.md.
# One paragraph per sub-class of the old MALE_GENITALIA_EXPOSED,
# with the hard edge cases called out so two reviewers would agree
# on the same photo most of the time.
RUBRIC_CONTENT = """# Annotation rubric for MALE_GENITALIA sub-classes

Every pre-label with class_id 14 (`MALE_GENITALIA_EXPOSED`) must be
reclassified into one of the three labels below during review.
Labels that can't confidently be committed should be **deleted**
— noisy training labels do more damage than fewer clean ones.

## 18. MALE_GENITALIA_FLACCID

Use for any exposed male genitalia with no visible erection.
Diagnostic: the shaft is in its baseline pendulous state, hanging
roughly parallel to gravity, with no rigidity and no circumferential
engorgement. Use this class in non-erotic contexts (bathing,
medical, locker-room, casualties, skinny-dipping) and also for the
post-ejaculatory resolution phase once detumescence has completed
— if there is no visible erection in *this* image, it is FLACCID
regardless of what the photo sequence suggests preceded it. Edge
case: a semi-erect / partially-engorged state where the shaft is
rigid enough to sit at a modest angle (say, 20–40° from vertical-
hanging) but is clearly not full-erect. Default to FLACCID unless
the rigidity is unambiguous.

## 19. MALE_GENITALIA_AROUSAL

Use for any visibly erect male genitalia. Diagnostic signs (all
three should be present for a confident call): the shaft is rigid
and held at a high angle (≥45° from vertical-hanging, often
pointing toward the navel), the glans is engorged and noticeably
redder or darker than its flaccid baseline, and the overall
circumference is larger than the flaccid comparator. Include every
stage from full erection through pre-ejaculation. Pre-ejaculate
fluid at the meatus — clear, small volume, no trajectory — remains
AROUSAL, not ORGASM. Do not use this class for ambiguous semi-erect
states; those belong to FLACCID.

## 20. MALE_GENITALIA_ORGASM

Use only when ejaculation is in progress or has just concluded with
*visible* evidence. Diagnostic: opaque white/off-white fluid (semen)
visibly present — either in mid-trajectory, or deposited on the
body/surroundings — in a volume noticeably larger than pre-
ejaculate. The genitals themselves will typically still be erect.
Clear pre-ejaculate alone is AROUSAL. Once the visible fluid has
been wiped, absorbed, or dried and the erection subsides, the
correct class becomes FLACCID again (post-resolution). When in
doubt between AROUSAL and ORGASM on a still frame, prefer AROUSAL
— ORGASM should be the rarer, unambiguous label.
"""


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
    run_group.add_argument("--name-prefix", default="",
                           help="String prepended to every exported "
                                "image / label / crop filename. Use to "
                                "avoid collisions when running multiple "
                                "libraries into the same output dir "
                                "(e.g. `laulea_`, `outdoor_`).")
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
    crops_dir = output_path / "crops"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

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

    # Set up the export pipeline. osxphotos' newer API moves
    # convert_to_jpeg off of PhotoInfo.export() onto ExportOptions,
    # which PhotoExporter consumes. Building it once outside the
    # loop avoids repeated validation.
    export_options = osxphotos.ExportOptions(
        convert_to_jpeg=convert,
        overwrite=False,  # don't clobber earlier-run output
        update=False,
    )

    exported_count = 0
    labeled_count = 0
    skipped_export = 0
    skipped_label = 0

    # Resolve a per-photo filename at iterate time. Earlier attempt
    # used osxphotos' `{original_name}` template string via
    # `filename=`, but `PhotoExporter.export()` treats that argument
    # as a literal filename (not a template) and wrote every file as
    # `laulea_{original_name}`, `(1)`, `(2)` — all colliding and
    # incrementing instead of each carrying the original stem.
    # Compute the target stem per photo explicitly.
    prefix = args.name_prefix or ""

    for photo in tqdm(photos, desc="Processing"):
        # PhotoExporter.export() COPIES the original (converting to
        # JPEG if requested). It does not modify the library. The
        # boolean options above only affect the copied output.
        try:
            exporter = osxphotos.PhotoExporter(photo)
            if prefix:
                # osxphotos' `filename` parameter is a *literal* full
                # filename, not a stem — we have to include the
                # extension ourselves. With convert_to_jpeg=True the
                # output is always JPEG; otherwise keep the source's
                # extension. Without an extension here, files end up
                # extensionless on disk (PIL still reads them, but
                # CVAT / LabelImg filter by extension at import time
                # and would skip them silently).
                orig = photo.original_filename or photo.uuid
                stem = Path(orig).stem
                if convert:
                    out_ext = ".jpeg"
                else:
                    src_ext = Path(orig).suffix.lower()
                    out_ext = src_ext if src_ext else ".jpeg"
                target = f"{prefix}{stem}{out_ext}"
                results = exporter.export(
                    str(images_dir),
                    filename=target,
                    options=export_options,
                )
            else:
                results = exporter.export(
                    str(images_dir), options=export_options
                )
            # `exported` holds final output paths regardless of whether
            # a JPEG conversion happened; `skipped` shows up when an
            # earlier run produced the same file (overwrite=False).
            paths = results.exported or results.skipped
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

        # Open the source image once for both label math and crop
        # writing; closing is handled below after all detections are
        # processed.
        try:
            source_image = Image.open(exported).convert("RGB")
        except Exception as exc:
            print(f"  PIL reopen failed on {exported.name}: {exc}",
                  file=sys.stderr)
            skipped_label += 1
            continue

        label_path = labels_dir / (exported.stem + ".txt")
        wrote_any = False
        # Write atomically via a .tmp then rename so an interrupted
        # run doesn't leave partial labels.
        tmp_path = label_path.with_suffix(".txt.tmp")
        with open(tmp_path, "w") as out:
            for det_idx, det in enumerate(detections or []):
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
                # NudeNet returns pixel-coord xywh — [x_topleft,
                # y_topleft, width, height] — NOT xyxy. Convert to
                # YOLO's normalized [cx, cy, w, h] per image size,
                # clamping the box to [0, 1] in image space.
                # NudeNet occasionally emits boxes that extend past
                # the image edge (the YOLO decoder can produce
                # negative coords or coords > image size); those
                # should be clipped rather than dropped.
                x_tl, y_tl, bw_px, bh_px = map(float, box)
                x1_px = max(0.0, x_tl)
                y1_px = max(0.0, y_tl)
                x2_px = min(float(img_w), x_tl + bw_px)
                y2_px = min(float(img_h), y_tl + bh_px)
                if x2_px <= x1_px or y2_px <= y1_px:
                    continue
                x1 = x1_px / img_w
                y1 = y1_px / img_h
                x2 = x2_px / img_w
                y2 = y2_px / img_h
                bw = x2 - x1
                bh = y2 - y1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                out.write(f"{class_id} {cx:.6f} {cy:.6f} "
                          f"{bw:.6f} {bh:.6f}\n")
                wrote_any = True

                # Per-detection crop thumbnail for fast review.
                # Padded outward by CROP_PADDING_FRAC on each side
                # (capped to image bounds) so the reviewer has a
                # little context around the box.
                pad = CROP_PADDING_FRAC * min(bw_px, bh_px)
                cx1 = int(max(0, x1_px - pad))
                cy1 = int(max(0, y1_px - pad))
                cx2 = int(min(img_w, x2_px + pad))
                cy2 = int(min(img_h, y2_px + pad))
                if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                    continue  # skip degenerate micro-crops
                class_dir = crops_dir / name
                class_dir.mkdir(exist_ok=True)
                crop_name = (f"{exported.stem}_{det_idx:02d}"
                             f"_{int(round(score * 100)):02d}.jpg")
                try:
                    source_image.crop((cx1, cy1, cx2, cy2)).save(
                        class_dir / crop_name, quality=85
                    )
                except Exception as exc:
                    # Crop failure is non-fatal — the full image +
                    # label already hit disk.
                    print(f"  crop save failed for {crop_name}: {exc}",
                          file=sys.stderr)
        source_image.close()
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
        "- [ ] Duplicate boxes on the same subject.\n"
        "- [ ] **Every `MALE_GENITALIA_EXPOSED` box (class 14) must be\n"
        "      reclassified** to one of `MALE_GENITALIA_FLACCID` (18),\n"
        "      `MALE_GENITALIA_AROUSAL` (19), or\n"
        "      `MALE_GENITALIA_ORGASM` (20) per the rules in\n"
        "      `RUBRIC.md`. Any class-14 boxes remaining when training\n"
        "      starts will be dropped (the class is present in the\n"
        "      schema only so NudeNet's pre-labels decode cleanly).\n\n"
        "## Fast-review flow using `crops/`\n\n"
        "Every pre-labeled detection is also saved as a thumbnail\n"
        "under `crops/<CLASS_NAME>/<image>_<det>_<score>.jpg`. Scroll\n"
        "`crops/MALE_GENITALIA_EXPOSED/` in Finder's icon view first\n"
        "— a dozen obvious false positives and correctly-NudeNet-\n"
        "caught cases are usually visible at a glance, which tells you\n"
        "how much work the detector-review step will actually involve\n"
        "before you open CVAT. Crops are NOT training inputs; they're\n"
        "review convenience.\n\n"
        "After review, carve `images/` + `labels/` into `train/` and\n"
        "`val/` subdirectories (~80/20), update `data.yaml`'s `path:`,\n"
        "and fine-tune with Ultralytics YOLOv8m using the upstream\n"
        "640m.onnx's `.pt` checkpoint as starting weights.\n"
    )

    # RUBRIC.md — the exact annotation rules for the three new
    # genital-state sub-classes. Kept in the output directory so the
    # reviewer has one source of truth beside the data.
    (output_path / "RUBRIC.md").write_text(RUBRIC_CONTENT)

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
