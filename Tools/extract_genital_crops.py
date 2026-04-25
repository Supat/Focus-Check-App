#!/usr/bin/env python3
"""
extract_genital_crops.py — Extract NudeNet detections from a
macOS Photos Library into a flat folder of crop thumbnails,
ready for manual review and CreateML image-classifier training.

=======================================================================
SAFETY GUARANTEE (read this before running):

  This script is READ-ONLY on the Photos Library.

  Concretely:
    * `osxphotos` opens the Photos SQLite database in read-only mode.
      No path inside osxphotos modifies the library.
    * The only file-system writes are:
        - READS from the library (via osxphotos' export, which COPIES)
        - WRITES to the user-specified `--output` directory
        - per-photo temp files via `osxphotos.PhotoExporter` (copies,
          never mutates the library)
    * `--output` is validated against the library path. If they
      overlap in either direction the script aborts before any I/O.
    * No osxphotos method that could modify the library (`add`,
      `remove`, `update_metadata`) is imported or called.
    * NudeNet only ever sees the copied JPEGs in `--output`.
    * `--dry-run` does zero I/O — enumerates UUIDs and exits.

  No code from `bootstrap_nudenet_dataset.py` is imported here. This
  is a fresh standalone tool.

=======================================================================
PURPOSE

  Build a training corpus for a downstream Core ML image classifier
  that re-labels NudeNet's genital-region detections into a finer
  five-class schema:

    MALE_GENITALIA_COVERED
    MALE_GENITALIA_FLACCID
    MALE_GENITALIA_AROUSAL
    MALE_GENITALIA_ORGASM
    OTHER  (NudeNet false positive or non-genital private-region anatomy)

  The detector side stays as upstream-NudeNet; only this classifier
  is trained from the corpus. The classifier consumes the same
  crop NudeNet scored, runs after each detection in the relevant
  input classes, and overrides the label with one of the five
  output classes.

=======================================================================
INSTALL

    pip install osxphotos nudenet pillow pillow-heif tqdm \\
                opencv-python

    # Larger NudeNet variant (better recall than the bundled 320n):
    curl -L -O https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx

=======================================================================
USAGE

    # Dry-run on a small album — lists candidate UUIDs only
    python3 Tools/extract_genital_crops.py \\
        --library  ~/Pictures/Outdoor\\ Library.photoslibrary \\
        --output   ~/Desktop/genital-crops \\
        --album    "Training" \\
        --dry-run

    # Real run on full library, larger model
    python3 Tools/extract_genital_crops.py \\
        --library  ~/Pictures/Outdoor\\ Library.photoslibrary \\
        --output   ~/Desktop/genital-crops \\
        --no-album-filter \\
        --model-path /tmp/640m.onnx \\
        --confidence 0.05

  See `Tools/README-extract-genital-crops.md` for the post-extraction
  review workflow + CreateML training steps.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


# NudeNet classes we keep crops for. The mapping value is the short
# tag embedded in the output filename so the reviewer can see at a
# glance which NudeNet class produced each crop.
KEPT_NUDENET_CLASSES: dict[str, str] = {
    "MALE_GENITALIA_EXPOSED":   "MGE",
    "FEMALE_GENITALIA_COVERED": "FGC",
    "FEMALE_GENITALIA_EXPOSED": "FGE",
    "BUTTOCKS_EXPOSED":         "BTE",
    "BUTTOCKS_COVERED":         "BTC",
    "ANUS_EXPOSED":             "ANE",
}

# Five empty class folders seeded at output-dir creation. Reviewer
# moves crops from `unsorted/_*/` into these by hand. CreateML reads
# only these top-level folders when training.
REVIEW_CLASSES: list[str] = [
    "MALE_GENITALIA_COVERED",
    "MALE_GENITALIA_FLACCID",
    "MALE_GENITALIA_AROUSAL",
    "MALE_GENITALIA_ORGASM",
    "OTHER",
]

# Outward padding around each NudeNet box before crop. CreateML's
# input resize is to a fixed square; the surrounding skin / fabric
# context lets the classifier see edges (waistband, underwear seams)
# that disambiguate covered vs exposed cases.
CROP_PADDING_FRAC: float = 0.15

# Skip crops smaller than this on the short side. Below 32 px there
# isn't enough information for the classifier to learn from after
# CreateML's downsample to the model input size.
MIN_CROP_SHORT_SIDE: int = 32

# JPEG quality for output thumbnails. 88 is a defensible default —
# high enough to preserve fine anatomical detail, low enough to keep
# the corpus disk size manageable.
OUTPUT_JPEG_QUALITY: int = 88


def resolve_safely(path_str: str) -> Path:
    try:
        return Path(path_str).expanduser().resolve(strict=False)
    except Exception as exc:
        sys.exit(f"ERROR: could not resolve path {path_str!r}: {exc}")


def resolve_library_arg(raw: str) -> Path:
    """Validate the `--library` argument the same way the earlier
    bootstrap script did. Accepts a `.photoslibrary` bundle or the
    direct `database/Photos.sqlite` path inside one. Aborts with a
    specific error before any I/O when the path is wrong."""
    path = resolve_safely(raw)
    if not path.exists():
        sys.exit(f"ERROR: --library path does not exist: {path}")
    if path.is_dir():
        if path.suffix.lower() != ".photoslibrary":
            sys.exit(
                f"ERROR: --library directory {path} isn't a "
                "`.photoslibrary` bundle. Pass the bundle itself "
                "(the folder Finder shows as a single Photos Library "
                "with a badge), or the internal Photos.sqlite path."
            )
        return path
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


def assert_output_outside_library(output: Path, library: Path) -> None:
    """Refuse to write inside the Photos Library, and refuse a
    library path inside the output (the latter shouldn't happen but
    a wrong arg ordering would be quietly catastrophic)."""
    out = output.resolve(strict=False)
    lib = library.resolve(strict=False)
    try:
        out.relative_to(lib)
        sys.exit(
            f"ERROR: --output {out} is inside --library {lib}. "
            "Refusing to write into the Photos Library."
        )
    except ValueError:
        pass
    try:
        lib.relative_to(out)
        sys.exit(
            f"ERROR: --library {lib} is inside --output {out}. "
            "Path arguments look swapped."
        )
    except ValueError:
        pass


def seed_output_dirs(output: Path) -> tuple[Path, dict[str, Path]]:
    """Create `unsorted/_<TAG>/` per kept NudeNet class plus the
    five empty REVIEW_CLASSES folders. Returns (unsorted_root,
    {tag: subdir}) for the extraction loop."""
    output.mkdir(parents=True, exist_ok=True)
    unsorted_root = output / "unsorted"
    unsorted_root.mkdir(exist_ok=True)
    tag_dirs: dict[str, Path] = {}
    for tag in KEPT_NUDENET_CLASSES.values():
        d = unsorted_root / f"_{tag}"
        d.mkdir(exist_ok=True)
        tag_dirs[tag] = d
    for cls in REVIEW_CLASSES:
        (output / cls).mkdir(exist_ok=True)
    return unsorted_root, tag_dirs


def pad_to_square(crop):
    """Center-pad a crop to a square with black borders so CreateML's
    resize-to-input-square doesn't distort the aspect ratio."""
    import cv2
    h, w = crop.shape[:2]
    if h == w:
        return crop
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(
        crop, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def unique_path(directory: Path, stem: str, ext: str) -> Path:
    """Filename collisions are extremely rare here (source stem +
    detection index + class tag + score percent), but if two runs
    against different libraries happen into the same output and a
    photo from each has the same stem and same NudeNet output for
    that detection index, append a numeric suffix so we don't
    silently overwrite."""
    candidate = directory / f"{stem}{ext}"
    if not candidate.exists():
        return candidate
    n = 1
    while True:
        candidate = directory / f"{stem}__{n}{ext}"
        if not candidate.exists():
            return candidate
        n += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract NudeNet genital-region crops for "
                    "CreateML classifier training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--library", required=True,
        help="Path to a .photoslibrary bundle (or its internal "
             "Photos.sqlite). Read-only.",
    )
    p.add_argument(
        "--output", required=True,
        help="Directory to write crops into. Will be created if "
             "missing. Aborts if it overlaps the library path.",
    )
    scope = p.add_mutually_exclusive_group()
    scope.add_argument(
        "--album", default=None,
        help="Restrict to photos in the named album (and its "
             "sub-albums).",
    )
    scope.add_argument(
        "--no-album-filter", action="store_true",
        help="Walk the entire library. Required when --album is not "
             "specified, to avoid accidentally running over a 100k-"
             "photo library by reflex.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap candidate count (for smoke-tests).",
    )
    p.add_argument(
        "--confidence", type=float, default=0.05,
        help="NudeNet score floor for keeping a detection. Defaults "
             "to 0.05 — permissive; the reviewer is the filter.",
    )
    p.add_argument(
        "--model-path", default=None,
        help="Path to a NudeNet ONNX (e.g. 640m.onnx). Defaults to "
             "the `nudenet` package's bundled 320n model.",
    )
    p.add_argument(
        "--inference-resolution", type=int, default=640,
        help="NudeNet inference resolution. 640 matches the 640m "
             "weights; bundled 320n uses 320.",
    )
    p.add_argument(
        "--keep-originals", action="store_true",
        help="Skip the HEIC/RAW → JPEG conversion. NudeNet still "
             "needs a JPEG for cv2.imread, so this only makes sense "
             "if the source library is JPEG-only.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Enumerate candidate UUIDs and exit without I/O.",
    )
    args = p.parse_args()
    if args.album is None and not args.no_album_filter:
        p.error(
            "Pass --album NAME to restrict to one album, or "
            "--no-album-filter to walk the entire library. Defaulting "
            "to whole-library by accident is too risky."
        )
    return args


def main() -> None:
    args = parse_args()

    library = resolve_library_arg(args.library)
    output = resolve_safely(args.output)
    assert_output_outside_library(output, library)

    print(f"Photos Library (read-only): {library}")
    print(f"Output directory:           {output}")
    if args.album:
        print(f"Album filter:               {args.album!r}")
    else:
        print("Album filter:               (none — full library)")
    print(f"Confidence floor:           {args.confidence}")
    print()

    # Lazy-import osxphotos so --help is fast and a missing dep gets
    # a specific message instead of a traceback at import time.
    try:
        import osxphotos
    except ImportError:
        sys.exit(
            "ERROR: osxphotos not installed. "
            "Run `pip install osxphotos`."
        )

    try:
        db = osxphotos.PhotosDB(dbfile=str(library))
    except Exception as exc:
        sys.exit(f"ERROR: could not open Photos Library: {exc}")

    # Album scoping. osxphotos returns photos across all albums by
    # default; we filter ourselves so the album-name match is
    # case-insensitive and explicit.
    if args.album:
        wanted = args.album.lower()
        photos = [
            p for p in db.photos(intrash=False)
            if any(a.title.lower() == wanted for a in p.album_info)
        ]
    else:
        photos = list(db.photos(intrash=False))
    # Deterministic order so smoke-test reruns produce stable output.
    photos.sort(key=lambda p: p.uuid)
    if args.limit is not None:
        photos = photos[: args.limit]
    print(f"Candidates: {len(photos)} photo(s)")

    if args.dry_run:
        for i, p in enumerate(photos[:20]):
            print(f"  {p.uuid}  {p.original_filename}")
        if len(photos) > 20:
            print(f"  ... and {len(photos) - 20} more")
        return

    if not photos:
        print("Nothing to do.")
        return

    # Seed output dirs only after the dry-run check so a dry-run on a
    # nonexistent path doesn't materialise it.
    unsorted_root, tag_dirs = seed_output_dirs(output)
    print(f"Seeded {unsorted_root} + {len(REVIEW_CLASSES)} class folders")

    # Lazy-import NudeNet + cv2 / PIL so a missing dep aborts after
    # the cheap path is done (album scoping, dry-run).
    try:
        from nudenet import NudeDetector
    except ImportError:
        sys.exit(
            "ERROR: nudenet not installed. "
            "Run `pip install nudenet`."
        )
    try:
        import cv2
    except ImportError:
        sys.exit(
            "ERROR: opencv-python not installed. "
            "Run `pip install opencv-python`."
        )
    try:
        import pillow_heif  # noqa: F401  registers HEIC opener
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **_): return it  # noqa: E731

    # Build the detector. NudeNet expects an ONNX path; if --model-
    # path isn't given, fall back to the package-bundled 320n.
    detector_kwargs: dict = {}
    if args.model_path:
        model_path = resolve_safely(args.model_path)
        if not model_path.is_file():
            sys.exit(
                f"ERROR: --model-path {model_path} doesn't exist."
            )
        detector_kwargs["model_path"] = str(model_path)
    if args.inference_resolution:
        detector_kwargs["inference_resolution"] = args.inference_resolution
    try:
        detector = NudeDetector(**detector_kwargs)
    except Exception as exc:
        sys.exit(f"ERROR: NudeNet init failed: {exc}")

    convert = not args.keep_originals
    export_options = osxphotos.ExportOptions(
        convert_to_jpeg=convert,
        overwrite=False,
        update=False,
    )

    exported_count = 0
    decoded_count = 0
    crop_count = 0
    skipped_export = 0
    skipped_decode = 0

    # One temp directory shared across the run. Each photo's export
    # writes into here; we delete the file as soon as NudeNet is done
    # with it so the temp dir doesn't grow unbounded.
    with tempfile.TemporaryDirectory(prefix="genital-crops-") as tmp:
        tmp_path = Path(tmp)

        for photo in tqdm(photos, desc="Processing"):
            # Export — copies the original (converting if requested).
            try:
                exporter = osxphotos.PhotoExporter(photo)
                results = exporter.export(
                    str(tmp_path), options=export_options
                )
                paths = results.exported or results.skipped
            except Exception as exc:
                print(
                    f"  export failed for {photo.uuid}: {exc}",
                    file=sys.stderr,
                )
                skipped_export += 1
                continue
            if not paths:
                skipped_export += 1
                continue
            exported = Path(paths[0])
            exported_count += 1

            # Decode. Movies and a few RAW edge cases come back as
            # files cv2 can't read — skip and move on.
            img_bgr = cv2.imread(str(exported))
            if img_bgr is None:
                skipped_decode += 1
                _safe_unlink(exported)
                continue
            decoded_count += 1
            img_h, img_w = img_bgr.shape[:2]

            # NudeNet needs a 4-channel ndarray (its _read_image does
            # an unconditional cvtColor RGBA → BGR), so stack alpha.
            try:
                rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
                detections = detector.detect(rgba) or []
            except Exception as exc:
                print(
                    f"  NudeNet failed on {exported.name}: {exc}",
                    file=sys.stderr,
                )
                _safe_unlink(exported)
                continue

            stem = exported.stem
            for det_idx, det in enumerate(detections):
                cls = det.get("class", "")
                if cls not in KEPT_NUDENET_CLASSES:
                    continue
                score = float(det.get("score", 0))
                if score < args.confidence:
                    continue
                box = det.get("box", [])
                if len(box) != 4:
                    continue
                x_tl, y_tl, w_px, h_px = (float(v) for v in box)
                if w_px <= 0 or h_px <= 0:
                    continue

                # Outward padding by max-side fraction so the
                # padding amount is symmetric around portrait /
                # landscape boxes.
                pad = max(w_px, h_px) * CROP_PADDING_FRAC
                x1 = max(0, int(round(x_tl - pad)))
                y1 = max(0, int(round(y_tl - pad)))
                x2 = min(img_w, int(round(x_tl + w_px + pad)))
                y2 = min(img_h, int(round(y_tl + h_px + pad)))
                if min(x2 - x1, y2 - y1) < MIN_CROP_SHORT_SIDE:
                    continue

                crop = img_bgr[y1:y2, x1:x2]
                squared = pad_to_square(crop)
                tag = KEPT_NUDENET_CLASSES[cls]
                conf_pct = max(0, min(99, int(round(score * 100))))
                fname_stem = f"{stem}_det{det_idx:02d}_{tag}_{conf_pct:02d}"
                out_path = unique_path(tag_dirs[tag], fname_stem, ".jpg")
                cv2.imwrite(
                    str(out_path), squared,
                    [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_JPEG_QUALITY],
                )
                crop_count += 1

            _safe_unlink(exported)

    print()
    print(f"  exported:        {exported_count}")
    print(f"  decoded:         {decoded_count}")
    print(f"  crops written:   {crop_count}")
    print(f"  skipped export:  {skipped_export}")
    print(f"  skipped decode:  {skipped_decode}")
    print()
    print("Next steps:")
    print(f"  1. Open {unsorted_root} in Finder.")
    print(f"  2. Move thumbnails into the five class folders under")
    print(f"     {output}/ — see README-extract-genital-crops.md.")
    print(f"  3. Open CreateML.app → Image Classification → drop")
    print(f"     {output}/ onto the Training Data well.")


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"  warning: could not delete temp file {path}: {exc}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
