#!/bin/zsh
# refresh_haneda_metadata.sh — backup for the scheduled Tuesday refresh.
#
# Copies the 22-class classes.txt / data.yaml / RUBRIC.md from the
# completed Outdoor bootstrap dataset over the Haneda dataset and
# fixes the data.yaml path, so Haneda picks up the
# MALE_GENITALIA_COVERED (class 21) addition that landed after the
# Haneda Python run was already underway.
#
# Idempotent: safe to run multiple times; only the four files
# touched, the labels/ + images/ + crops/ directories untouched.
#
# Refuses to run while the Haneda Python process is still alive,
# since the in-flight script writes its own (stale) metadata at
# completion and would clobber whatever this script wrote.
#
# Usage:
#   Tools/refresh_haneda_metadata.sh
#
# Override paths (rare — only if datasets moved):
#   SRC=/path/to/outdoor DST=/path/to/haneda Tools/refresh_haneda_metadata.sh

set -euo pipefail

SRC="${SRC:-/Volumes/Narita/bootstrap-dataset}"
DST="${DST:-/Volumes/Narita/bootstrap-hires-dataset}"

# 1. Sanity: source must be the 22-class dataset; destination must
#    be a valid Haneda output dir.
[[ -f "$SRC/classes.txt" && -f "$SRC/data.yaml" && -f "$SRC/RUBRIC.md" ]] || {
    echo "error: source missing one of classes.txt / data.yaml / RUBRIC.md at $SRC" >&2
    exit 1
}
src_class_count=$(wc -l < "$SRC/classes.txt")
[[ "$src_class_count" -eq 22 ]] || {
    echo "error: $SRC/classes.txt has $src_class_count lines, expected 22 (the 22-class schema)" >&2
    exit 1
}
[[ -d "$DST/images" && -d "$DST/labels" ]] || {
    echo "error: $DST does not look like a bootstrap output dir (missing images/ or labels/)" >&2
    exit 1
}

# 2. Refuse to run while the Haneda Python is still active. Match
#    on the destination path inside the cmdline so we don't false-
#    positive on an unrelated python.
if pgrep -fl "bootstrap_nudenet_dataset.py" | grep -q "$DST"; then
    echo "error: Haneda bootstrap is still running. wait for completion before refreshing metadata." >&2
    pgrep -fl "bootstrap_nudenet_dataset.py" | grep "$DST" >&2
    exit 1
fi

# 3. Copy the three metadata files. cp -p preserves timestamps.
cp -p "$SRC/classes.txt" "$DST/classes.txt"
cp -p "$SRC/data.yaml"   "$DST/data.yaml"
cp -p "$SRC/RUBRIC.md"   "$DST/RUBRIC.md"

# 4. Rewrite data.yaml's `path:` to point at the Haneda dataset.
#    Use a temp-file + atomic rename so an interrupted run doesn't
#    leave a half-written yaml.
tmp="$DST/data.yaml.tmp"
awk -v dst="$DST" '
    /^path:/ { print "path: " dst; next }
    { print }
' "$DST/data.yaml" > "$tmp"
mv "$tmp" "$DST/data.yaml"

# 5. Verify and report.
nc_line=$(grep '^nc:' "$DST/data.yaml" || true)
last_class=$(tail -n1 "$DST/classes.txt")
label_count=$(ls "$DST/labels/" | wc -l | tr -d ' ')
size=$(du -sh "$DST" 2>/dev/null | awk '{print $1}')
echo "refreshed $DST"
echo "  $nc_line"
echo "  last class: $last_class"
echo "  labels: $label_count"
echo "  size: $size"
