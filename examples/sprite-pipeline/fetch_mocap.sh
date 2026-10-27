#!/usr/bin/env bash
# Fetch a CMU motion-capture walk (subject 07, clip 01) as BVH.
# CMU Graphics Lab data is free for any use. Usage: ./fetch_mocap.sh [dest-dir] [subject_clip]
set -euo pipefail

dest="${1:-mocap}"
clip="${2:-07_01}"
subject="${clip%%_*}"
mkdir -p "$dest"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

git clone --quiet --depth 1 --filter=blob:none --no-checkout https://github.com/Shriinivas/cmubvh "$work/cmubvh"
n=$((10#$subject))
lo=$(( (n - 1) / 9 * 9 + 1 ))
hi=$(( lo + 8 ))
seq_dir=$(printf "Sequence-%03d-%03d" "$lo" "$hi")
path="$seq_dir/$subject/Data/$clip.zip"
git -C "$work/cmubvh" checkout --quiet HEAD -- "$path"
unzip -o -q "$work/cmubvh/$path" -d "$dest"
echo "wrote $dest/$clip.bvh"
