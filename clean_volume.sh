#!/usr/bin/env bash
set -euo pipefail
VOL="${1:-/workspace}"; AUTO="${AUTO:-0}"
if ! findmnt -n -o TARGET "$VOL" | grep -qx "$VOL"; then echo "✗ not a mount"; exit 1; fi
if [ "$AUTO" != "1" ]; then read -p "Delete everything in $VOL except .ssh and lost+found? [yes/NO]: " ans; [ "$ans" = "yes" ] || exit 0; fi
find "$VOL" -mindepth 1 -maxdepth 1 ! -name '.ssh' ! -name 'lost+found' -exec rm -rf {} +
ls -la "$VOL"
