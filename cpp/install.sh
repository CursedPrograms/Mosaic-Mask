#!/usr/bin/env bash
# Mosaic Mask's window is Windows-only for now; on Linux, build.sh makes mosaic-cli. This copies it to ~/.local/bin.
set -e
cd "$(dirname "$0")"
[ -f build/mosaic-cli ] || { echo "Build it first: ./build.sh"; exit 1; }
mkdir -p ~/.local/bin && cp build/mosaic-cli ~/.local/bin/ && echo "Installed: ~/.local/bin/mosaic-cli"
