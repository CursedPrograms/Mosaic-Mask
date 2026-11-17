#!/usr/bin/env bash
# Builds mosaic-cli on Linux (cmake, a C++17 compiler, OpenCV 4 and ONNX Runtime). The window needs Windows for now
# (the virtual camera is OBS's Windows one).
set -e
cd "$(dirname "$0")"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMOSAIC_GUI=OFF
cmake --build build -j
echo "Built: build/mosaic-cli"
