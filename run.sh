#!/usr/bin/env bash
# run.sh - Mosaic Mask. Sets up ./venv the first time (and again when requirements.txt changes), fetches the
# models once, then starts app.py. Options pass through: ./run.sh --camera 1 ...
set -e
cd "$(dirname "$0")"
if [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe; else PY=venv/bin/python; fi
if [ ! -x "$PY" ]; then
    echo "Creating venv ..."
    if command -v py >/dev/null 2>&1; then py -3.11 -m venv venv; else python3.11 -m venv venv; fi
    if [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe; else PY=venv/bin/python; fi
fi
if ! cmp -s requirements.txt venv/requirements.installed; then
    echo "Installing requirements ..."
    "$PY" -m pip install -r requirements.txt
    cp requirements.txt venv/requirements.installed
fi
fetch() {  # file url what
    if [ ! -f "models/$1" ]; then
        echo "Downloading $3, once ..."
        mkdir -p models
        curl -L --fail -o "models/$1.part" "$2"
        mv "models/$1.part" "models/$1"
    fi
}
fetch xseg_1.onnx "https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx" "the occlusion model, about 70 MB"
exec "$PY" app.py "$@"
