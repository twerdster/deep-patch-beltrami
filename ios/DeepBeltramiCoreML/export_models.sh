#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${DPB_COREML_PYTHON:-python3.11}"
if [[ -z "${DPB_COREML_PYTHON:-}" && -x /opt/homebrew/bin/python3.11 ]]; then
  PYTHON_BIN=/opt/homebrew/bin/python3.11
fi
VENV_DIR="${DPB_COREML_VENV:-$HOME/Library/Caches/deep-patch-beltrami-coreml/venv}"
if [[ "$(uname -s)" != Darwin ]]; then
  echo "Core ML model compilation requires macOS." >&2
  exit 1
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
if ! "$VENV_DIR/bin/python" -c 'import torch,coremltools,numpy,h5py,PIL; assert torch.__version__.startswith("2.5.1") and coremltools.__version__=="8.3.0" and numpy.__version__=="1.26.4"' >/dev/null 2>&1; then
  "$VENV_DIR/bin/python" -m pip install 'torch==2.5.1' 'coremltools==8.3.0' 'numpy==1.26.4' h5py Pillow
fi
"$VENV_DIR/bin/python" -m pip install --no-deps -e "$REPO_DIR"
exec "$VENV_DIR/bin/python" "$REPO_DIR/tools/export_coreml.py" "$@"
