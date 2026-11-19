#!/usr/bin/env bash
# Creates .venv in the package folder and installs the pinned requirements (needs internet once, or a wheel cache).
# Usage:  bash install.sh        (from the package root; copy the package to a local disk first, not NTFS)
set -euo pipefail
cd "$(dirname "$0")"
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
      "torch==$(grep '^torch==' requirements.txt | cut -d= -f3)"
  uv pip install --python .venv/bin/python -r <(grep -v '^torch==' requirements.txt)
else
  python3.11 -m venv .venv
  .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
      "torch==$(grep '^torch==' requirements.txt | cut -d= -f3)"
  .venv/bin/pip install -r <(grep -v '^torch==' requirements.txt)
fi
.venv/bin/python -c "import numpy, torch, mujoco, cv2, rtde_control, pyrealsense2; print('environment OK')"
