"""Build a self-contained deployment package of the real-cell runner (code + checkpoint + scene + calibration inputs).

    python scripts/real_cell/make_package.py --dest /media/<user>/<disk>/dlo_real_cell_YYYYMMDD

Layout keeps the repository-relative paths (the Controller checks the physical input files by path and sha256):
  dlovw/                       library (deployment uses dlovw.cell.*, dlovw.real_rope.*, dlovw.imagination.*)
  scripts/real_cell/           runner, controller, checks, tools
  outputs/...                  checkpoint, v40 demo scenes (meshes copied into assets/, paths made relative),
                               profile / workspace / camera extrinsics used by the checkpoint
  selftest/frames.npz          two rendered camera frames for the planning-latency self-test (no GL needed there)
  requirements.txt, install.sh, 部署说明.md, MANIFEST.json (sha256 of every file)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCENES = 'outputs/cell_calibrated_20261113/v41_ur5'
POLICY = 'outputs/cell_calibrated_20261113/v39_occlusion'
CKPT = f'{POLICY}/policy/seed_0/ckpt_r60.pt'
INPUTS = ['outputs/real_rope_refine_20261113/profile.json',
          'outputs/visibility_reference_20261113/recommended_workspace.json',
          'outputs/tcp_camera_recalibration_20261113/candidate_extrinsics.json']
DATA = [CKPT, f'{POLICY}/config.json', f'{SCENES}/config.json', f'{SCENES}/scene.json',
        f'{SCENES}/tasks/test/scenes.json'] + INPUTS
REQS = ['numpy', 'scipy', 'torch', 'mujoco', 'opencv-python', 'ur_rtde', 'pyrealsense2', 'glfw', 'PyOpenGL']
SELFTEST_SCENE = 'test_reach_0087'


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_scene_xml(dest):
    """scene.xml with every mesh / texture copied to assets/ and referenced relative to the xml."""
    src = ROOT/SCENES/'scene.xml'
    xml = src.read_text()
    meshdir = Path(re.search(r'meshdir="([^"]*)"', xml).group(1))
    out_dir = dest/SCENES
    assets = out_dir/'assets'
    files = sorted(set(re.findall(r'file="([^"]+)"', xml)))
    for f in files:
        absf = (meshdir/f).resolve()
        rel = Path(os.path.relpath(absf, meshdir.parent.parent))      # e.g. third_party/mujoco_menagerie/...
        copy(absf, assets/rel)
        xml = xml.replace(f'file="{f}"', f'file="{rel.as_posix()}"')
    xml = re.sub(r'meshdir="[^"]*"', 'meshdir="assets/"', xml)
    xml = re.sub(r'texturedir="[^"]*"', 'texturedir="assets/"', xml)
    (out_dir/'scene.xml').write_text(xml)
    return len(files)


def render_selftest(dest):
    import numpy as np
    from dlovw.cell.calibrated.config import Config
    from dlovw.cell.calibrated.physics import RopeState
    from dlovw.cell.calibrated.render import SceneView
    c = Config(out=str(ROOT/SCENES))
    rec = {r['id']: r for r in json.loads((c.path/'tasks/test/scenes.json').read_text())['scenes']}[SELFTEST_SCENE]
    state = RopeState(rec['start_left'], c, 0, False, qpos=rec.get('start_qpos'))
    view = SceneView(c)
    try:
        view.sync(state, np.asarray(rec['motion']['center'], float))
        im = view.images()
    finally:
        view.close()
    (dest/'selftest').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest/'selftest/frames.npz', front=im['front'], top=im['top'], scene=SELFTEST_SCENE)


def write_requirements(dest):
    import importlib.metadata as md
    lines = [f'# Python {sys.version.split()[0]}; tested {time.strftime("%Y-%m-%d")} on this build machine',
             '# torch: the CPU build is enough (the controller runs on CPU); install.sh uses the CPU wheel index']
    for d in REQS:
        lines.append(f'{d}=={md.version(d).split("+")[0]}')
    lines.append('threadpoolctl')
    (dest/'requirements.txt').write_text('\n'.join(lines)+'\n')


INSTALL = r'''#!/usr/bin/env bash
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
'''


def manifest(dest):
    sys.path.insert(0, str(ROOT/'scripts/real_cell'))
    from preflight import sha256
    files = {p.relative_to(dest).as_posix(): sha256(p) for p in sorted(dest.rglob('*'))
             if p.is_file() and p.name != 'MANIFEST.json'}
    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    (dest/'MANIFEST.json').write_text(json.dumps(dict(built=time.strftime('%Y-%m-%d %H:%M:%S'), source=str(ROOT),
                                                      git_head=commit, files=files), indent=1, ensure_ascii=False))
    return len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dest', required=True)
    ap.add_argument('--force', action='store_true', help='replace an existing destination folder')
    a = ap.parse_args()
    dest = Path(a.dest)
    if dest.exists():
        if not a.force:
            sys.exit(f'{dest} exists (use --force to replace it)')
        shutil.rmtree(dest)
    for p in sorted((ROOT/'dlovw').rglob('*.py')):
        copy(p, dest/p.relative_to(ROOT))
    for p in sorted((ROOT/'scripts/real_cell').glob('*')):
        if p.is_file() and p.suffix in ('.py', '.md'):
            copy(p, dest/p.relative_to(ROOT))
    for p in sorted((ROOT/'docs/deployment').glob('*.md')):        # field notes from the cell PC
        copy(p, dest/p.relative_to(ROOT))
    for d in DATA:
        copy(ROOT/d, dest/d)
    for p in sorted((ROOT/SCENES/'trajectories').glob('*')):       # export_sim_trajectory.py outputs
        copy(p, dest/p.relative_to(ROOT))
    n_assets = copy_scene_xml(dest)
    render_selftest(dest)
    write_requirements(dest)
    (dest/'install.sh').write_text(INSTALL)
    doc = ROOT/'scripts/real_cell/部署说明.md'
    if doc.exists():
        copy(doc, dest/'部署说明.md')
    n = manifest(dest)
    print(f'package: {dest}  ({n} files, {n_assets} mesh / texture assets)')


if __name__ == '__main__':
    main()
