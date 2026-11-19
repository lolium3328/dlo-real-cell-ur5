"""Standalone source/weights/physical-assets bundle with explicit acceptance evidence."""
import json
from pathlib import Path
import shutil
import sys
import subprocess
import io
import tarfile
import tempfile
import torch
import mujoco
import numpy as np
import cv2
import scipy
from .config import ROOT, digest


def package(config, checkpoints):
    reports = [json.loads((p.parent/'eval_test/summary.json').read_text()) for p in checkpoints]
    checks = [json.loads((p.parent/'deployment_check.json').read_text()) for p in checkpoints]
    training = [json.loads((p.parent/'completed.json').read_text()) for p in checkpoints]
    accepted = all(r['performance_acceptance'] for r in reports) and all(r['accepted'] for r in checks)
    accepted &= all(r['rounds'] == config.rounds and not r['smoke'] for r in training)
    result = dict(protocol='calibrated_cell_deploy_v1', accepted=bool(accepted), hardware_executed=False,
                  checkpoints=[str(p) for p in checkpoints], training=training, evaluations=reports,
                  deployment_checks=[{k:v for k,v in c.items() if k != 'rows'} for c in checks], config=config.provenance())
    (config.path/'acceptance.json').write_text(json.dumps(result, indent=2))
    if not accepted:
        return result
    out = config.path/'deploy_bundle'
    out.mkdir(exist_ok=True)
    runtime = checks[0]['config']
    for check in checks[1:]:
        if check['config']['source_sha256'] != runtime['source_sha256']:
            raise ValueError('Deployment checks used different runtime sources; rerun both on the final runtime')
    runtime_commit = runtime['code_commit']
    committed = subprocess.check_output(['git', 'archive', '--format=tar', runtime_commit, 'dlovw'], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(committed)) as archive:
        archive.extractall(out, filter='data')
    # Preserve the relative asset names referenced by the frozen profile and weights.
    assets = list(config.provenance()['inputs'])
    for name in assets:
        dest = out/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    shutil.copy2(checkpoints[0], out/'policy.pt')
    for name, expected in runtime['source_sha256'].items():
        if digest(out/name) != expected:
            raise ValueError(f'Bundle source differs from evaluated runtime: {name}')
    shutil.copy2(config.path/'acceptance.json', out/'acceptance.json')
    (out/'run_controller.py').write_text('from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent))\nfrom dlovw.cell.calibrated.runtime_cli import main\nmain()\n')
    (out/'runtime_versions.json').write_text(json.dumps(dict(python=sys.version, torch=torch.__version__,
         mujoco=mujoco.__version__, numpy=np.__version__, scipy=scipy.__version__, opencv=cv2.__version__), indent=2))
    (out/'README.md').write_text('# Calibrated cell deployment candidate\n\n'
        'Run with the recorded uv Python environment. The adapter writes bounded left-end delta XYZ commands; it does not connect to a robot.\n\n'
        '`uv run --no-project --python /path/to/python python run_controller.py --policy policy.pt --goal goal.json --input observations.jsonl`\n\n'
        'Each input row contains t, left, right_tcp, right_rotation (3x3), images (front/top paths), image_times, and optional robot timestamps/frame. '
        'Goal JSON contains centerline (39x3) and target_left (3). Units: metres, seconds; world frame: left robot base. '
        'Ball radius 40 mm; mount offset is -25 mm along tool z towards flange. Tool orientation is fixed in this training round. '
        'See acceptance.json for conditional oracle screening coverage, independent test counts, faults, runtime and limitations. '
        'No physical robot trial is claimed.\n')
    manifest = {str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    # Import from outside the repository; deployment must not need a Git checkout.
    subprocess.run([sys.executable, '-I', str(out/'run_controller.py'), '--help'], cwd='/tmp', check=True,
                   stdout=subprocess.DEVNULL)
    with tempfile.TemporaryDirectory(prefix='dlo_bundle_check_') as temporary:
        copied = Path(temporary)/'bundle'
        shutil.copytree(out, copied)
        checker = copied/'check_assets.py'
        checker.write_text('from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent))\n'
            'from dlovw.cell.calibrated.config import Config\nfrom dlovw.cell.calibrated.checkpoint import load\n'
            'nets,meta=load(Path(__file__).parent/"policy.pt")\nc=Config()\n'
            'assert c.provenance()["inputs"]==meta["config"]["inputs"]\n')
        subprocess.run([sys.executable, '-I', str(checker)], cwd=temporary, check=True)
    return result
