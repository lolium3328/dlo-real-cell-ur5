"""Read-only access to the inherited ImaginationDLO code (pinned at bf1a93b).

Import this module before any `control.*`, `envs.*` or `wm.*` import. Run Python with
`-B` (or PYTHONDONTWRITEBYTECODE=1) so no __pycache__ is written into the backup repo.
"""
from __future__ import annotations

import os
import sys

LEGACY_ROOT = os.path.expanduser(os.environ.get("DLOVW_LEGACY_ROOT", "~/dlo/ImaginationDLO"))
LEGACY_REVISION = "bf1a93b7de00d18cabe4afdd343d9882ed27cd16"
sys.dont_write_bytecode = True
if LEGACY_ROOT not in sys.path:
    sys.path.insert(0, LEGACY_ROOT)

# Scene XMLs that contain exactly N obstacle spheres, so no unused sphere is left in the
# workspace (the legacy `obstacle_positions_m` parking writes geom_pos, which MuJoCo
# ignores for these mocap geoms; see docs/notes/scene_audit_and_step0_20261020.md).
XML_BY_OBSTACLES = {
    1: "envs/xml/bimanual_dlo_ur5e_sim2real_532mm_yellow_dyn_sphere_deform.xml",
    2: "envs/xml/bimanual_dlo_ur5e_sim2real_532mm_yellow_dyn_two_spheres_deform.xml",
    3: "envs/xml/bimanual_dlo_ur5e_sim2real_532mm_yellow_dyn_three_spheres_deform.xml",
}

P2_TEMPLATE_MANIFEST = os.path.join(
    LEGACY_ROOT, "outputs/sim_532mm/task_value_dataset/p2_manifest_20261018_v2_a02/scenario_manifest.json")
