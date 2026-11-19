# DLO real cell — dual UR5 (CB3) deployment

Self-contained deployment package: v39 r60 policy, v41 UR5 scene, demo scene `test_reach_0087`.
Read `部署说明.md` first (sections "v42 更新" and "v41 更新").

v42: the scene runs 6x slower than simulated (`--time-scale 6`, default) so the cell PC's ~0.46 s frame-to-action
latency fits the trained decision cycle; each run records front / top videos (`--no-video` to skip). Ball other than the trained 40 mm: `--sphere-radius`, `--sphere-centre-behind-tip`. Includes the
cell-PC commits (Windows preflight, timing logs, CB3 identification).

    bash install.sh
    .venv/bin/python scripts/real_cell/run_real_cell.py --preflight-only --left-ip <left> --right-ip <right>
    .venv/bin/python scripts/real_cell/run_real_cell.py --check-only --left-ip <left> --right-ip <right>
    .venv/bin/python scripts/real_cell/run_real_cell.py --left-ip <left> --right-ip <right> --torch-threads 3

`MANIFEST.json` holds the sha256 of every packaged file; `.gitattributes` keeps line endings untouched so the
check passes on Windows checkouts too. Built from lolium3328/dlo-value-wm @ 0fe93c0.
