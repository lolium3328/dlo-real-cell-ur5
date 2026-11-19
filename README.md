# DLO real cell — dual UR5 (CB3) deployment

Self-contained deployment package: v39 r60 policy, v41 UR5 scene, demo scene `test_reach_0087`.
Read `部署说明.md` first (section "v41 更新").

    bash install.sh
    .venv/bin/python scripts/real_cell/run_real_cell.py --preflight-only --left-ip <left> --right-ip <right>

`MANIFEST.json` holds the sha256 of every packaged file; `.gitattributes` keeps line endings untouched so the
check passes on Windows checkouts too. Built from lolium3328/dlo-value-wm @ 784c7f0.
