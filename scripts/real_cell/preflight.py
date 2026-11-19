"""Preflight safety checks, run before anything moves (and on their own with --preflight-only).

Each check -> PASS / WARN / FAIL with a message. Any FAIL blocks motion (error E700 with the failed items).

  package    every file of the deployment package matches MANIFEST.json (sha256)
  software   Python packages importable, versions vs the package's requirements
  blas       numpy BLAS single-threaded (multithreaded BLAS made tracking ~12x slower and starved the 125 Hz loops)
  load       CPU load (planning latency depends on it)
  selftest   v39 Controller on stored camera frames: decision latency p95 (sim: fine up to ~80 ms asynchronous)
  robots     RTDE connected, RUNNING, safety NORMAL, speed slider 100 %, ur_rtde control script running,
             TCP offset = probe tip, payload shown for the operator
  cameras    both RealSense serials streaming >= 25 Hz, images neither black nor saturated
  disk       free space for logs

touch_check: both probe tips touched together by hand (freedrive) -> distance in the world frame, which checks the
base-to-base transform layout.T_WORLD_FROM_RIGHT_BASE that every clearance and ball position depends on.
"""
import hashlib
import importlib
import importlib.metadata as md
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from robots import PROBE_TCP, T_RIGHT

ROOT = Path(__file__).resolve().parents[2]
PKGS = {'numpy': 'numpy', 'scipy': 'scipy', 'torch': 'torch', 'mujoco': 'mujoco', 'cv2': 'opencv-python',
        'rtde_control': 'ur_rtde', 'pyrealsense2': 'pyrealsense2'}
LAT_FAIL_MS, LAT_WARN_MS = 80., 50.


def _r(name, level, msg):
    return dict(name=name, level=level, msg=msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_package():
    man = ROOT/'MANIFEST.json'
    if not man.exists():
        return [_r('package', 'WARN', '未找到 MANIFEST.json（不是从部署包运行），跳过文件校验')]
    files = json.loads(man.read_text())['files']
    bad = [p for p, h in files.items() if not (ROOT/p).exists() or sha256(ROOT/p) != h]
    if bad:
        return [_r('package', 'FAIL', f'{len(bad)}/{len(files)} 个文件缺失或损坏，例如 {bad[:3]}；重新拷贝部署包')]
    return [_r('package', 'PASS', f'{len(files)} 个文件校验通过')]


def check_software():
    out, req = [], {}
    rf = ROOT/'requirements.txt'
    if rf.exists():
        for line in rf.read_text().splitlines():
            if '==' in line and not line.startswith('#'):
                k, v = line.split('==')
                req[k.strip().lower()] = v.strip()
    for mod, dist in PKGS.items():
        try:
            importlib.import_module(mod)
        except Exception as e:                       # noqa: BLE001
            out.append(_r(f'software:{mod}', 'FAIL', f'无法导入 {mod} ({e})；按部署说明安装环境'))
            continue
        try:
            v = md.version(dist)
        except md.PackageNotFoundError:
            v = '?'
        want = req.get(dist.lower())
        if want and v.split('+')[0] != want.split('+')[0]:
            out.append(_r(f'software:{mod}', 'WARN', f'{dist} {v}，部署包测试用的是 {want}'))
        else:
            out.append(_r(f'software:{mod}', 'PASS', f'{dist} {v}'))
    return out


def check_blas():
    try:
        from threadpoolctl import threadpool_info
        n = max([i.get('num_threads', 1) for i in threadpool_info() if i.get('user_api') == 'blas'] or [1])
    except ImportError:
        n = int(os.environ.get('OPENBLAS_NUM_THREADS', '0') or 0) or None
    if n == 1:
        return [_r('blas', 'PASS', 'BLAS 单线程')]
    return [_r('blas', 'FAIL', f'BLAS 线程数 {n}；必须为 1（run_real_cell.py 开头会设置，确认没有被外部环境覆盖）')]


def check_load():
    la, n = os.getloadavg()[0], os.cpu_count() or 1
    if la/n > .5:
        return [_r('load', 'WARN', f'CPU 负载 {la:.1f}/{n} 核；关闭其他任务，否则规划延迟变大')]
    return [_r('load', 'PASS', f'CPU 负载 {la:.1f}/{n} 核')]


def check_selftest(cfg, recs, checkpoint, commit, decisions=30):
    """Decision latency of the deployed Controller on stored frames (no robot, no camera)."""
    from dlovw.cell.calibrated.config import right_rotation, tcp_from_sphere
    from dlovw.cell.calibrated.deployment import Controller
    f = ROOT/'selftest/frames.npz'
    if not f.exists():
        return [_r('selftest', 'WARN', '未找到 selftest/frames.npz，跳过规划延迟自检')]
    d = np.load(f, allow_pickle=False)
    rec = recs[str(d['scene'])]
    images = {'front': d['front'], 'top': d['top']}
    ctl = Controller(str(checkpoint), rec['goal_curve'], rec['target_left'], commit=commit)
    R = right_rotation(cfg)
    tcp = tcp_from_sphere(np.asarray(rec['motion']['center'], float), R, cfg)
    start, dec, t, frame_t = np.asarray(rec['start_left'], float), [], 0., 0.
    t_wall = time.monotonic()
    while len(dec) < decisions and time.monotonic()-t_wall < 60:
        if int(round(t/cfg.dt)) % 4 == 0:
            frame_t = t
        r = ctl.step(t, start, tcp, R, images, {k: frame_t for k in images}, right_timestamp=t)
        if r.get('decision'):
            dec.append(r['elapsed_ms'])
        t += cfg.dt
    if len(dec) < decisions:
        return [_r('selftest', 'FAIL', f'60 s 内只完成 {len(dec)} 次决策（状态 {r.get("status")}）')]
    p50, p95 = np.percentile(dec, 50), np.percentile(dec, 95)
    level = 'FAIL' if p95 > LAT_FAIL_MS else 'WARN' if p95 > LAT_WARN_MS else 'PASS'
    return [_r('selftest', level, f'决策延迟 p50 {p50:.0f} ms、p95 {p95:.0f} ms（FAIL > {LAT_FAIL_MS:.0f} ms）')]


def check_robot(arm, name):
    out = []
    try:
        st = arm.status()
    except Exception as e:                           # noqa: BLE001
        return [_r(f'robot:{name}', 'FAIL', f'读取状态失败 ({e})')]
    from errors import ROBOT_MODE, SAFETY_MODE
    rm, sm = st['robot_mode'], st['safety_mode']
    out.append(_r(f'robot:{name}:mode', 'PASS' if rm == 7 else 'FAIL', f'机器人模式 {ROBOT_MODE.get(rm, rm)}（需要 RUNNING：上电并释放抱闸）'))
    out.append(_r(f'robot:{name}:safety', 'PASS' if sm == 1 else 'FAIL', f'安全模式 {SAFETY_MODE.get(sm, sm)}（需要 NORMAL）'))
    out.append(_r(f'robot:{name}:speed', 'PASS' if st['speed_scaling'] >= .99 else 'FAIL',
                  f'速度滑块 {st["speed_scaling"]*100:.0f}%（需要 100%，否则伺服跟不上）'))
    out.append(_r(f'robot:{name}:script', 'PASS' if st['program_running'] and st['control_connected'] else 'FAIL',
                  'ur_rtde 控制脚本' + ('运行中' if st['program_running'] else '未运行（确认远程控制模式，重新运行程序）')))
    off = arm.tcp_offset()
    ok = np.abs(off[:3]-np.array(PROBE_TCP[:3])).max() <= .003 and np.abs(off[3:]).max() < 1e-3
    out.append(_r(f'robot:{name}:tcp', 'PASS' if ok else 'FAIL', f'TCP 偏置 {np.round(off, 4).tolist()}（需要 {PROBE_TCP}）'))
    out.append(_r(f'robot:{name}:payload', 'WARN' if st['payload_kg'] <= 0 else 'PASS',
                  f'负载 {st["payload_kg"]} kg（确认与实际工具一致；0 表示可能未设置）'))
    return out


def check_cameras(cams, seconds=2.):
    if cams is None:
        return []
    out, stamps = [], {}
    t_end = time.monotonic()+seconds
    while time.monotonic() < t_end:
        for r, (_, ft) in cams.latest().items():
            stamps.setdefault(r, set()).add(ft)
        time.sleep(.005)
    frames = cams.latest()
    for role in ('front', 'top'):
        n = len(stamps.get(role, ()))
        if role not in frames:
            out.append(_r(f'camera:{role}', 'FAIL', '没有画面'))
            continue
        im = frames[role][0]
        hz, mean, std = n/seconds, float(im.mean()), float(im.std())
        ok = hz >= 25 and 15 < mean < 240 and std > 5
        out.append(_r(f'camera:{role}', 'PASS' if ok else 'FAIL',
                      f'{hz:.0f} Hz，亮度均值 {mean:.0f}，标准差 {std:.0f}（需要 ≥ 25 Hz，图像不能全黑或过曝）'))
    return out


def check_disk(out_dir):
    free = shutil.disk_usage(out_dir if out_dir.exists() else Path('.')).free/2**30
    return [_r('disk', 'PASS' if free > 1 else 'WARN', f'日志目录剩余 {free:.1f} GB')]


def run(cfg, recs, args, out_dir, left=None, right=None, cams=None, selftest=True):
    res = check_package()+check_software()+check_blas()+check_load()
    if selftest:
        try:
            res += check_selftest(cfg, recs, ROOT/args.checkpoint, args.commit)
        except Exception as e:                       # noqa: BLE001
            res.append(_r('selftest', 'FAIL', f'Controller 自检异常 ({e!r})'))
    for name, arm in (('left', left), ('right', right)):
        if arm is not None:
            res += check_robot(arm, name)
    res += check_cameras(cams)+check_disk(out_dir)
    print('\n  ---- 运行前安全检查 ----')
    for r in res:
        print(f"    [{r['level']:4s}] {r['name']:22s} {r['msg']}")
    fails = [r for r in res if r['level'] == 'FAIL']
    print(f"    => {'通过' if not fails else f'{len(fails)} 项未通过'}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'preflight.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return not fails, res


def touch_check(left, right, points=3, tol_m=.005):
    """Operator touches both probe tips together (freedrive on the pendants) at `points` places."""
    print('\n  ---- 双探针对点检查（核对两个底座之间的变换）----')
    print('  在示教器上用自由驱动把两根探针的尖端轻轻碰在一起，分散在工作区不同位置。')
    errs = []
    for k in range(points):
        input(f'  第 {k+1}/{points} 个点：两尖端接触后按回车 ')
        pl = left.tcp()[:3]
        pr = T_RIGHT[:3, :3] @ right.tcp()[:3]+T_RIGHT[:3, 3]
        e = float(np.linalg.norm(pl-pr))
        errs.append(e)
        print(f'    左 {np.round(pl, 4).tolist()}  右(世界系) {np.round(pr, 4).tolist()}  差 {e*1e3:.1f} mm')
    ok = max(errs) <= tol_m
    print(f"  => 最大差 {max(errs)*1e3:.1f} mm，{'通过' if ok else f'超过 {tol_m*1e3:.0f} mm：底座变换或 TCP 设置不对，不要运行'}")
    return ok, errs
