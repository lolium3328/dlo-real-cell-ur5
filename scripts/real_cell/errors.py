"""Error codes and failure reports for the real-cell runner.

Every non-success ending (abort, fault, timeout, planning / homing / preflight failure, exception) produces a report:
printed as a block for the operator and written as <out>/<scene>_error.json with the code, the measured values,
the state of both robots at that moment, the last log rows and a traceback when there is one.
"""
import json
import time
import traceback

# code, title, likely causes, operator actions
CATALOG = {
    'W001': ('超时：30 s 内未达到成功标准',
             ['策略没能把绳形收敛到目标（rmse ≤ 12 mm、p95 ≤ 25 mm、末端 ≤ 16 mm，并保持 0.4 s）',
              '跟踪估计有偏差（相机外参、光照），策略按错误的绳形在调整'],
             ['不是安全故障；查看日志里 rmse / end 的走势', '检查相机图像和跟踪质量后可重跑该场景']),
    'E101': ('机械臂保护性停止 / 急停',
             ['碰撞或外力触发保护性停止', '急停按钮被按下', '控制器安全限制被触发（关节限位、安全平面、速度）'],
             ['在示教器上查看日志并确认原因', '清除保护性停止后，人工把机械臂移到安全位置，重新运行（会重新归位）']),
    'E102': ('左臂跟随误差过大',
             ['左臂被阻挡或碰撞', '示教器速度滑块低于 100%', 'servoL 跟不上（控制器负载、网络延迟）'],
             ['检查左臂周围有无障碍和绳子缠绕', '示教器速度滑块调到 100%', '检查网线直连、无交换机拥塞']),
    'E103': ('右臂跟随误差过大',
             ['右臂被阻挡或碰撞', '示教器速度滑块低于 100%', '规划轨迹的关节速度对实机太快'],
             ['检查右臂周围', '示教器速度滑块调到 100%', '查看 feasibility 报告中的 right_max_joint_speed_rad_s']),
    'E104': ('两臂之间 / 球与左臂的胶囊间隙低于阈值',
             ['策略路径使左臂靠近右臂或球（运行时按实际关节计算）', '工具尺寸大于胶囊模型（工具半径按 40 mm 估计）'],
             ['查看报告中的 gap_arm / gap_ball 值和时间', '若是模型过保守，测量真实工具尺寸后调整 geometry.RADII']),
    'E105': ('球与绳的估计间隙低于阈值',
             ['球即将碰到绳：策略/CBF 没能避开', '跟踪估计的中心线有偏差'],
             ['检查绳子是否被碰到、是否需要复位', '查看视频/图像确认跟踪是否正确']),
    'E201': ('视觉 / 跟踪故障',
             ['相机断开或帧格式错误', '跟踪器无法初始化（第一帧看不到绳子）', '跟踪 / 规划线程异常'],
             ['检查两个 RealSense 的 USB 连接', '确认绳子在两个视野内、光照正常', '查看报告中的 fault 文本']),
    'E202': ('跟踪长时间失效（observation_stale 超过限制）',
             ['绳子被遮挡或离开视野', '跟踪质量门持续暂停'],
             ['检查遮挡物和光照', '确认绳子初始状态与仿真初始形状接近']),
    'E203': ('相机没有画面', ['相机未连接或被其他程序占用', 'USB 带宽不足'],
             ['关闭其他占用相机的程序', '两个相机分别接到不同的 USB3 控制器']),
    'E204': ('Controller 一直没有进入 tracking 状态',
             ['跟踪器在预热期内无法稳定', '球的读数不可用（右臂 TCP 数据异常）'],
             ['检查绳子是否在视野内', '检查右臂 TCP 设置']),
    'E205': ('机器人数据过期（控制环超过 0.1 s 没有拿到新状态）',
             ['电脑负载过高，伺服线程被拖慢', 'RTDE 通信中断'],
             ['关闭电脑上的其他任务', '检查网线；确认 BLAS 单线程（preflight 会检查）']),
    'E206': ('时钟 / 输入异常', ['时间戳非单调或输入非有限值（程序内部错误）'],
             ['保存报告并反馈；不要重复运行同一配置']),
    'E301': ('约束不可行（execute_action 无法给出满足加速度与工作区限制的动作）',
             ['指令点贴着工作区边界且速度朝外', '工作区配置与场景不一致'],
             ['查看报告中的 left_cmd 与工作区边界', '重新运行；若复现请反馈']),
    'E302': ('左臂在工作区之外', ['归位后的位置与场景起点不一致', '工作区定义与实机标定不一致'],
             ['重新归位；检查 TCP 偏置是否为探针尖端 [0, 0, 0.165]']),
    'E303': ('策略输出了非法指令（非有限值或超过速度上限）', ['程序内部错误'], ['保存报告并反馈']),
    'E401': ('Controller 异常', ['程序内部异常，见 traceback'], ['保存报告并反馈']),
    'E402': ('伺服环异常', ['与机器人通信异常或程序内部异常，见 traceback'],
             ['检查网络和机器人状态；确认示教器处于远程控制模式']),
    'E403': ('工具姿态在规划后被改变', ['规划完成后有人移动了机械臂'], ['重新运行该场景（会重新规划）']),
    'E501': ('无法连接机器人（RTDE）',
             ['IP 错误或网线未连接', '示教器上有程序在运行或有弹窗（CB3）；e-Series 未处于远程控制（Remote Control）模式', '控制器上 RTDE / 外部控制服务被禁用',
              '另一个程序已连接该机器人'],
             ['ping 机器人 IP', '示教器切换到远程控制', '设置 → 安全 → 服务 中启用 RTDE 和 Primary/Secondary 接口',
              '关闭其他连接机器人的程序']),
    'E601': ('场景可行性检查未通过', ['IK、安全限制、关节速度或两臂间隙检查失败，见 fail 列表'],
             ['按 fail 列表处理；不要用 --ignore-check 强行运行，除非确认是模型过保守']),
    'E602': ('归位失败或归位后检查未通过', ['moveL / moveJ 被控制器拒绝或中断', '归位位置 / 关节分支 / 工具姿态与规划不一致'],
             ['人工把机械臂移到接近仿真初始构型的位置后重跑', '--ignore-check 不会跳过这一项']),
    'E700': ('运行前安全检查未通过', ['见各项检查结果'], ['逐项处理 FAIL 项后重新运行 preflight']),
    'E901': ('人为中止（Ctrl-C）', ['操作员中止'], ['确认机械臂已停止；需要时人工复位绳子']),
    'E902': ('运行被意外停止', ['程序内部状态异常'], ['保存报告并反馈']),
}

TERMINAL_CODES = {
    'timeout': 'W001', 'robot_fault': 'E101', 'left_lag_abort': 'E102', 'right_lag_abort': 'E103',
    'abort_arm_clearance': 'E104', 'abort_surface_clearance': 'E105', 'abort_observation_fault': 'E201',
    'abort_stale_too_long': 'E202', 'camera_timeout': 'E203', 'abort_never_tracking': 'E204',
    'abort_robot_stale': 'E205', 'abort_invalid_robot_clock': 'E206', 'abort_invalid_clock': 'E206',
    'abort_invalid_input': 'E206', 'abort_constraint_infeasible': 'E301', 'abort_outside_workspace': 'E302',
    'outside_workspace_at_start': 'E302', 'abort_bad_command': 'E303', 'orientation_changed': 'E403',
    'user_interrupt': 'E901', 'stopped': 'E902',
}

ROBOT_MODE = {-1: 'NO_CONTROLLER', 0: 'DISCONNECTED', 1: 'CONFIRM_SAFETY', 2: 'BOOTING', 3: 'POWER_OFF', 4: 'POWER_ON',
              5: 'IDLE', 6: 'BACKDRIVE', 7: 'RUNNING', 8: 'UPDATING_FIRMWARE'}
SAFETY_MODE = {1: 'NORMAL', 2: 'REDUCED', 3: 'PROTECTIVE_STOP', 4: 'RECOVERY', 5: 'SAFEGUARD_STOP',
               6: 'SYSTEM_EMERGENCY_STOP', 7: 'ROBOT_EMERGENCY_STOP', 8: 'VIOLATION', 9: 'FAULT',
               10: 'VALIDATE_JOINT_ID', 11: 'UNDEFINED_SAFETY_MODE', 12: 'AUTOMATIC_MODE_SAFEGUARD_STOP',
               13: 'SYSTEM_THREE_POSITION_ENABLING_STOP'}


def code_for(terminal):
    if terminal.startswith('controller_error'):
        return 'E401'
    if terminal.startswith('servo_error'):
        return 'E402'
    return TERMINAL_CODES.get(terminal, 'E902')


def robot_status(arm):
    try:
        return arm.status()
    except Exception as e:                       # noqa: BLE001 - a report must never fail
        return dict(error=repr(e))


def _fmt_status(st):
    if 'error' in st:
        return f"读取失败 ({st['error']})"
    rm, sm = ROBOT_MODE.get(st.get('robot_mode'), st.get('robot_mode')), SAFETY_MODE.get(st.get('safety_mode'), st.get('safety_mode'))
    return (f"{rm} / {sm}, 保护停止={st.get('protective')}, 急停={st.get('emergency')}, "
            f"速度比例={st.get('speed_scaling')}, 控制脚本运行={st.get('program_running')}")


def report(out_dir, scene, terminal, *, code=None, detail=None, left=None, right=None, servo_log=None,
           control_log=None, exc=None, extra=None):
    """Print the operator block and write <out_dir>/<scene>_error.json. Returns the report dict."""
    code = code or code_for(terminal)
    title, causes, actions = CATALOG.get(code, ('未知错误', [], ['保存报告并反馈']))
    last_c = (control_log or [])[-1] if control_log else {}
    rep = dict(code=code, title=title, scene=scene, terminal=terminal, wall_time=time.strftime('%Y-%m-%d %H:%M:%S'),
               t_s=last_c.get('t'), ball_clock_s=last_c.get('s'), last_status=last_c.get('status'),
               detail=detail or {}, causes=causes, actions=actions,
               robots=dict(left=robot_status(left) if left else None, right=robot_status(right) if right else None),
               last_servo=(servo_log or [])[-25:], last_control=(control_log or [])[-25:],
               traceback=''.join(traceback.format_exception(exc)) if exc is not None else None, **(extra or {}))
    path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir/f'{scene}_error.json'
        path.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    bar = '=' * 72
    lines = [bar, f'  {"警告" if code.startswith("W") else "错误"} {code}  {title}', f'  场景: {scene}    终止: {terminal}']
    if rep['t_s'] is not None:
        lines.append(f"  时间: t = {rep['t_s']:.2f} s（球时钟 {rep['ball_clock_s']:.2f} s），最后状态 {rep['last_status']}")
    for k, v in (detail or {}).items():
        lines.append(f'  {k}: {v}')
    for name in ('left', 'right'):
        st = rep['robots'][name]
        if st is not None:
            lines.append(f'  {"左" if name == "left" else "右"}臂: {_fmt_status(st)}')
    lines.append('  可能原因:')
    lines += [f'    - {c}' for c in causes]
    lines.append('  处理:')
    lines += [f'    - {a}' for a in actions]
    if exc is not None:
        lines.append(f'  异常: {exc!r}')
    if path is not None:
        lines.append(f'  报告文件: {path}')
    lines.append(bar)
    print('\n'.join(lines), flush=True)
    return rep
