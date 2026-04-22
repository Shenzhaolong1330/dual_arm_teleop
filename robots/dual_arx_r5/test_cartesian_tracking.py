"""
笛卡尔空间轨迹跟踪测试 — 绕开 VR 输入，纯 IK + RPC 验证。

测试内容:
  1. 连接 RPC, 读取当前关节位置
  2. 初始化 Placo IK solver, 同步到当前构型
  3. 以固定频率发送 delta EE pose (直线/圆弧轨迹)
  4. 记录每步耗时, 判断管线是否流畅

用法:
  python robots/dual_arx_r5/test_cartesian_tracking.py

注意: 确保 ARX RPC server 已启动 (robot_ip 和 port 匹配)
"""

import time
import logging
import numpy as np
from pathlib import Path
from robots.dual_arx_r5.arx_interface_client import ArxDualArmClient
from robots.dual_arx_r5.arx_ik_solver import ArxR5IKSolver

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================
ROBOT_IP = "192.168.110.57"
ROBOT_PORT = 4242
URDF_PATH = str(Path(__file__).parents[2] / "assets" / "arx_r5" / "dual_R5a.urdf")
SERVO_TIME = 0.017                  # Placo solver dt

FREQ_HZ = 30                       # 控制频率 (与 record_loop 一致)
DURATION_SEC = 5                    # 测试总时长
MAX_JOINT_DELTA = 0.3               # 关节钳位 (rad/step)

# 轨迹参数: 右臂沿 x 方向前进 5cm 再返回, 左臂保持不动
TRAJECTORY = "line_x"               # "line_x" | "circle_xz" | "hold"
LINE_AMPLITUDE = 0.10               # 直线幅度 (m), 前进+后退各 5cm
CIRCLE_RADIUS = 0.03                # 圆弧半径 (m)


def clamp_joint_delta(target: np.ndarray, current: np.ndarray, max_d: float) -> np.ndarray:
    """限制单步关节变化量。"""
    delta = target - current
    clipped = np.clip(delta, -max_d, max_d)
    return current + clipped


def generate_delta(t: float, trajectory: str, freq: float) -> np.ndarray:
    """生成当前时刻的 delta EE pose [dx, dy, dz, drx, dry, drz]。

    Args:
        t: 当前时间 (秒)
        trajectory: 轨迹类型
        freq: 控制频率 (Hz)

    Returns:
        6D delta (米/弧度), 表示这一步的增量
    """
    dt = 1.0 / freq
    delta = np.zeros(6)

    if trajectory == "line_x":
        # 三角波: 前半段前进, 后半段后退
        period = DURATION_SEC
        # 速度 (m/s): 走 amplitude 用 period/2 秒
        speed = LINE_AMPLITUDE / (period / 2)
        half = period / 2
        if t < half:
            delta[0] = speed * dt       # +x 方向前进
        else:
            delta[0] = -speed * dt      # -x 方向后退

    elif trajectory == "circle_xz":
        # 在 xz 平面画圆
        omega = 2 * np.pi / DURATION_SEC       # 角速度
        delta[0] = -CIRCLE_RADIUS * omega * np.sin(omega * t) * dt   # dx
        delta[2] = CIRCLE_RADIUS * omega * np.cos(omega * t) * dt    # dz

    elif trajectory == "hold":
        pass    # 保持不动, delta 全零

    return delta


def main():
    # ============================================================
    # 1. 连接 RPC
    # ============================================================
    logger.info(f"Connecting to ARX RPC server at {ROBOT_IP}:{ROBOT_PORT} ...")
    client = ArxDualArmClient(ip=ROBOT_IP, port=ROBOT_PORT)
    if not client.system_connect(timeout=10.0):
        raise RuntimeError("Failed to connect to ARX server")

    state = client.get_full_state()
    if state is None:
        raise RuntimeError("Failed to read initial state")

    left_joints = state["left_arm"]["joint_positions"][:6].copy()
    right_joints = state["right_arm"]["joint_positions"][:6].copy()
    left_gripper = state["left_arm"]["gripper"]
    right_gripper = state["right_arm"]["gripper"]

    logger.info(f"Initial left  joints: {np.round(left_joints, 4)}")
    logger.info(f"Initial right joints: {np.round(right_joints, 4)}")

    # ============================================================
    # 2. 初始化 IK solver
    # ============================================================
    logger.info(f"Loading IK solver from {URDF_PATH} ...")
    solver = ArxR5IKSolver(URDF_PATH, servo_time=SERVO_TIME)
    solver.sync_from_joint_positions(left_joints, right_joints)
    logger.info("IK solver initialized and synced to current pose.")

    # ============================================================
    # 3. 轨迹跟踪循环
    # ============================================================
    total_steps = int(FREQ_HZ * DURATION_SEC)
    dt = 1.0 / FREQ_HZ

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting trajectory: {TRAJECTORY}")
    logger.info(f"  Frequency: {FREQ_HZ} Hz, Duration: {DURATION_SEC}s, Steps: {total_steps}")
    logger.info(f"  Line amplitude: {LINE_AMPLITUDE*100:.1f} cm")
    logger.info(f"{'='*60}\n")

    timing_sync = []    # IK sync 耗时
    timing_ik = []      # IK solve 耗时
    timing_rpc = []     # RPC send 耗时
    timing_total = []   # 整步耗时

    cached_left = left_joints.copy()
    cached_right = right_joints.copy()

    try:
        t_start = time.perf_counter()

        for step in range(total_steps):
            t_step_start = time.perf_counter()
            t_elapsed = t_step_start - t_start

            # 生成右臂的 delta (左臂保持)
            right_delta = generate_delta(t_elapsed, TRAJECTORY, FREQ_HZ)
            left_delta = np.zeros(6)    # 左臂不动

            # IK sync: 从缓存关节同步 Placo 模型
            t0 = time.perf_counter()
            solver.sync_from_joint_positions(cached_left, cached_right)
            t_sync_ms = (time.perf_counter() - t0) * 1e3

            # IK solve: 应用 delta, 求解关节角
            t0 = time.perf_counter()
            new_left, new_right = solver.solve_delta_ee(left_delta, right_delta)
            t_ik_ms = (time.perf_counter() - t0) * 1e3

            # 关节钳位
            new_left = clamp_joint_delta(new_left, cached_left, MAX_JOINT_DELTA)
            new_right = clamp_joint_delta(new_right, cached_right, MAX_JOINT_DELTA)

            # RPC 发送
            left_7 = np.append(new_left, left_gripper)
            right_7 = np.append(new_right, right_gripper)

            t0 = time.perf_counter()
            client.set_dual_joint_positions(left_7, right_7)
            t_rpc_ms = (time.perf_counter() - t0) * 1e3

            # 更新缓存
            cached_left = new_left.copy()
            cached_right = new_right.copy()

            t_total_ms = (time.perf_counter() - t_step_start) * 1e3
            timing_sync.append(t_sync_ms)
            timing_ik.append(t_ik_ms)
            timing_rpc.append(t_rpc_ms)
            timing_total.append(t_total_ms)

            # 每 30 步打印
            if (step + 1) % 30 == 0:
                logger.info(
                    f"[Step {step+1:4d}/{total_steps}] "
                    f"total={t_total_ms:.1f}ms  sync={t_sync_ms:.1f}ms  "
                    f"ik={t_ik_ms:.1f}ms  rpc={t_rpc_ms:.1f}ms  "
                    f"delta_R={np.round(right_delta, 4)}"
                )

            # 等待到下一步
            t_elapsed_step = time.perf_counter() - t_step_start
            sleep_time = dt - t_elapsed_step
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("\n[INTERRUPTED] Ctrl+C detected, stopping...")
    finally:
        # ============================================================
        # 4. 统计
        # ============================================================
        sync_arr = np.array(timing_sync)
        ik_arr = np.array(timing_ik)
        rpc_arr = np.array(timing_rpc)
        total_arr = np.array(timing_total)

        logger.info(f"\n{'='*60}")
        logger.info(f"Performance Summary ({len(total_arr)} steps)")
        logger.info(f"{'='*60}")
        logger.info(f"  {'':12s}  {'mean':>8s}  {'p50':>8s}  {'p95':>8s}  {'max':>8s}")
        for name, arr in [("sync", sync_arr), ("ik_solve", ik_arr),
                          ("rpc", rpc_arr), ("total", total_arr)]:
            if len(arr) > 0:
                logger.info(
                    f"  {name:12s}  {np.mean(arr):7.1f}ms  {np.median(arr):7.1f}ms  "
                    f"{np.percentile(arr, 95):7.1f}ms  {np.max(arr):7.1f}ms"
                )

        over_budget = np.sum(total_arr > (1000 / FREQ_HZ))
        logger.info(f"\n  Overruns (>{1000/FREQ_HZ:.0f}ms): {over_budget}/{len(total_arr)} "
                     f"({100*over_budget/max(len(total_arr),1):.1f}%)")

        logger.info(f"\nFinal right joints: {np.round(cached_right, 4)}")
        logger.info(f"Initial right joints: {np.round(right_joints, 4)}")
        logger.info(f"Joint change: {np.round(cached_right - right_joints, 4)}")

        client.disconnect()
        logger.info("Done.")


if __name__ == "__main__":
    main()
