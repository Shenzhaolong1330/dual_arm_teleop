"""Exercise the actual patched ServoP method with fake arms/IK, without hardware.

Only the audited method and its limit helper are compiled from the supplied
source. The server module, constructor, SDK drivers and CAN are never loaded.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import logging
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robots.dual_agilex_nero.install_physical_protocol import patch_source
from robots.dual_agilex_nero.physical_protocol import (
    PhysicalProtocolMixin, serialized_cartesian, validate_receipt, rpy_pose_to_rotvec,
)


def verify(server_path, tf_path):
    tree = ast.parse(patch_source(Path(server_path).read_text()))
    server = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NeroDualArmServer")
    server.body = [n for n in server.body if isinstance(n, ast.FunctionDef)
                   and n.name in {"servo_p_OL", "_limit_pose_delta"}]
    quat = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "quat_multiply")
    pure_tree = ast.Module(body=[quat, server], type_ignores=[])
    spec = importlib.util.spec_from_file_location("pyAgxArm.utiles.tf", tf_path)
    tf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tf)
    # Supply only the SDK's pure rotation utilities to the method's local import.
    sys.modules["pyAgxArm"] = ModuleType("pyAgxArm")
    sys.modules["pyAgxArm.utiles"] = ModuleType("pyAgxArm.utiles")
    sys.modules["pyAgxArm.utiles.tf"] = tf
    ns = dict(np=np, PhysicalProtocolMixin=PhysicalProtocolMixin,
              serialized_cartesian=serialized_cartesian, log=logging.getLogger("nero-offline"))
    exec(compile(pure_tree, str(server_path), "exec"), ns)

    class IK:
        def __init__(self):
            self.state = None
            self.fail = False

        def init_state(self, q):
            self.state = SimpleNamespace(q_prev=np.asarray(q))

        def fk_pose(self, q):
            return np.array([.4, .2, .3, .5, -.3, .7])

        def solve(self, pose, limit_output_step):
            return None if self.fail else np.zeros(7)

    sent = []
    s = ns["NeroDualArmServer"]()
    s.left_robot = SimpleNamespace(move_js=lambda q: sent.append(q))
    s.right_robot = s.left_robot
    s.left_ik_solver, s.right_ik_solver = IK(), IK()
    s.left_cur_pose = s.right_cur_pose = None
    s._get_current_joints = lambda robot, timeout: np.zeros(7)
    s.max_cart_step_m, s.max_rot_step_rad = .03, .35
    s.ik_resync_idle_s, s.max_ik_solve_ms = .5, 30
    s.ik_delta_min_scale, s.ik_delta_decay, s.ik_delta_recover = .2, .6, .08
    s.ik_orientation_freeze_s, s.ik_fail_cooldown_s = .2, .25
    s._ik_skip_until = dict(left_robot=0., right_robot=0.)
    s._ik_delta_scale = dict(left_robot=1., right_robot=1.)
    s._ik_freeze_rot_until = dict(left_robot=0., right_robot=0.)
    s.enable_servo_timing_print, s.limit_z = False, None
    identity = [0, 0, 0, 1, 0, 0, 0]

    reply = s.servo_p_OL_v2("left_robot", [.1, 0, 0, .02, -.01, .03], identity)
    np.testing.assert_allclose(validate_receipt(reply), [.03, 0, 0, .02, -.01, .03], atol=1e-10)
    np.testing.assert_allclose(reply["reference_pose"], rpy_pose_to_rotvec(IK().fk_pose(None)), atol=1e-10)
    # Initial IK state already exists while the cached Cartesian goal is None.
    s.left_cur_pose = None
    validate_receipt(s.servo_p_OL_v2("left_robot", [.001, 0, 0, 0, 0, 0], identity))
    s._ik_delta_scale["left_robot"] = .25
    s._ik_freeze_rot_until["left_robot"] = time.perf_counter() + 10
    reply = s.servo_p_OL_v2("left_robot", [.1, 0, 0, .1, .2, -.1], identity)
    np.testing.assert_allclose(validate_receipt(reply), [.0075, 0, 0, 0, 0, 0], atol=1e-10)
    s._ik_delta_scale["left_robot"] = 1
    s._ik_freeze_rot_until["left_robot"] = 0
    tcp = [.12, .01, -.02, 1, 0, 0, 0]
    reply = s.servo_p_OL_v2("left_robot", [0, 0, 0, .02, -.01, .01], tcp)
    np.testing.assert_allclose(validate_receipt(reply), [0, 0, 0, .02, -.01, .01], atol=1e-10)
    count = len(sent)
    s.left_ik_solver.fail = True
    reply = s.servo_p_OL_v2("left_robot", [.01, 0, 0, 0, 0, 0], identity)
    np.testing.assert_array_equal(validate_receipt(reply), np.zeros(6))
    assert len(sent) == count
    print("PASS: audited ServoP clipping, RPY/rotvec, first frame, attenuation, freeze, TCP lever arm, IK rejection; hardware_connected=False")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", type=Path, required=True)
    p.add_argument("--sdk-tf", type=Path, required=True)
    a = p.parse_args()
    verify(a.server, a.sdk_tf)
