"""Install receipt hooks without importing or starting the Nero server.

Run with --server /absolute/path/nero_interface_server.py [--apply]. The original
file is backed up, syntax is checked, and unknown source shapes fail closed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import shutil


def patch_source(source):
    if "class NeroDualArmServer(PhysicalProtocolMixin):" in source:
        return source
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NeroDualArmServer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "servo_p_OL")
    lines = source.splitlines(keepends=True)
    edits = []
    # Every explicit refusal is known to have sent no target. Exceptions stay
    # 'unknown', because a transport error cannot prove a command was not sent.
    handler_returns = {
        n.lineno for handler in ast.walk(method) if isinstance(handler, ast.ExceptHandler)
        for n in ast.walk(handler) if isinstance(n, ast.Return)
    }
    for node in ast.walk(method):
        if (isinstance(node, ast.Return) and isinstance(node.value, ast.Constant)
                and node.value.value is False and node.lineno not in handler_returns):
            indent = lines[node.lineno - 1][:node.col_offset]
            edits.append((node.lineno - 1, indent + 'self._physical_outcome = "rejected"\n'))
    text = "".join(lines[method.lineno - 1:method.end_lineno])
    reference = "                cur_xyz = np.asarray(cur_pose[:3], dtype=float)"
    sent = "            robot.move_js(q_cmd)\n"
    limit = "                pose = self._limit_pose_delta(pose)\n"
    if text.count(reference) != 1 or text.count(sent) != 1 or text.count(limit) != 1:
        raise ValueError("Unrecognized Nero integration/send layout; review source before patching")
    for index in range(method.lineno - 1, method.end_lineno):
        if reference in lines[index]:
            edits.append((index, "                physical_reference = cur_pose.copy()\n"))
        if lines[index] == limit:
            edits.append((index, "                pose = self._physical_prepare_delta(robot_arm, pose)\n"))
        if lines[index] == sent:
            edits.append((index + 1, "            if delta:\n"
                         "                self._physical_receipt = (physical_reference.copy(), target_pose.copy())\n"))
    if not handler_returns or len(edits) < 4:
        raise ValueError("Unrecognized Nero rejection paths")
    for index, insertion in sorted(edits, reverse=True):
        lines.insert(index, insertion)
    result = "".join(lines).replace("class NeroDualArmServer:",
                                  "from physical_protocol import PhysicalProtocolMixin, serialized_cartesian\n\n\n"
                                  "class NeroDualArmServer(PhysicalProtocolMixin):", 1)
    result = result.replace("    def servo_p_OL(self,", "    @serialized_cartesian\n    def servo_p_OL(self,", 1)
    compile(result, "patched_nero_server", "exec")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    old = args.server.read_text()
    new = patch_source(old)
    print(f"source_sha256={hashlib.sha256(old.encode()).hexdigest()}")
    print(f"changed={old != new}; service_start=False")
    if args.apply:
        if old != new:
            backup = args.server.with_suffix(".py.before-physical-v2")
            if backup.exists() and backup.read_text() != old:
                raise FileExistsError(f"Refusing to overwrite previous backup {backup}")
            if not backup.exists():
                shutil.copy2(args.server, backup)
            args.server.write_text(new)
        module = Path(__file__).with_name("physical_protocol.py")
        destination = args.server.with_name(module.name)
        if module.resolve() != destination.resolve():
            shutil.copy2(module, destination)


if __name__ == "__main__":
    main()
