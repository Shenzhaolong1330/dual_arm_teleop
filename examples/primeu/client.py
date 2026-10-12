"""Check and control the existing PRIMEU ROS runtime through its GR00T backend.

This client does not import LeRobot, load a model, or implement robot drivers.
``launch`` starts the runtime; only explicit ``start`` enables its policy loop.
"""

import argparse
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import yaml

# Wire-contract constants only: this file can be copied to the robot computer.
JOINT_DIMS = {"left_arm": 7, "left_hand": 20, "right_arm": 7, "right_hand": 20, "head": 3, "waist": 3}
CAMERAS = ["head", "left_wrist", "right_wrist"]
DEFAULT_CONFIG = Path(__file__).parent / "configs/robot.yaml"
ACTION_DIMS = {60, 33, 27, 15, 12}


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict) or set(cfg) != {"server", "runtime"}:
        raise ValueError("robot.yaml must contain only server and runtime")
    server, runtime = cfg["server"], cfg["runtime"]
    required_server = {
        "host", "port", "timeout_s", "expected_horizon", "expected_model_family",
        "expected_model_action_dim", "expected_basis_sha256",
    }
    image_options = {"expected_image_size", "expected_image_sizes"}
    if not isinstance(server, dict) or not required_server <= set(server) or set(server) - required_server - image_options:
        raise ValueError(f"server needs {sorted(required_server)}; optional image settings: {sorted(image_options)}")
    if not isinstance(runtime, dict) or set(runtime) != {
        "workspace",
        "setup_file",
        "ros_setup_file",
        "node",
        "task_description",
        "service_timeout_s",
    }:
        raise ValueError("Unexpected or missing runtime settings; use configs/robot.yaml")
    host = server["host"]
    if not isinstance(host, str) or not host or host in {"*", "0.0.0.0", "::"} or any(c.isspace() for c in host):
        raise ValueError("server.host must be the reachable server/tunnel address, not a bind address")
    if type(server["port"]) is not int or not 1 <= server["port"] <= 65535:
        raise ValueError("server.port must be an integer in 1..65535")
    if type(server["expected_horizon"]) is not int or server["expected_horizon"] < 1:
        raise ValueError("expected_horizon must be a positive integer")
    if server["expected_model_family"] not in {"act", "diffusion"}:
        raise ValueError("expected_model_family must be act or diffusion")
    if type(server["expected_model_action_dim"]) is not int or server["expected_model_action_dim"] not in ACTION_DIMS:
        raise ValueError(f"expected_model_action_dim must be one of {sorted(ACTION_DIMS)}")
    basis = server["expected_basis_sha256"]
    if not isinstance(basis, dict) or any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
        for k, v in basis.items()
    ):
        raise ValueError("expected_basis_sha256 must match the checkpoint basis identifiers and lowercase SHA256 digests")
    if (server["expected_model_action_dim"] == 60) != (not basis):
        raise ValueError("60D requires an empty basis map; reduced actions require the checkpoint basis hashes")
    if image_options <= set(server):
        raise ValueError("expected_image_size and expected_image_sizes are mutually exclusive")
    if "expected_image_sizes" in server:
        sizes = server["expected_image_sizes"]
        if not isinstance(sizes, dict) or set(sizes) != set(CAMERAS):
            raise ValueError(f"expected_image_sizes must specify exactly {CAMERAS}")
    else:
        sizes = {"all cameras": server.setdefault("expected_image_size", [256, 256])}
    for camera, size in sizes.items():
        if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
            raise ValueError(f"Expected image size for {camera} must be [height, width] with positive integers")
    for value in (server["timeout_s"], runtime["service_timeout_s"]):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 300:
            raise ValueError("Timeouts must be finite seconds in (0, 300]")
    if not isinstance(runtime["task_description"], str) or not runtime["task_description"].strip():
        raise ValueError("task_description must be nonempty")
    if not isinstance(runtime["node"], str) or not re.fullmatch(r"/[A-Za-z_]\w*(/[A-Za-z_]\w*)*", runtime["node"]):
        raise ValueError("runtime.node must be an absolute ROS node path")
    for key in ("workspace", "setup_file"):
        if not isinstance(runtime[key], str) or not runtime[key]:
            raise ValueError(f"runtime.{key} must be a nonempty path")
    if runtime["ros_setup_file"] is not None and (
        not isinstance(runtime["ros_setup_file"], str) or not runtime["ros_setup_file"]
    ):
        raise ValueError("ros_setup_file must be null or a setup.bash path")
    return cfg


def rpc(server, endpoint):
    """Read metadata or clear policy state; never request or execute robot actions."""
    import msgpack
    import zmq

    try:
        with zmq.Context() as context:
            with context.socket(zmq.REQ) as socket:
                timeout = max(1, int(server["timeout_s"] * 1000))
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, timeout)
                socket.setsockopt(zmq.SNDTIMEO, timeout)
                socket.connect(f"tcp://{server['host']}:{server['port']}")
                socket.send(msgpack.packb({"endpoint": endpoint}, use_bin_type=True))
                result = msgpack.unpackb(socket.recv(), raw=False)
    except zmq.Again as exc:
        raise TimeoutError(f"Policy server timed out during {endpoint}; no automatic retry") from exc
    except zmq.ZMQError as exc:
        raise ConnectionError(f"Policy server connection failed during {endpoint}; no automatic retry: {exc}") from exc
    if not isinstance(result, dict) or "error" in result:
        raise RuntimeError(f"Invalid {endpoint} response: {result}")
    return result


def check_server(cfg):
    start = time.monotonic()
    server = cfg["server"]
    ping = rpc(server, "ping")
    contract = ping.get("contract", {})
    expected = {
        "protocol": "primeu_lerobot_v1",
        "model_family": server["expected_model_family"],
        "state_dim": 60,
        "action_dim": 60,
        "model_action_dim": server["expected_model_action_dim"],
        "joint_groups": JOINT_DIMS,
        "joint_order": list(JOINT_DIMS),
        "camera_keys": CAMERAS,
        "action_representation": "absolute_joint_target",
        "action_horizon": server["expected_horizon"],
        "basis_sha256": server["expected_basis_sha256"],
    }
    if "expected_image_sizes" in server:
        sizes = server["expected_image_sizes"]
        unique_sizes = {tuple(size) for size in sizes.values()}
        expected["image_sizes"] = sizes
        expected["image_size"] = list(next(iter(unique_sizes))) if len(unique_sizes) == 1 else None
    else:
        expected["image_size"] = server["expected_image_size"]
    if ping.get("status") != "ok" or not isinstance(contract, dict) or any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Server is not the expected LeRobot PRIMEU policy/basis: {contract}")
    modalities = rpc(server, "get_modality_config")
    expected_keys = {
        "video": CAMERAS,
        "state": list(JOINT_DIMS),
        "action": list(JOINT_DIMS),
        "language": ["annotation.human.task_description"],
    }
    for name, keys in expected_keys.items():
        spec = modalities.get(name, {})
        if not isinstance(spec, dict) or spec.get("__ModalityConfig__") is not True:
            raise ValueError(f"Malformed {name} modality")
        value = spec.get("as_json")
        if not isinstance(value, dict):
            raise ValueError(f"Malformed {name} modality values")
        indices = list(range(server["expected_horizon"])) if name == "action" else [0]
        if value.get("modality_keys") != keys or value.get("delta_indices") != indices:
            raise ValueError(f"Unexpected {name} order or horizon: {value}")
        if name == "action":
            action_format = {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT"}
            if value.get("action_configs") != [action_format] * len(JOINT_DIMS):
                raise ValueError("Server actions must be absolute NON_EEF joint targets")
    return {
        "server": f"{server['host']}:{server['port']}",
        "contract": contract,
        "handshake_ms": (time.monotonic() - start) * 1000,
        "robot_motion": False,
    }


def reset_server(cfg):
    """Clear queues and held posture; the next valid observation anchors a new run."""
    result = rpc(cfg["server"], "reset")
    if result.get("status") != "ok" or result.get("robot_motion") is not False:
        raise RuntimeError(f"Policy reset did not confirm success without robot motion: {result}")
    return result


def ros_command(cfg, operation, *, check_paths=True):
    runtime, server = cfg["runtime"], cfg["server"]
    workspace = Path(runtime["workspace"]).expanduser()
    setup = Path(runtime["setup_file"]).expanduser()
    if not setup.is_absolute():
        setup = workspace / setup
    setups = []
    if runtime["ros_setup_file"]:
        setups.append(Path(runtime["ros_setup_file"]).expanduser())
    setups.append(setup)
    if check_paths:
        for path in setups:
            if not path.is_file():
                raise FileNotFoundError(f"Robot ROS setup missing: {path}. Run this operation on the robot computer.")
    if operation == "launch":
        args = [
            "ros2",
            "launch",
            "primeu_vla",
            "primeu_vla_runtime.launch.py",
            "backend:=gr00t",
            f"model_host:={server['host']}",
            f"model_port:={server['port']}",
            f"task_description:={runtime['task_description']}",
        ]
    elif operation in {"start", "stop"}:
        args = ["ros2", "service", "call", f"{runtime['node']}/{operation}", "std_srvs/srv/Trigger", "{}"]
    elif operation == "status":
        args = ["ros2", "topic", "echo", f"{runtime['node']}/status"]
    else:
        raise ValueError(f"Unsupported robot operation: {operation}")
    # Positional arguments, never interpolate YAML/task text into shell source.
    script = "set -e; " + 'source "$1"; shift; ' * len(setups) + 'exec "$@"'
    return ["bash", "-c", script, "primeu-client", *map(str, setups), *args]


def run_control(command, timeout):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            "ROS service timed out; result is unknown and was not retried. Use stop if start may have run."
        ) from exc
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode or not re.search(r"\bsuccess\s*[=:]\s*(True|true)\b", result.stdout):
        raise RuntimeError("ROS Trigger did not report success")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["check", "launch", "start", "stop", "status"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dry-run", action="store_true", help="Print only; no network, ROS, or hardware access")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.dry_run:
        command = None if args.operation == "check" else ros_command(cfg, args.operation, check_paths=False)
        print(
            json.dumps(
                {
                    "operation": args.operation,
                    "server": cfg["server"],
                    "command": shlex.join(command) if command else None,
                    "dry_run": True,
                    "robot_motion": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.operation == "check":
        print(json.dumps(check_server(cfg), ensure_ascii=False, indent=2))
        return
    command = ros_command(cfg, args.operation)
    # stop/status must work even if the policy server is disconnected.
    if args.operation in {"launch", "start"}:
        print(json.dumps(check_server(cfg), ensure_ascii=False, indent=2), flush=True)
    if args.operation == "start":
        # No automatic retry: resetting a live session changes its posture anchor.
        # Readiness/basis checks and reset must all finish before enabling the loop.
        reset_server(cfg)
        run_control(command, cfg["runtime"]["service_timeout_s"])
    elif args.operation == "stop":
        # Stopping the robot loop takes priority over any model/network check.
        run_control(command, cfg["runtime"]["service_timeout_s"])
        try:
            reset_server(cfg)
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"PRIMEU runtime stopped; policy reset unavailable: {exc}. Next start requires a successful reset.", file=sys.stderr)
    else:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"PRIMEU client: {exc}", file=sys.stderr)
        sys.exit(1)
