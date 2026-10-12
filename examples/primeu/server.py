"""YAML-driven, robot-free LeRobot server for the existing PRIMEU runtime."""

import argparse
from collections.abc import Mapping
from pathlib import Path
import time

import numpy as np
import yaml

from .wire import JOINT_GROUPS, from_runtime, modality_config, pack, unpack


class PrimeuPolicyAdapter:
    def __init__(self, policy):
        self.policy = policy
        self.contract, self.horizon = policy.contract, policy.horizon

    def dispatch(self, request):
        if not isinstance(request, Mapping):
            raise ValueError("Request must be a mapping")
        endpoint = request.get("endpoint")
        data = request.get("data") or {}
        if not isinstance(data, Mapping):
            raise ValueError("Request data must be a mapping")
        if endpoint == "ping":
            return {"status": "ok", "message": "PRIMEU LeRobot policy server", "contract": self.contract}
        if endpoint == "get_modality_config":
            return modality_config(self.horizon)
        if endpoint == "reset":
            self.policy.reset()
            return {"status": "ok", "robot_motion": False}
        if endpoint != "get_action":
            raise ValueError(f"Unsupported endpoint: {endpoint}")
        observation = from_runtime(data["observation"])
        options = data.get("options") or {}
        if not isinstance(options, Mapping) or set(options) - {"seed"}:
            raise ValueError("Only the optional replay seed is supported in options")
        seed = options.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32):
            raise ValueError("seed must be an integer in [0, 2**32)")
        start = time.monotonic()
        actions = np.asarray(self.policy.infer(observation, seed=seed), dtype=np.float32)
        if actions.shape != (self.horizon, 60) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid policy output {actions.shape}")
        grouped = {key: actions[None, :, a:b] for key, (a, b) in JOINT_GROUPS.items()}
        return grouped, {
            "inference_ms": (time.monotonic() - start) * 1000,
            "action_representation": "absolute_joint_target", "action_horizon": self.horizon,
        }


def serve(adapter, host="127.0.0.1", port=5559, *, max_requests=None):
    """One robot session per service; reset before starting a new run."""
    import zmq

    with zmq.Context() as context:
        with context.socket(zmq.REP) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(f"tcp://{host}:{int(port)}")
            print(f"PRIMEU LeRobot ready on {host}:{port}", flush=True)
            count = 0
            while max_requests is None or count < max_requests:
                message = socket.recv()
                try:
                    response = adapter.dispatch(unpack(message))
                except Exception as exc:
                    response = {"error": f"{type(exc).__name__}: {exc}"}
                socket.send(pack(response))
                count += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve(strict=True)
    config = yaml.safe_load(config_path.read_text())
    checkpoint = Path(config["checkpoint"]).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = config_path.parent / checkpoint
    from .policy import NativePrimeuPolicy

    policy = NativePrimeuPolicy(checkpoint, device=config.get("device", "cuda"), seed=config.get("seed"))
    serve(PrimeuPolicyAdapter(policy), config.get("host", "127.0.0.1"), config.get("port", 5559))


if __name__ == "__main__":
    main()
