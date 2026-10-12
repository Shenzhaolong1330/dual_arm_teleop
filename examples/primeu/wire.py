"""PRIMEU GR00T-compatible ZMQ wire helpers; no model or robot dependencies."""

from collections.abc import Mapping

import msgpack
import numpy as np

JOINT_GROUPS = {
    "left_arm": (0, 7), "left_hand": (7, 27),
    "right_arm": (27, 34), "right_hand": (34, 54),
    "head": (54, 57), "waist": (57, 60),
}
CAMERAS = ("head", "left_wrist", "right_wrist")
PROTOCOL = "primeu_lerobot_v1"


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError("Only numeric arrays are allowed")
        return {b"nd": True, b"type": value.dtype.str, b"kind": b"",
                b"shape": value.shape, b"data": value.tobytes()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _decode(value):
    if b"nd" not in value:
        return value
    dtype = np.dtype(value[b"type"])
    if dtype.kind not in "biuf":
        raise ValueError("Only numeric arrays are allowed")
    if value[b"nd"]:
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"])
    return dtype.type(value[b"data"])


def pack(value):
    return msgpack.packb(value, default=_encode)


def unpack(value):
    return msgpack.unpackb(value, raw=False, object_hook=_decode)


def modality_config(horizon):
    groups = list(JOINT_GROUPS)
    configs = {
        "video": {"delta_indices": [0], "modality_keys": list(CAMERAS)},
        "state": {"delta_indices": [0], "modality_keys": groups},
        "action": {"delta_indices": list(range(horizon)), "modality_keys": groups,
                   "action_configs": [{"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT"}
                                      for _ in groups]},
        "language": {"delta_indices": [0], "modality_keys": ["annotation.human.task_description"]},
    }
    return {key: {"__ModalityConfig__": True, "as_json": value} for key, value in configs.items()}


def from_runtime(observation):
    """Validate one raw RGB/physical-state observation before touching model state."""
    if not isinstance(observation, Mapping):
        raise ValueError("observation must be a mapping")
    parts = []
    for group, (start, end) in JOINT_GROUPS.items():
        raw = np.asarray(observation["state"][group])
        if raw.dtype.kind not in "fiu":
            raise ValueError(f"state.{group} must be numeric")
        value = raw.astype(np.float32)
        if value.shape != (1, 1, end - start) or not np.isfinite(value).all():
            raise ValueError(f"state.{group}: expected finite [1,1,{end - start}]")
        parts.append(value[0, 0])
    language = observation["language"]["annotation.human.task_description"]
    if (not isinstance(language, (list, tuple)) or len(language) != 1
            or not isinstance(language[0], (list, tuple)) or len(language[0]) != 1
            or not isinstance(language[0][0], str)):
        raise ValueError("Language must contain one string with batch/time dimensions [1,1]")
    result = {"observation.state": np.concatenate(parts), "task": language[0][0]}
    for camera in CAMERAS:
        value = np.asarray(observation["video"][camera])
        if (value.ndim != 5 or value.shape[:2] != (1, 1) or value.shape[-1] != 3
                or min(value.shape[2:4]) <= 0 or value.dtype != np.uint8):
            raise ValueError(f"video.{camera}: expected uint8 [1,1,H,W,3] RGB")
        result[f"observation.images.{camera}"] = value[0, 0]
    return result


def to_runtime(sample):
    """Construct a request from an unnormalized sample with uint8 HWC images."""
    state = np.asarray(sample["observation.state"], dtype=np.float32)
    if state.shape != (60,):
        raise ValueError("Replay state must have shape [60]")
    observation = {
        "state": {key: state[a:b][None, None] for key, (a, b) in JOINT_GROUPS.items()},
        "video": {key: np.asarray(sample[f"observation.images.{key}"])[None, None] for key in CAMERAS},
        "language": {"annotation.human.task_description": [[sample["task"]]]},
    }
    from_runtime(observation)
    return observation


def rpc(host, port, endpoint, data=None, timeout_ms=30000):
    """A timeout destroys this socket; an old response can never become a new action."""
    import zmq

    try:
        with zmq.Context() as context:
            with context.socket(zmq.REQ) as socket:
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
                socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
                socket.connect(f"tcp://{host}:{int(port)}")
                socket.send(pack({"endpoint": endpoint, "data": data or {}}))
                result = unpack(socket.recv())
    except zmq.Again as exc:
        raise TimeoutError(f"Policy server timed out during {endpoint}; no automatic retry") from exc
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])
    return result
