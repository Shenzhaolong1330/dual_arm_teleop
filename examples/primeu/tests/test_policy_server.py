"""Protocol integration tests intentionally use no torch, ROS or robot SDK."""

import socket
import threading

import numpy as np
import pytest

from examples.primeu.server import PrimeuPolicyAdapter, serve
from examples.primeu.wire import CAMERAS, JOINT_GROUPS, PROTOCOL, pack, rpc, to_runtime, unpack


def observation(offset=0):
    return to_runtime({
        "observation.state": np.arange(60, dtype=np.float32) + offset,
        "task": "Pick up the pipette with your right hand.",
        **{f"observation.images.{key}": np.full((5, 7, 3), i, np.uint8) for i, key in enumerate(CAMERAS)},
    })


class FakePolicy:
    horizon = 3
    contract = {"protocol": PROTOCOL, "action_horizon": 3}

    def __init__(self):
        self.seen, self.reset_count = [], 0

    def reset(self):
        self.reset_count += 1

    def infer(self, obs, seed=None):
        self.seen.append((obs, seed))
        return np.repeat(obs["observation.state"][None], self.horizon, axis=0)


def test_wire_numeric_roundtrip_preserves_dtype_shape_and_rgb():
    data = observation()
    result = unpack(pack(data))
    for kind in ("state", "video"):
        for key in data[kind]:
            assert result[kind][key].dtype == data[kind][key].dtype
            np.testing.assert_array_equal(result[kind][key], data[kind][key])
    assert result["language"] == data["language"]
    with pytest.raises(ValueError, match="numeric"):
        pack(np.array(["unsafe-object"], dtype=object))


@pytest.mark.parametrize("mutation", ["float_rgb", "missing_camera", "bad_state", "nan_state", "bad_language"])
def test_invalid_observation_does_not_touch_policy(mutation):
    policy = FakePolicy()
    data = observation()
    if mutation == "float_rgb":
        data["video"]["head"] = data["video"]["head"].astype(np.float32)
    elif mutation == "missing_camera":
        data["video"].pop("right_wrist")
    elif mutation == "bad_state":
        data["state"]["left_arm"] = np.zeros((7,), np.float32)
    elif mutation == "nan_state":
        data["state"]["right_arm"][0, 0, 0] = np.nan
    else:
        data["language"]["annotation.human.task_description"] = "not-batched"
    with pytest.raises((ValueError, KeyError)):
        PrimeuPolicyAdapter(policy).dispatch({"endpoint": "get_action", "data": {"observation": data}})
    assert policy.seen == []


def test_adapter_uses_fresh_observations_and_returns_absolute_groups():
    policy = FakePolicy()
    adapter = PrimeuPolicyAdapter(policy)
    for offset in (0, 100):
        response, metadata = adapter.dispatch({"endpoint": "get_action", "data": {"observation": observation(offset), "options": {"seed": 123}}})
        for key, (start, end) in JOINT_GROUPS.items():
            assert response[key].shape == (1, 3, end - start)
            np.testing.assert_array_equal(response[key][0, 0], np.arange(start, end) + offset)
        assert metadata["action_representation"] == "absolute_joint_target"
    assert len(policy.seen) == 2 and policy.seen[0][1] == 123
    assert adapter.dispatch({"endpoint": "reset"}) == {"status": "ok", "robot_motion": False}
    assert policy.reset_count == 1


def test_invalid_output_rejected_instead_of_reused():
    policy = FakePolicy()
    policy.infer = lambda *args, **kwargs: np.zeros((4, 60), np.float32)
    with pytest.raises(ValueError, match="Invalid policy output"):
        PrimeuPolicyAdapter(policy).dispatch({"endpoint": "get_action", "data": {"observation": observation()}})


def test_zmq_reports_error_and_serves_next_request():
    with socket.socket() as allocation:
        allocation.bind(("127.0.0.1", 0))
        port = allocation.getsockname()[1]
    policy = FakePolicy()
    worker = threading.Thread(target=serve, args=(PrimeuPolicyAdapter(policy), "127.0.0.1", port), kwargs={"max_requests": 4}, daemon=True)
    worker.start()
    try:
        with pytest.raises(RuntimeError, match="Unsupported endpoint"):
            rpc("127.0.0.1", port, "bad_endpoint", timeout_ms=3000)
        assert rpc("127.0.0.1", port, "ping")["contract"]["protocol"] == PROTOCOL
        result = rpc("127.0.0.1", port, "get_action", {"observation": observation()})
        assert result[0]["head"].shape == (1, 3, 3)
        assert rpc("127.0.0.1", port, "reset")["robot_motion"] is False
    finally:
        worker.join(timeout=3)
    assert not worker.is_alive()
    assert policy.reset_count == 1


def test_modality_describes_actual_valid_horizon():
    modality = PrimeuPolicyAdapter(FakePolicy()).dispatch({"endpoint": "get_modality_config"})
    action = modality["action"]["as_json"]
    assert action["delta_indices"] == [0, 1, 2]
    assert action["modality_keys"] == list(JOINT_GROUPS)
    assert all(item["rep"] == "ABSOLUTE" for item in action["action_configs"])


def test_rpc_timeout_is_explicit_and_does_not_retry():
    import zmq
    with zmq.Context() as context:
        with context.socket(zmq.REP) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            port = socket.bind_to_random_port("tcp://127.0.0.1")
            with pytest.raises(TimeoutError, match="no automatic retry"):
                rpc("127.0.0.1", port, "get_action", timeout_ms=30)
            # Exactly one request arrived; there was no application-level retry.
            assert unpack(socket.recv())["endpoint"] == "get_action"
