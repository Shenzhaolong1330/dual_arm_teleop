"""Robot-free checks of the standalone PRIMEU launcher and wire handshake."""

import importlib.util
import json
from pathlib import Path
import queue
import subprocess
import threading

import pytest
import yaml


_spec = importlib.util.spec_from_file_location("primeu_standalone_client", Path(__file__).parents[1] / "client.py")
client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client)


@pytest.fixture
def cfg():
    return {
        "server": {
            "host": "127.0.0.1", "port": 5559, "timeout_s": 1,
            "expected_horizon": 60, "expected_model_family": "act",
            "expected_model_action_dim": 60, "expected_basis_sha256": {},
            "expected_image_size": [256, 256],
        },
        "runtime": {
            "workspace": "/missing/robot", "setup_file": "install/setup.bash",
            "ros_setup_file": None, "node": "/primeu_vla_runtime",
            "task_description": "Pick up the pipette with your right hand.",
            "service_timeout_s": 1,
        },
    }


def responses(cfg):
    server = cfg["server"]
    contract = {
        "protocol": "primeu_lerobot_v1", "model_family": server["expected_model_family"],
        "state_dim": 60, "action_dim": 60, "model_action_dim": server["expected_model_action_dim"],
        "joint_groups": client.JOINT_DIMS, "joint_order": list(client.JOINT_DIMS),
        "camera_keys": client.CAMERAS, "action_representation": "absolute_joint_target",
        "action_horizon": server["expected_horizon"], "image_size": [256, 256],
        "basis_sha256": server["expected_basis_sha256"],
    }
    modalities = {}
    for kind, keys in {
        "video": client.CAMERAS, "state": list(client.JOINT_DIMS),
        "action": list(client.JOINT_DIMS), "language": ["annotation.human.task_description"],
    }.items():
        value = {"modality_keys": keys, "delta_indices": list(range(server["expected_horizon"])) if kind == "action" else [0]}
        if kind == "action":
            value["action_configs"] = [{"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT"}] * 6
        modalities[kind] = {"__ModalityConfig__": True, "as_json": value}
    return {"ping": {"status": "ok", "contract": contract}, "get_modality_config": modalities,
            "reset": {"status": "ok", "robot_motion": False}}


@pytest.mark.parametrize("family,horizon,dim", [("act", 60, 60), ("diffusion", 7, 27), ("act", 20, 15), ("act", 20, 12), ("act", 20, 33)])
def test_handshake_checks_reduced_model_and_basis(monkeypatch, cfg, family, horizon, dim):
    cfg["server"].update(expected_model_family=family, expected_horizon=horizon, expected_model_action_dim=dim,
                         expected_basis_sha256={} if dim == 60 else {"basis.json": "a" * 64})
    replies = responses(cfg)
    endpoints = []

    def rpc(server, endpoint):
        endpoints.append(endpoint)
        return replies[endpoint]

    monkeypatch.setattr(client, "rpc", rpc)
    result = client.check_server(cfg)
    assert result["contract"]["model_action_dim"] == dim
    assert result["robot_motion"] is False
    assert endpoints == ["ping", "get_modality_config"]


@pytest.mark.parametrize("key,value", [
    ("protocol", "primeu_openpi_v1"), ("model_family", "diffusion"),
    ("camera_keys", ["left_wrist", "head", "right_wrist"]), ("action_dim", 27),
    ("action_representation", "relative_joint_target"), ("action_horizon", 40),
    ("basis_sha256", {"basis.json": "a" * 64}),
])
def test_wrong_policy_rejected_before_modality_request(monkeypatch, cfg, key, value):
    replies = responses(cfg)
    replies["ping"]["contract"][key] = value
    endpoints = []

    def rpc(server, endpoint):
        endpoints.append(endpoint)
        return replies[endpoint]

    monkeypatch.setattr(client, "rpc", rpc)
    with pytest.raises(ValueError, match="expected LeRobot"):
        client.check_server(cfg)
    assert endpoints == ["ping"]


def test_wrong_modality_action_semantics_rejected(monkeypatch, cfg):
    replies = responses(cfg)
    replies["get_modality_config"]["action"]["as_json"]["action_configs"] = []
    monkeypatch.setattr(client, "rpc", lambda s, endpoint: replies[endpoint])
    with pytest.raises(ValueError, match="absolute NON_EEF"):
        client.check_server(cfg)


def install_fake_control(monkeypatch, cfg, events):
    monkeypatch.setattr(client, "load_config", lambda p: cfg)
    monkeypatch.setattr(client, "ros_command", lambda c, op: [op])
    monkeypatch.setattr(client, "check_server", lambda c: events.append("check") or {})
    monkeypatch.setattr(client, "reset_server", lambda c: events.append("reset") or {})
    monkeypatch.setattr(client, "run_control", lambda command, timeout: events.append(command[0]))


def test_start_checks_and_resets_before_ros_start(monkeypatch, cfg):
    events = []
    install_fake_control(monkeypatch, cfg, events)
    client.main(["start"])
    assert events == ["check", "reset", "start"]


def test_failed_reset_never_starts_or_retries(monkeypatch, cfg):
    events = []
    install_fake_control(monkeypatch, cfg, events)

    def reset(c):
        events.append("reset")
        raise TimeoutError("offline")

    monkeypatch.setattr(client, "reset_server", reset)
    with pytest.raises(TimeoutError):
        client.main(["start"])
    assert events == ["check", "reset"]


def test_stop_works_without_model_server(monkeypatch, cfg, capsys):
    events = []
    install_fake_control(monkeypatch, cfg, events)

    def reset(c):
        events.append("reset")
        raise TimeoutError("offline")

    monkeypatch.setattr(client, "reset_server", reset)
    client.main(["stop"])
    assert events == ["stop", "reset"]
    assert "runtime stopped" in capsys.readouterr().err


def test_failed_ros_stop_does_not_change_running_posture_anchor(monkeypatch, cfg):
    events = []
    install_fake_control(monkeypatch, cfg, events)

    def stop(command, timeout):
        events.append(command[0])
        raise TimeoutError("ROS unavailable")

    monkeypatch.setattr(client, "run_control", stop)
    with pytest.raises(TimeoutError):
        client.main(["stop"])
    assert events == ["stop"]


def test_launch_dry_run_needs_neither_ros_nor_network(monkeypatch, tmp_path, cfg, capsys):
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(client, "rpc", lambda *a: pytest.fail("dry-run used network"))
    monkeypatch.setattr(client.subprocess, "run", lambda *a, **k: pytest.fail("dry-run started a process"))
    client.main(["launch", "--config", str(path), "--dry-run"])
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True and report["robot_motion"] is False
    assert "backend:=gr00t" in report["command"]


def test_task_text_never_enters_shell_program(cfg):
    cfg["runtime"]["task_description"] = "$(touch /tmp/never-created); quoted ' task"
    command = client.ros_command(cfg, "launch", check_paths=False)
    assert "touch" not in command[2]
    assert command[-1] == "task_description:=" + cfg["runtime"]["task_description"]


def test_ros_timeout_has_no_automatic_retry(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(client.subprocess, "run", run)
    with pytest.raises(TimeoutError, match="not retried"):
        client.run_control(["fake-start"], 0.1)
    assert len(calls) == 1


@pytest.mark.parametrize("success", [True, False])
def test_ros_trigger_result_must_explicitly_report_success(monkeypatch, success):
    monkeypatch.setattr(client.subprocess, "run", lambda *a, **kw:
                        subprocess.CompletedProcess(a[0], 0, f"Response(success={success}, message='')", ""))
    if success:
        client.run_control(["fake-start"], 1)
    else:
        with pytest.raises(RuntimeError, match="did not report success"):
            client.run_control(["fake-start"], 1)


@pytest.mark.parametrize("dim,basis", [(60, {"basis.json": "a" * 64}), (27, {}), (12, {"basis.json": "unknown"})])
def test_config_requires_explicit_matching_basis_hashes(tmp_path, cfg, dim, basis):
    cfg["server"].update(expected_model_action_dim=dim, expected_basis_sha256=basis)
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="basis"):
        client.load_config(path)


@pytest.mark.parametrize("bad_timeout", [float("nan"), float("inf"), -1, True])
def test_invalid_timeouts_rejected(tmp_path, cfg, bad_timeout):
    cfg["server"]["timeout_s"] = bad_timeout
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Timeouts"):
        client.load_config(path)


def test_real_zmq_metadata_and_reset_without_robot(cfg):
    """Exercise the actual MessagePack bytes against a fake REP server."""
    msgpack = pytest.importorskip("msgpack")
    zmq = pytest.importorskip("zmq")
    replies = responses(cfg)
    ports, errors, endpoints = queue.Queue(), [], []

    def serve():
        try:
            with zmq.Context() as context:
                with context.socket(zmq.REP) as socket:
                    socket.setsockopt(zmq.RCVTIMEO, 3000)
                    socket.setsockopt(zmq.LINGER, 0)
                    ports.put(socket.bind_to_random_port("tcp://127.0.0.1"))
                    for _ in range(3):
                        request = msgpack.unpackb(socket.recv(), raw=False)
                        assert set(request) == {"endpoint"}
                        endpoint = request["endpoint"]
                        endpoints.append(endpoint)
                        socket.send(msgpack.packb(replies[endpoint], use_bin_type=True))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    cfg["server"]["port"] = ports.get(timeout=2)
    client.check_server(cfg)
    assert client.reset_server(cfg)["robot_motion"] is False
    thread.join(timeout=4)
    assert not thread.is_alive()
    assert not errors
    assert endpoints == ["ping", "get_modality_config", "reset"]


def test_client_accepts_actual_server_adapter(monkeypatch, cfg):
    from examples.primeu.server import PrimeuPolicyAdapter

    class FakePolicy:
        contract = responses(cfg)["ping"]["contract"]
        horizon = cfg["server"]["expected_horizon"]
        reset_count = 0

        def reset(self):
            self.reset_count += 1

    policy = FakePolicy()
    adapter = PrimeuPolicyAdapter(policy)
    monkeypatch.setattr(client, "rpc", lambda server, endpoint: adapter.dispatch({"endpoint": endpoint}))
    client.check_server(cfg)
    client.reset_server(cfg)
    assert policy.reset_count == 1
