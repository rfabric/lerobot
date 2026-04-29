#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the rFabric remote teleoperator plugin."""

import socket
import struct
import time
from pathlib import Path

import pytest

from lerobot.utils.import_utils import is_package_available

if not is_package_available("cbor2"):
    pytest.skip("cbor2 not available; install lerobot[rfabric] extra", allow_module_level=True)

from lerobot.teleoperators.rfabric_remote.action_mapping import (
    EE_POSE_DELTA_FIELDS,
    MappingState,
    hold_ee_pose_delta,
    hold_joint_action,
    map_payload_to_ee_pose_delta,
    map_payload_to_joint_action,
)
from lerobot.teleoperators.rfabric_remote.configuration_rfabric_remote import (
    RFabricRemoteTeleopConfig,
)
from lerobot.teleoperators.rfabric_remote.teleop_rfabric_remote import (
    ACTION_MODE_EE_POSE_DELTA,
    RFabricRemoteTeleop,
)
from lerobot.teleoperators.rfabric_remote.wire import (
    CONTROL_KIND,
    LENGTH_PREFIX_FORMAT,
    MAX_FRAME_BYTES,
    MODE_KIND,
    Frame,
    FrameError,
    read_frame,
    write_frame,
)


@pytest.fixture(autouse=True)
def isolate_calibration_dir(tmp_path, monkeypatch):
    """Avoid touching the user's calibration directory in tests."""

    monkeypatch.setattr(
        "lerobot.teleoperators.teleoperator.HF_LEROBOT_CALIBRATION",
        tmp_path,
    )
    monkeypatch.setattr(
        "lerobot.utils.constants.HF_LEROBOT_CALIBRATION",
        tmp_path,
    )
    return tmp_path


def _make_socket_path(tmp_path: Path) -> Path:
    # AF_UNIX path length is constrained on macOS / Linux (~104 bytes);
    # tmp_path can blow that. Trim by hashing the dir name.
    base = tmp_path / "control.sock"
    if len(str(base)) < 100:
        return base
    return Path("/tmp") / f"rfabric-test-{abs(hash(tmp_path)) & 0xFFFF:04x}.sock"


def _connect_client(socket_path: Path) -> socket.socket:
    """Connect to the teleop's UDS, retrying briefly while the
    listener wakes up. ``accept`` is racy on AF_UNIX so the first
    couple of probes may hit ``FileNotFoundError``.
    """

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(2.0)
    for _ in range(50):
        try:
            client.connect(str(socket_path))
            break
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.02)
    else:  # pragma: no cover - infrastructure failure
        pytest.fail("teleop never bound the UDS within 1s")
    return client


def _wait_for_action(teleop: RFabricRemoteTeleop, predicate, timeout_s: float = 1.0) -> dict:
    deadline = time.monotonic() + timeout_s
    action: dict = {}
    while time.monotonic() < deadline:
        action = teleop.get_action()
        if predicate(action):
            return action
        time.sleep(0.01)
    return action


# ─── Wire format ────────────────────────────────────────────────────


def test_frame_round_trips_through_cbor():
    original = Frame(
        kind=CONTROL_KIND,
        seq=42,
        ts_ns=1_700_000_000_000_000_000,
        payload={"kind": "joint_deltas", "joints": {"shoulder_pan": 0.05}},
    )
    decoded = Frame.decode(original.encode())
    assert decoded == original


def test_frame_decode_rejects_invalid_kind():
    import cbor2

    bogus = cbor2.dumps({"kind": "unknown", "seq": 0, "tsNs": 0, "payload": None})
    with pytest.raises(FrameError):
        Frame.decode(bogus)


def test_read_frame_rejects_oversize_length_prefix(tmp_path):
    socket_path = _make_socket_path(tmp_path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(socket_path))
        server, _ = listener.accept()
        try:
            oversized = struct.pack(LENGTH_PREFIX_FORMAT, MAX_FRAME_BYTES + 1)
            client.sendall(oversized)
            client.shutdown(socket.SHUT_WR)
            with pytest.raises(FrameError, match="out of range"):
                read_frame(server)
        finally:
            client.close()
            server.close()
    finally:
        listener.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


# ─── Action mapping (joint-direct path) ─────────────────────────────


def test_joint_path_maps_twist_payload():
    state = MappingState.empty()
    action = map_payload_to_joint_action(
        {"kind": "twist", "linear": 0.5, "angular": -0.25},
        state,
        motion_scale={},
    )
    assert action == {"linear_velocity": 0.5, "angular_velocity": -0.25}


def test_joint_path_maps_end_effector_payload():
    state = MappingState.empty()
    action = map_payload_to_joint_action(
        {"kind": "end_effector", "deltaX": 0.01, "deltaY": 0.0, "deltaZ": -0.005, "gripper": 2},
        state,
        motion_scale={},
    )
    assert action == {"delta_x": 0.01, "delta_y": 0.0, "delta_z": -0.005, "gripper": 2.0}


def test_joint_path_replaces_state_on_joint_targets():
    state = MappingState.empty()
    state.joint_targets_by_arm[""] = {"shoulder_pan": 0.1}
    action = map_payload_to_joint_action(
        {"kind": "joint_targets", "joints": {"elbow_flex": 0.4, "shoulder_lift": -0.2}},
        state,
        motion_scale={},
    )
    assert action == {"elbow_flex": 0.4, "shoulder_lift": -0.2}
    assert state.joint_targets_by_arm[""] == {"elbow_flex": 0.4, "shoulder_lift": -0.2}


def test_joint_path_accumulates_and_scales_deltas():
    state = MappingState.empty()
    state.joint_targets_by_arm[""] = {"elbow_flex": 0.10}
    motion_scale = {"elbow_flex": 0.5}
    map_payload_to_joint_action(
        {"kind": "joint_deltas", "joints": {"elbow_flex": 0.04}},
        state,
        motion_scale=motion_scale,
    )
    map_payload_to_joint_action(
        {"kind": "joint_deltas", "joints": {"elbow_flex": 0.04}},
        state,
        motion_scale=motion_scale,
    )
    assert state.joint_targets_by_arm[""]["elbow_flex"] == pytest.approx(0.10 + 2 * (0.04 * 0.5))


def test_joint_path_unknown_payload_returns_empty():
    state = MappingState.empty()
    action = map_payload_to_joint_action({"kind": "totally_made_up"}, state, motion_scale={})
    assert action == {}


def test_joint_path_hold_returns_last_targets():
    state = MappingState.empty()
    state.joint_targets_by_arm[""] = {"elbow_flex": 0.42}
    assert hold_joint_action(state) == {"elbow_flex": 0.42}


def test_joint_path_home_returns_sentinel_marker():
    state = MappingState.empty()
    action = map_payload_to_joint_action({"kind": "home"}, state, motion_scale={})
    assert action == {"__rfabric_remote_home__": 1.0}


def test_joint_path_per_arm_state_is_isolated():
    state = MappingState.empty()
    map_payload_to_joint_action(
        {"kind": "joint_deltas", "joints": {"elbow_flex": 0.10}},
        state,
        motion_scale={},
        arm="left",
    )
    map_payload_to_joint_action(
        {"kind": "joint_deltas", "joints": {"elbow_flex": -0.20}},
        state,
        motion_scale={},
        arm="right",
    )
    assert state.joint_targets_by_arm["left"]["elbow_flex"] == pytest.approx(0.10)
    assert state.joint_targets_by_arm["right"]["elbow_flex"] == pytest.approx(-0.20)


# ─── Action mapping (EE-pose-delta path) ────────────────────────────


def test_ee_pose_delta_maps_end_effector_payload_unprefixed():
    action = map_payload_to_ee_pose_delta(
        {"kind": "end_effector", "deltaX": 0.5, "deltaY": -0.25, "deltaZ": 1.0, "gripper": 2}
    )
    assert action == {
        "enabled": True,
        "target_x": 0.5,
        "target_y": -0.25,
        "target_z": 1.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper_vel": -1.0,
    }


def test_ee_pose_delta_prefixes_keys_per_arm():
    action = map_payload_to_ee_pose_delta(
        {"kind": "end_effector", "deltaX": 0.1, "deltaY": 0.2, "deltaZ": 0.3, "gripper": 1},
        arm="right",
    )
    expected_keys = {f"right_{key}" for key in EE_POSE_DELTA_FIELDS}
    assert set(action.keys()) == expected_keys
    assert action["right_enabled"] is True
    assert action["right_target_x"] == 0.1
    assert action["right_gripper_vel"] == 0.0


@pytest.mark.parametrize(
    ("gripper_input", "expected_velocity"),
    [
        (0, 1.0),
        (1, 0.0),
        (2, -1.0),
        (None, 0.0),
        ("nonsense", 0.0),
    ],
)
def test_ee_pose_delta_gripper_state_translation(gripper_input, expected_velocity):
    action = map_payload_to_ee_pose_delta(
        {"kind": "end_effector", "deltaX": 0.0, "deltaY": 0.0, "deltaZ": 0.0, "gripper": gripper_input}
    )
    assert action["gripper_vel"] == expected_velocity


def test_ee_pose_delta_unknown_payload_returns_disabled_hold():
    action = map_payload_to_ee_pose_delta({"kind": "stop"}, arm="left")
    assert action == hold_ee_pose_delta(arm="left")
    assert action["left_enabled"] is False


def test_ee_pose_delta_hold_uses_disabled_pattern():
    action = hold_ee_pose_delta()
    assert action["enabled"] is False
    for axis in ("target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz"):
        assert action[axis] == 0.0
    assert action["gripper_vel"] == 0.0


# ─── Teleoperator end-to-end ────────────────────────────────────────


def _make_teleop(
    tmp_path,
    *,
    arms: list[str] | None = None,
    action_mode: str = "joint_direct",
) -> RFabricRemoteTeleop:
    socket_path = _make_socket_path(tmp_path)
    config = RFabricRemoteTeleopConfig(
        id="test-remote",
        socket_path=socket_path,
        watchdog_ms=50,
        accept_min_backoff_ms=10,
        accept_max_backoff_ms=50,
        state_publish_hz=200.0,
        arms=list(arms) if arms is not None else [],
        action_mode=action_mode,
    )
    return RFabricRemoteTeleop(config)


def test_get_action_returns_held_position_when_no_frame_received(tmp_path):
    teleop = _make_teleop(tmp_path)
    teleop.connect()
    try:
        assert teleop.get_action() == {}
    finally:
        teleop.disconnect()


def test_get_action_decodes_inbound_control_frame(tmp_path):
    teleop = _make_teleop(tmp_path)
    teleop.connect()
    try:
        client = _connect_client(teleop.config.socket_path)

        capability = read_frame(client)
        assert capability is not None
        assert capability.kind == MODE_KIND
        assert capability.payload["kind"] == "capabilities"

        write_frame(
            client,
            Frame(
                kind=CONTROL_KIND,
                seq=1,
                ts_ns=1_700_000_000_000_000_000,
                payload={"kind": "twist", "linear": 0.5, "angular": -0.5},
            ),
        )

        action = _wait_for_action(teleop, predicate=lambda a: bool(a))
        assert action == {"linear_velocity": 0.5, "angular_velocity": -0.5}
        client.close()
    finally:
        teleop.disconnect()


def test_get_action_falls_back_to_hold_after_watchdog(tmp_path):
    teleop = _make_teleop(tmp_path)
    teleop.connect()
    try:
        client = _connect_client(teleop.config.socket_path)
        read_frame(client)  # capabilities

        write_frame(
            client,
            Frame(
                kind=CONTROL_KIND,
                seq=1,
                ts_ns=0,
                payload={"kind": "joint_targets", "joints": {"elbow_flex": 0.3}},
            ),
        )
        _wait_for_action(teleop, predicate=lambda a: bool(a))
        time.sleep(teleop.config.watchdog_ms / 1000.0 * 3)
        # Watchdog returns the held joint targets, not zeros, so the
        # arm stays put rather than slamming back to a home pose.
        assert teleop.get_action() == {"elbow_flex": 0.3}
        client.close()
    finally:
        teleop.disconnect()


def test_get_action_emits_per_arm_ee_pose_delta_keys(tmp_path):
    teleop = _make_teleop(tmp_path, arms=["left", "right"], action_mode=ACTION_MODE_EE_POSE_DELTA)
    teleop.connect()
    try:
        client = _connect_client(teleop.config.socket_path)
        read_frame(client)  # capabilities

        write_frame(
            client,
            Frame(
                kind=CONTROL_KIND,
                seq=1,
                ts_ns=0,
                payload={
                    "kind": "end_effector",
                    "arm": "left",
                    "deltaX": 0.4,
                    "deltaY": 0.0,
                    "deltaZ": 0.0,
                    "gripper": 2,
                },
            ),
        )
        write_frame(
            client,
            Frame(
                kind=CONTROL_KIND,
                seq=2,
                ts_ns=0,
                payload={
                    "kind": "end_effector",
                    "arm": "right",
                    "deltaX": -0.1,
                    "deltaY": 0.2,
                    "deltaZ": 0.0,
                    "gripper": 0,
                },
            ),
        )

        action = _wait_for_action(
            teleop,
            predicate=lambda a: a.get("left_enabled") is True and a.get("right_enabled") is True,
        )
        assert action["left_target_x"] == 0.4
        assert action["left_gripper_vel"] == -1.0
        assert action["right_target_y"] == 0.2
        assert action["right_gripper_vel"] == 1.0
        client.close()
    finally:
        teleop.disconnect()


def test_per_arm_watchdog_disables_only_stale_arm(tmp_path):
    teleop = _make_teleop(tmp_path, arms=["left", "right"], action_mode=ACTION_MODE_EE_POSE_DELTA)
    teleop.connect()
    try:
        client = _connect_client(teleop.config.socket_path)
        read_frame(client)  # capabilities

        write_frame(
            client,
            Frame(
                kind=CONTROL_KIND,
                seq=1,
                ts_ns=0,
                payload={
                    "kind": "end_effector",
                    "arm": "left",
                    "deltaX": 0.4,
                    "deltaY": 0.0,
                    "deltaZ": 0.0,
                    "gripper": 1,
                },
            ),
        )
        # Wait for the watchdog window to elapse so the left frame
        # also goes stale, then immediately ship a fresh right frame.
        # `get_action` should now report the right arm enabled and the
        # left arm holding (disabled).
        time.sleep(teleop.config.watchdog_ms / 1000.0 * 3)
        write_frame(
            client,
            Frame(
                kind=CONTROL_KIND,
                seq=2,
                ts_ns=0,
                payload={
                    "kind": "end_effector",
                    "arm": "right",
                    "deltaX": 0.2,
                    "deltaY": 0.0,
                    "deltaZ": 0.0,
                    "gripper": 1,
                },
            ),
        )

        action = _wait_for_action(
            teleop,
            predicate=lambda a: a.get("right_enabled") is True,
        )
        assert action["left_enabled"] is False
        assert action["right_enabled"] is True
        assert action["right_target_x"] == 0.2
        client.close()
    finally:
        teleop.disconnect()


def test_get_teleop_events_clears_one_shot_flags(tmp_path):
    teleop = _make_teleop(tmp_path)
    teleop.connect()
    try:
        client = _connect_client(teleop.config.socket_path)
        read_frame(client)  # capabilities

        write_frame(
            client,
            Frame(kind=MODE_KIND, seq=1, ts_ns=0, payload={"kind": "success"}),
        )
        time.sleep(0.1)
        events = teleop.get_teleop_events()
        from lerobot.teleoperators.utils import TeleopEvents

        assert events[TeleopEvents.SUCCESS] is True
        assert events[TeleopEvents.TERMINATE_EPISODE] is True

        # Subsequent reads must clear the one-shot flags.
        events = teleop.get_teleop_events()
        assert events[TeleopEvents.SUCCESS] is False
        assert events[TeleopEvents.TERMINATE_EPISODE] is False
        client.close()
    finally:
        teleop.disconnect()
