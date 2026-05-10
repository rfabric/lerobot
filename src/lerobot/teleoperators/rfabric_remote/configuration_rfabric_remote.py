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
"""Configuration for the rFabric remote teleoperator.

This teleoperator binds a Unix-domain socket and accepts a connection
from the local ``rfabric-remote`` agent. The agent forwards every
operator -> robot CBOR ``Frame`` from the WebRTC data channel onto the
socket; this teleoperator decodes them and returns lerobot-shaped
actions to the standard ``lerobot-teleoperate`` loop.

Wire contract is documented in ``REAL_TIME_CONTROL_BRIDGE.md`` at the
rFabric platform repository root.
"""

from dataclasses import dataclass, field
from pathlib import Path

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rfabric_remote")
@dataclass
class RFabricRemoteTeleopConfig(TeleoperatorConfig):
    """Configuration for ``RFabricRemoteTeleop``.

    Attributes:
        socket_path: Filesystem path of the Unix-domain socket the
            teleoperator binds. The rFabric agent connects to this same
            path; both ends agree on
            ``/run/rfabric/control.sock`` under systemd. Local-dev runs
            without systemd typically point both ends at
            ``/tmp/rfabric/control.sock`` instead.
        watchdog_ms: When no ``control`` frame has been seen for longer
            than this window, ``get_action`` returns a hold-position
            action so the follower freezes its current pose. The agent
            emits a wire-level heartbeat every 100 ms by default, so a
            200 ms watchdog covers a single missed heartbeat without
            tripping.
        accept_min_backoff_ms: Initial backoff between accept retries.
            Doubles up to ``accept_max_backoff_ms`` on consecutive
            failures.
        accept_max_backoff_ms: Upper bound on the accept backoff.
        state_publish_hz: Rate at which the most recent observation is
            mirrored back to the agent as a ``state`` frame. The
            operator UI reads this to render the live joint / pose
            overlay.
        socket_mode: Octal permissions applied to the socket file after
            bind. Default ``0o600`` matches the trust boundary the
            ``rfabric`` systemd user already establishes.
        accepts: ``payload.kind`` values this follower accepts. Sent to
            the agent as a ``mode`` capability frame on each new
            connection so the operator UI can disable inapplicable
            controls.
        arms: Arm tags advertised in the same capability frame, e.g.
            ``["left", "right"]`` for a bimanual follower. Empty for
            mobile / wheeled platforms.
        action_mode: Output shape ``get_action`` returns:

            * ``"ee_pose_delta"`` — emit
              ``{prefix}enabled / target_x / target_y / target_z /
              target_wx / target_wy / target_wz / gripper_vel`` keys.
              Designed to feed straight into the SO IK pipeline
              (``EEReferenceAndDelta`` ->
              ``GripperVelocityToJoint`` ->
              ``InverseKinematicsEEToJoints``). This is the mode the
              ``bimanual_so101`` entry point uses.
            * ``"joint_direct"`` — emit raw joint targets / EE deltas
              with no IK layer. Suitable for followers that take joint
              targets directly.
        motion_scale: Per-joint normalised -> radians scaling applied
            when decoding ``joint_deltas`` payloads in the
            ``joint_direct`` action mode. Ignored in the
            ``ee_pose_delta`` mode.
    """

    socket_path: Path = Path("/run/rfabric/control.sock")
    watchdog_ms: int = 200
    accept_min_backoff_ms: int = 50
    accept_max_backoff_ms: int = 2_000
    state_publish_hz: float = 30.0
    socket_mode: int = 0o600
    accepts: list[str] = field(
        default_factory=lambda: [
            "twist",
            "stop",
            "home",
            "joint_deltas",
            "joint_targets",
            "end_effector",
        ]
    )
    arms: list[str] = field(default_factory=list)
    action_mode: str = "joint_direct"
    motion_scale: dict[str, float] = field(default_factory=dict)
