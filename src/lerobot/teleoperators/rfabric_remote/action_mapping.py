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
"""Convert operator-side control payloads into lerobot-shaped action dicts.

Three output shapes are supported, picked by the caller based on what
the follower expects:

* **Joint-direct shape** — ``map_payload_to_joint_action`` emits
  ``{joint_name: target_pos}`` (or accumulates deltas). Suitable for
  followers that take raw joint targets (e.g. lerobot leader-follower
  loops without IK).
* **Joint-velocity shape** — ``map_payload_to_joint_velocity`` emits
  ``{enabled, vel_<joint>: float}``. Each value is a normalised
  velocity in ``[-1, 1]`` that the downstream
  :class:`~lerobot.robots.so_follower.robot_kinematic_processor.JointVelocityServo`
  integrates into a target position. This is the path used by the
  SO-101 bimanual entry point for direct per-servo teleop.
* **EE-pose-delta shape** — ``map_payload_to_ee_pose_delta`` emits
  ``{enabled, target_x, target_y, target_z, target_wx, target_wy,
  target_wz, gripper_vel}``. Available for callers that want to
  compose a Cartesian IK pipeline on top.

All helpers support a per-arm prefix (``"left"`` / ``"right"``) so the
same code path serves single-arm and bimanual followers — the bimanual
teleoperate entry point splits the prefixed dict per arm before running
its servo step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.types import RobotAction

PAYLOAD_TWIST = "twist"
PAYLOAD_STOP = "stop"
PAYLOAD_HOME = "home"
PAYLOAD_JOINT_TARGETS = "joint_targets"
PAYLOAD_JOINT_DELTAS = "joint_deltas"
PAYLOAD_JOINT_VELOCITY = "joint_velocity"
PAYLOAD_END_EFFECTOR = "end_effector"

EE_POSE_DELTA_FIELDS = (
    "enabled",
    "target_x",
    "target_y",
    "target_z",
    "target_wx",
    "target_wy",
    "target_wz",
    "gripper_vel",
)

# Action-key prefix used by ``map_payload_to_joint_velocity`` for each
# joint's normalised velocity. Reading this prefix downstream is what
# couples the wire payload to the
# :class:`~lerobot.robots.so_follower.robot_kinematic_processor.JointVelocityServo`
# step.
JOINT_VELOCITY_KEY_PREFIX = "vel_"


@dataclass
class MappingState:
    """Per-arm accumulator for the joint-direct payload paths.

    The EE-pose-delta path is stateless on this side — the SO IK
    pipeline maintains its own reference pose / IK seed.

    ``joint_targets_by_arm[""]`` is the single-arm slot used when the
    follower is not bimanual.
    """

    joint_targets_by_arm: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> MappingState:
        return cls()

    def get(self, arm: str) -> dict[str, float]:
        return self.joint_targets_by_arm.setdefault(arm, {})


# ─────────────────────────────────────────────────────────────────────
# EE-pose-delta path (used by SO arms with the IK pipeline downstream)
# ─────────────────────────────────────────────────────────────────────


def map_payload_to_ee_pose_delta(payload: Any, arm: str = "") -> RobotAction:
    """Translate a control payload to ``EEReferenceAndDelta`` input keys.

    ``arm`` is the prefix applied to every output key, e.g. ``"left"``
    -> ``left_target_x`` / ``left_gripper_vel`` / ``left_enabled``.
    Empty ``arm`` produces un-prefixed keys for single-arm setups.

    Unknown / malformed payloads return a "disabled" frame so the
    upstream watchdog converges on a hold-position cue.
    """

    if not isinstance(payload, dict):
        return hold_ee_pose_delta(arm)
    kind = payload.get("kind")
    if kind == PAYLOAD_END_EFFECTOR:
        return _prefixed(
            arm,
            enabled=True,
            target_x=_coerce_float(payload.get("deltaX")),
            target_y=_coerce_float(payload.get("deltaY")),
            target_z=_coerce_float(payload.get("deltaZ")),
            target_wx=0.0,
            target_wy=0.0,
            target_wz=0.0,
            gripper_vel=_gripper_state_to_velocity(payload.get("gripper")),
        )
    # `stop`, `home`, `twist`, `joint_*` — none of these belong on the
    # EE-pose-delta path. The processor's hold-when-disabled semantics
    # cleanly map a `stop` / unknown frame to "freeze the last
    # commanded pose".
    return hold_ee_pose_delta(arm)


def hold_ee_pose_delta(arm: str = "") -> RobotAction:
    """Action returned when no recent control frame is available.

    ``EEReferenceAndDelta`` interprets ``enabled=False`` as
    "keep sending the last commanded pose"; the values of the target
    fields don't matter when disabled but must be present.
    """

    return _prefixed(
        arm,
        enabled=False,
        target_x=0.0,
        target_y=0.0,
        target_z=0.0,
        target_wx=0.0,
        target_wy=0.0,
        target_wz=0.0,
        gripper_vel=0.0,
    )


# ─────────────────────────────────────────────────────────────────────
# Joint-velocity path (per-servo direct teleop, used by SO-101 bimanual)
# ─────────────────────────────────────────────────────────────────────


def map_payload_to_joint_velocity(payload: Any, arm: str = "") -> RobotAction:
    """Translate a ``joint_velocity`` payload to ``JointVelocityServo`` keys.

    The payload shape is::

        {"kind": "joint_velocity",
         "joints": {"shoulder_pan": -1..+1, "shoulder_lift": ..., ...},
         "arm": "left"}

    Each value is a normalised velocity in ``[-1, 1]``; missing joints
    default to ``0.0`` (hold). The output dict has the per-arm prefix
    applied to ``enabled`` and to every ``vel_<joint>`` key, so the
    bimanual splitter can route it cleanly.

    Unknown / malformed payloads return a "disabled" frame so the
    upstream watchdog converges on a hold-position cue.
    """

    if not isinstance(payload, dict):
        return hold_joint_velocity(arm)
    if payload.get("kind") != PAYLOAD_JOINT_VELOCITY:
        return hold_joint_velocity(arm)
    raw_joints = payload.get("joints")
    if not isinstance(raw_joints, dict):
        return hold_joint_velocity(arm)
    velocities = {
        f"{JOINT_VELOCITY_KEY_PREFIX}{name}": _coerce_float(value)
        for name, value in raw_joints.items()
        if isinstance(name, str)
    }
    return _prefixed(arm, enabled=True, **velocities)


def hold_joint_velocity(arm: str = "") -> RobotAction:
    """Return a watchdog-friendly disabled frame for the velocity path.

    With ``enabled=False`` the
    :class:`~lerobot.robots.so_follower.robot_kinematic_processor.JointVelocityServo`
    step zeroes every joint velocity for this tick — i.e. the follower
    holds its present position.
    """

    return _prefixed(arm, enabled=False)


# ─────────────────────────────────────────────────────────────────────
# Joint-direct path (used when the follower takes raw joint targets)
# ─────────────────────────────────────────────────────────────────────


def map_payload_to_joint_action(
    payload: Any,
    state: MappingState,
    motion_scale: dict[str, float],
    arm: str = "",
) -> RobotAction:
    """Translate a control payload to a flat joint-target action dict.

    See module docstring for when to prefer this over
    :func:`map_payload_to_ee_pose_delta`.
    """

    if not isinstance(payload, dict):
        return {}
    kind = payload.get("kind")
    arm_state = state.get(arm)
    if kind == PAYLOAD_TWIST:
        return _prefixed(
            arm,
            linear_velocity=_coerce_float(payload.get("linear")),
            angular_velocity=_coerce_float(payload.get("angular")),
        )
    if kind == PAYLOAD_STOP:
        # Hold position: replay the last accumulated targets.
        return _prefix_keys(arm, arm_state)
    if kind == PAYLOAD_HOME:
        return _prefixed(arm, **{"__rfabric_remote_home__": 1.0})
    if kind == PAYLOAD_JOINT_TARGETS:
        joints = payload.get("joints") or {}
        if not isinstance(joints, dict):
            return {}
        targets = {name: _coerce_float(value) for name, value in joints.items()}
        state.joint_targets_by_arm[arm] = dict(targets)
        return _prefix_keys(arm, targets)
    if kind == PAYLOAD_JOINT_DELTAS:
        joints = payload.get("joints") or {}
        if not isinstance(joints, dict):
            return {}
        for name, delta in joints.items():
            scaled = _coerce_float(delta) * motion_scale.get(name, 1.0)
            arm_state[name] = arm_state.get(name, 0.0) + scaled
        return _prefix_keys(arm, arm_state)
    if kind == PAYLOAD_END_EFFECTOR:
        return _prefixed(
            arm,
            delta_x=_coerce_float(payload.get("deltaX")),
            delta_y=_coerce_float(payload.get("deltaY")),
            delta_z=_coerce_float(payload.get("deltaZ")),
            gripper=_coerce_float(payload.get("gripper"), default=1.0),
        )
    return {}


def hold_joint_action(state: MappingState, arm: str = "") -> RobotAction:
    """Hold-position action for the joint-direct path."""

    return _prefix_keys(arm, state.get(arm))


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _prefixed(arm: str, **fields: Any) -> RobotAction:
    if not arm:
        return dict(fields)
    return {f"{arm}_{key}": value for key, value in fields.items()}


def _prefix_keys(arm: str, mapping: dict[str, Any]) -> RobotAction:
    if not arm:
        return dict(mapping)
    return {f"{arm}_{key}": value for key, value in mapping.items()}


def _gripper_state_to_velocity(value: Any) -> float:
    """Convert the operator-side discrete gripper state to a continuous
    velocity command in [-1, 1].

    The browser emits ``gripper ∈ {0, 1, 2}`` (open / hold / close); the
    SO IK pipeline's :class:`GripperVelocityToJoint` step (with
    ``discrete_gripper=False``) integrates a continuous velocity into a
    target gripper position. LeRobot models the gripper joint as
    ``0`` = fully closed, ``100`` = fully open, so positive velocity must
    **open** and negative velocity must **close**. Mapping:

        0 -> +1.0  (open)
        1 ->  0.0  (hold)
        2 -> -1.0  (close)
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    numeric = float(value)
    if numeric <= 0.5:
        return 1.0
    if numeric >= 1.5:
        return -1.0
    return 0.0


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default
