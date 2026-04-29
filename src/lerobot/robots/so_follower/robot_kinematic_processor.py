#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model import RobotKinematics
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    TransitionKey,
)
from lerobot.utils.rotation import Rotation


def joint_position_array_from_observation(
    observation: RobotObservation, motor_names: Sequence[str]
) -> np.ndarray:
    """Return joint positions stacked in ``motor_names`` order.

    Observations use string keys such as ``\"shoulder_pan.pos\"``. Dict
    iteration order is not guaranteed to match the URDF /
    :class:`~lerobot.model.kinematics.RobotKinematics` joint ordering, so
    FK/IK must assemble ``q`` strictly in ``motor_names`` order.
    """

    return np.array(
        [float(observation[f"{motor_name}.pos"]) for motor_name in motor_names],
        dtype=float,
    )


@ProcessorStepRegistry.register("ee_reference_and_delta")
@dataclass
class EEReferenceAndDelta(RobotActionProcessorStep):
    """
    Computes a target end-effector pose from a relative delta command.

    This step takes a desired change in position and orientation (`target_*`) and applies it to a
    reference end-effector pose to calculate an absolute target pose. The reference pose is derived
    from the current robot joint positions using forward kinematics.

    The processor can operate in two modes:
    1.  `use_latched_reference=True`: The reference pose is "latched" or saved at the moment the action
        is first enabled. Subsequent commands are relative to this fixed reference.
    2.  `use_latched_reference=False`: The reference pose is updated to the robot's current pose at
        every step.

    Attributes:
        kinematics: The robot's kinematic model for forward kinematics.
        end_effector_step_sizes: A dictionary scaling the input delta commands.
        motor_names: A list of motor names required for forward kinematics.
        use_latched_reference: If True, latch the reference pose on enable; otherwise, always use the
            current pose as the reference.
        reference_ee_pose: Internal state storing the latched reference pose.
        _prev_enabled: Internal state to detect the rising edge of the enable signal.
        _command_when_disabled: Internal state to hold the last command while disabled.
        delta_position_in_reference_frame: If True, scale ``target_*`` deltas with
            ``ref[:3,:3]`` so motion follows the tool; if False, add deltas in fixed world axes.
    """

    kinematics: RobotKinematics
    end_effector_step_sizes: dict
    motor_names: list[str]
    use_latched_reference: bool = (
        True  # If True, latch reference on enable; if False, always use current pose
    )
    use_ik_solution: bool = False
    # If True, ``target_{x,y,z}`` deltas are interpreted in the reference
    # EE frame (columns of ``ref[:3,:3]``), so "right" on the operator pad
    # moves the tool to its own +X. If False, deltas are in the fixed URDF
    # world frame (legacy phone demos).
    delta_position_in_reference_frame: bool = False

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _command_when_disabled: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.use_ik_solution and "IK_solution" in self.transition.get(TransitionKey.COMPLEMENTARY_DATA):
            q_raw = self.transition.get(TransitionKey.COMPLEMENTARY_DATA)["IK_solution"]
        else:
            q_raw = joint_position_array_from_observation(observation, self.motor_names)

        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        # Current pose from FK on measured joints
        t_curr = self.kinematics.forward_kinematics(q_raw)

        enabled = bool(action.pop("enabled"))
        tx = float(action.pop("target_x"))
        ty = float(action.pop("target_y"))
        tz = float(action.pop("target_z"))
        wx = float(action.pop("target_wx"))
        wy = float(action.pop("target_wy"))
        wz = float(action.pop("target_wz"))
        gripper_vel = float(action.pop("gripper_vel"))

        desired = None

        if enabled:
            ref = t_curr
            if self.use_latched_reference:
                # Latched reference mode: latch reference at the rising edge
                if not self._prev_enabled or self.reference_ee_pose is None:
                    self.reference_ee_pose = t_curr.copy()
                ref = self.reference_ee_pose if self.reference_ee_pose is not None else t_curr

            delta_p = np.array(
                [
                    tx * self.end_effector_step_sizes["x"],
                    ty * self.end_effector_step_sizes["y"],
                    tz * self.end_effector_step_sizes["z"],
                ],
                dtype=float,
            )
            if self.delta_position_in_reference_frame:
                delta_world = ref[:3, :3] @ delta_p
            else:
                delta_world = delta_p
            r_abs = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
            desired = np.eye(4, dtype=float)
            desired[:3, :3] = ref[:3, :3] @ r_abs
            desired[:3, 3] = ref[:3, 3] + delta_world

            self._command_when_disabled = desired.copy()
        else:
            # While disabled, keep sending the same command to avoid drift.
            if self._command_when_disabled is None:
                # If we've never had an enabled command yet, freeze current FK pose once.
                self._command_when_disabled = t_curr.copy()
            desired = self._command_when_disabled.copy()

        # Write action fields
        pos = desired[:3, 3]
        tw = Rotation.from_matrix(desired[:3, :3]).as_rotvec()
        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(tw[0])
        action["ee.wy"] = float(tw[1])
        action["ee.wz"] = float(tw[2])
        action["ee.gripper_vel"] = gripper_vel

        self._prev_enabled = enabled
        return action

    def reset(self):
        """Resets the internal state of the processor."""
        self._prev_enabled = False
        self.reference_ee_pose = None
        self._command_when_disabled = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION].pop(f"{feat}", None)

        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_vel"]:
            features[PipelineFeatureType.ACTION][f"ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


@ProcessorStepRegistry.register("ee_bounds_and_safety")
@dataclass
class EEBoundsAndSafety(RobotActionProcessorStep):
    """
    Clips the end-effector pose to predefined bounds and checks for unsafe jumps.

    This step ensures that the target end-effector pose remains within a safe operational workspace.
    It also moderates the command to prevent large, sudden movements between consecutive steps.

    Attributes:
        end_effector_bounds: A dictionary with "min" and "max" keys for position clipping.
        max_ee_step_m: The maximum allowed change in position (in meters) between steps.
        _last_pos: Internal state storing the last commanded position.
    """

    end_effector_bounds: dict
    max_ee_step_m: float = 0.05
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        x = action["ee.x"]
        y = action["ee.y"]
        z = action["ee.z"]
        wx = action["ee.wx"]
        wy = action["ee.wy"]
        wz = action["ee.wz"]
        # TODO(Steven): ee.gripper_vel does not need to be bounded

        if None in (x, y, z, wx, wy, wz):
            raise ValueError(
                "Missing required end-effector pose components: x, y, z, wx, wy, wz must all be present in action"
            )

        pos = np.array([x, y, z], dtype=float)
        twist = np.array([wx, wy, wz], dtype=float)

        # Clip position
        pos = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        # Moderate jumps in position: scale down overshoot to max step length.
        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_m and n > 0:
                pos = self._last_pos + dpos * (self.max_ee_step_m / n)

        self._last_pos = pos

        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(twist[0])
        action["ee.wy"] = float(twist[1])
        action["ee.wz"] = float(twist[2])
        return action

    def reset(self):
        """Resets the last known position and orientation."""
        self._last_pos = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("inverse_kinematics_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints(RobotActionProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using inverse kinematics (IK).

    This step translates a Cartesian command (position and orientation of the end-effector) into
    the corresponding joint-space commands for each motor.

    Attributes:
        kinematics: The robot's kinematic model for inverse kinematics.
        motor_names: A list of motor names for which to compute joint positions.
        q_curr: Internal state storing the last joint positions, used as an initial guess for the IK solver.
        initial_guess_current_joints: If True, use the robot's current joint state as the IK guess.
            If False, use the solution from the previous step.
        position_weight: Placo position task weight passed to :meth:`RobotKinematics.inverse_kinematics`.
        orientation_weight: Placo orientation weight; use ``0.0`` for translation-first Cartesian teleop.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True
    position_weight: float = 1.0
    orientation_weight: float = 0.01

    def action(self, action: RobotAction) -> RobotAction:
        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = joint_position_array_from_observation(observation, self.motor_names)
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.initial_guess_current_joints:  # Use current joints as initial guess
            self.q_curr = q_raw
        else:  # Use previous ik solution as initial guess
            if self.q_curr is None:
                self.q_curr = q_raw

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute inverse kinematics
        q_target = self.kinematics.inverse_kinematics(
            self.q_curr,
            t_des,
            position_weight=self.position_weight,
            orientation_weight=self.orientation_weight,
        )
        self.q_curr = q_target

        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """Resets the initial guess for the IK solver."""
        self.q_curr = None


@ProcessorStepRegistry.register("gripper_velocity_to_joint")
@dataclass
class GripperVelocityToJoint(RobotActionProcessorStep):
    """
    Converts a gripper velocity command into a target gripper joint position.

    This step integrates a normalized velocity command over time to produce a position command,
    taking the current gripper position as a starting point. It also supports a discrete mode
    where integer actions map to open, close, or no-op.

    Attributes:
        motor_names: A list of motor names, which must include 'gripper'.
        speed_factor: A scaling factor to convert the normalized velocity command to a position change.
        clip_min: The minimum allowed gripper joint position.
        clip_max: The maximum allowed gripper joint position.
        discrete_gripper: If True, treat the input action as discrete (0: open, 1: close, 2: stay).
    """

    speed_factor: float = 20.0
    clip_min: float = 0.0
    clip_max: float = 100.0
    discrete_gripper: bool = False

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        gripper_vel = action.pop("ee.gripper_vel")

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        gripper_present = float(observation["gripper.pos"])

        if self.discrete_gripper:
            # Discrete gripper actions are in [0, 1, 2]
            # 0: open, 1: close, 2: stay
            # We need to shift them to [-1, 0, 1] and then scale them to clip_max
            gripper_vel = (gripper_vel - 1) * self.clip_max

        # Compute desired gripper position (read present gripper by key so
        # it stays correct regardless of dict iteration order).
        delta = gripper_vel * float(self.speed_factor)
        gripper_pos = float(np.clip(gripper_present + delta, self.clip_min, self.clip_max))
        action["ee.gripper_pos"] = gripper_pos

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION].pop("ee.gripper_vel", None)
        features[PipelineFeatureType.ACTION]["ee.gripper_pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )

        return features


@ProcessorStepRegistry.register("joint_velocity_servo")
@dataclass
class JointVelocityServo(RobotActionProcessorStep):
    """Integrate per-joint normalised velocities into joint position targets.

    The operator UI emits one normalised velocity per servo (range
    ``[-1, 1]``) and this step converts it into a per-tick position
    command for the follower. Two integration shapes are used:

    * **Actuators** (everything except the gripper). The target marches
      forward independently of the live joint reading::

          target ← clamp(prev_target + velocity * step,
                          present - lead_cap, present + lead_cap)

      If the motor tracks the goal cleanly, ``prev_target`` stays near
      ``present`` and the position error feeding the Feetech PID is
      small (smooth motion). If the motor lags — e.g. the elbow under
      gravity load briefly fails to overcome static friction — the
      target keeps advancing while ``present`` doesn't, the position
      error grows, and the PID develops more torque to break free.
      The lead cap bounds this windup so a sustained stall against a
      mechanical hard stop never produces unbounded torque.

    * **Gripper** (``"gripper"``). Pure direct-drive integration in the
      ``[0, 100]`` range::

          target ← clamp(present + velocity * gripper_step, 0, 100)

      The gripper has no inertia worth integrating across ticks and
      its position scale is bounded, so the simpler form is strictly
      better here.

    ``enabled=False`` (or a missing velocity) holds every joint at its
    present position and clears any accumulated lead.

    Before integration, each joint's raw command velocity is passed
    through a first-order low-pass toward its target so presses ramp
    gently. When the operator releases (raw goes to zero), the
    **smoothed** velocity still decays for a few ticks — integration
    continues with that decay so the joint eases out instead of
    snapping still. Once smoothed magnitude drops below
    :attr:`velocity_idle_epsilon`, the actuator target latches to the
    live reading for gravity hold.

    Input action keys (produced by
    :func:`~lerobot.teleoperators.rfabric_remote.action_mapping.map_payload_to_joint_velocity`,
    with any per-arm prefix already stripped):

        ``enabled`` (bool), ``vel_<joint>`` (float, ``[-1, 1]``).

    Output action keys: ``<joint>.pos`` for every motor in
    :attr:`motor_names` (degrees for actuators; ``[clip_min, clip_max]``
    for the gripper).

    Attributes:
        motor_names: Bus-ordered motor names. The gripper, if present,
            is named ``"gripper"`` and uses :attr:`gripper_step_per_tick`
            instead of :attr:`step_per_tick_deg`.
        step_per_tick_deg: Per-tick angular gain (deg) for non-gripper
            joints at full input (``vel = ±1``).
        gripper_step_per_tick: Per-tick gain for the gripper joint at
            full input. The gripper is range-normalised to
            ``[gripper_clip_min, gripper_clip_max]``.
        gripper_clip_min, gripper_clip_max: Hard clamp on the gripper
            command (default ``0..100`` to match lerobot's gripper range).
        lead_cap_deg: Maximum signed lead the integrating target may
            hold over the live joint reading for actuators. Sized to
            give the Feetech PID a position error large enough to
            overcome static friction on loaded joints (the elbow under
            gravity is the worst case on the SO-101) without producing
            damaging torque against mechanical hard stops.
        velocity_tau_seconds: Time constant (seconds) for smoothing raw
            ``[-1, 1]`` command velocities toward the operator input.
            ``0`` disables smoothing (instant step). Larger values make
            ramps slower and softer.
        control_period_seconds: Nominal control tick duration used with
            ``velocity_tau_seconds`` to compute the per-tick blend
            factor ``1 - exp(-dt / tau)``.
        velocity_idle_epsilon: When both raw and smoothed command
            velocities are below this magnitude (after release), the
            actuator target latches to ``present`` for holding. Larger
            values shorten the coast-out tail; too large clips the end
            of the ramp early.
    """

    motor_names: list[str] = field(default_factory=list)
    step_per_tick_deg: float = 2.0
    gripper_step_per_tick: float = 6.0
    gripper_clip_min: float = 0.0
    gripper_clip_max: float = 100.0
    velocity_key_prefix: str = "vel_"
    lead_cap_deg: float = 12.0
    velocity_tau_seconds: float = 0.10
    control_period_seconds: float = 1.0 / 60.0
    velocity_idle_epsilon: float = 0.04

    def __post_init__(self) -> None:
        self._target_deg: dict[str, float] = {}
        self._previous_raw_velocity: dict[str, float] = {}
        self._smoothed_velocity: dict[str, float] = {}
        self._gripper_hold_target: dict[str, float] = {}

    def _velocity_blend_factor(self) -> float:
        if self.velocity_tau_seconds <= 0.0:
            return 1.0
        return 1.0 - math.exp(-self.control_period_seconds / self.velocity_tau_seconds)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("JointVelocityServo requires the joint observation in the transition.")

        enabled = bool(action.pop("enabled", False))
        blend = self._velocity_blend_factor()

        for motor in self.motor_names:
            present = float(observation[f"{motor}.pos"])
            raw_velocity = float(action.pop(f"{self.velocity_key_prefix}{motor}", 0.0))
            if not enabled:
                raw_velocity = 0.0
            previous_smoothed = self._smoothed_velocity.get(motor, 0.0)
            smoothed_velocity = previous_smoothed + blend * (raw_velocity - previous_smoothed)
            self._smoothed_velocity[motor] = smoothed_velocity

            if motor == "gripper":
                target = self._gripper_target_from_smoothed_velocity(
                    motor=motor,
                    present=present,
                    raw_velocity=raw_velocity,
                    smoothed_velocity=smoothed_velocity,
                    previous_smoothed=previous_smoothed,
                )
            else:
                target = self._integrate_actuator_target(
                    motor,
                    present,
                    smoothed_velocity=smoothed_velocity,
                    raw_velocity=raw_velocity,
                    previous_smoothed=previous_smoothed,
                )

            action[f"{motor}.pos"] = float(target)

        # Drop any leftover velocity keys for joints we don't drive
        # (keeps the action dict tidy if the wire payload referenced
        # an unknown joint name).
        for key in [k for k in action if isinstance(k, str) and k.startswith(self.velocity_key_prefix)]:
            action.pop(key, None)
        return action

    def _gripper_target_from_smoothed_velocity(
        self,
        *,
        motor: str,
        present: float,
        raw_velocity: float,
        smoothed_velocity: float,
        previous_smoothed: float,
    ) -> float:
        eps_raw = 1e-12
        eps_idle = self.velocity_idle_epsilon
        previous_raw = self._previous_raw_velocity.get(motor, 0.0)
        self._previous_raw_velocity[motor] = raw_velocity
        fully_idle = abs(raw_velocity) < eps_raw and abs(smoothed_velocity) < eps_idle
        if fully_idle:
            had_motion = (abs(previous_raw) >= eps_raw) or (abs(previous_smoothed) >= eps_idle)
            if had_motion or motor not in self._gripper_hold_target:
                self._gripper_hold_target[motor] = float(
                    np.clip(present, self.gripper_clip_min, self.gripper_clip_max)
                )
            return self._gripper_hold_target[motor]
        return float(
            np.clip(
                present + smoothed_velocity * self.gripper_step_per_tick,
                self.gripper_clip_min,
                self.gripper_clip_max,
            )
        )

    def _integrate_actuator_target(
        self,
        motor: str,
        present: float,
        *,
        smoothed_velocity: float,
        raw_velocity: float,
        previous_smoothed: float,
    ) -> float:
        """Advance the integrating target for one actuator joint.

        While raw or smoothed command is non-trivial, integrate with
        ``smoothed_velocity * step`` (lead-capped). When the operator
        releases, raw hits zero first but smoothed decays — integration
        continues so the joint coasts down. Once both are below
        :attr:`velocity_idle_epsilon`, latch ``present`` for gravity
        hold. Direction reversal uses raw sign against lead.
        """

        eps_raw = 1e-12
        eps_idle = self.velocity_idle_epsilon
        previous_raw = self._previous_raw_velocity.get(motor, 0.0)
        self._previous_raw_velocity[motor] = raw_velocity

        fully_idle = abs(raw_velocity) < eps_raw and abs(smoothed_velocity) < eps_idle
        if fully_idle:
            had_motion = (abs(previous_raw) >= eps_raw) or (abs(previous_smoothed) >= eps_idle)
            if had_motion or motor not in self._target_deg:
                self._target_deg[motor] = present
            return self._target_deg[motor]

        previous_target = self._target_deg.get(motor, present)
        if raw_velocity * (previous_target - present) < 0.0:
            previous_target = present
        candidate_target = previous_target + smoothed_velocity * self.step_per_tick_deg
        upper_bound = present + self.lead_cap_deg
        lower_bound = present - self.lead_cap_deg
        clamped_target = max(lower_bound, min(upper_bound, candidate_target))
        self._target_deg[motor] = clamped_target
        return clamped_target

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        action_features.pop("enabled", None)
        for key in [k for k in list(action_features) if k.startswith(self.velocity_key_prefix)]:
            action_features.pop(key, None)
        for motor in self.motor_names:
            action_features[f"{motor}.pos"] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features


def compute_forward_kinematics_joints_to_ee(
    joints: dict[str, Any], kinematics: RobotKinematics, motor_names: list[str]
) -> dict[str, Any]:
    motor_joint_values = [joints[f"{n}.pos"] for n in motor_names]

    q = np.array(motor_joint_values, dtype=float)
    t = kinematics.forward_kinematics(q)
    pos = t[:3, 3]
    tw = Rotation.from_matrix(t[:3, :3]).as_rotvec()
    gripper_pos = joints["gripper.pos"]
    for n in motor_names:
        joints.pop(f"{n}.pos")
    joints["ee.x"] = float(pos[0])
    joints["ee.y"] = float(pos[1])
    joints["ee.z"] = float(pos[2])
    joints["ee.wx"] = float(tw[0])
    joints["ee.wy"] = float(tw[1])
    joints["ee.wz"] = float(tw[2])
    joints["ee.gripper_pos"] = float(gripper_pos)
    return joints


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_observation")
@dataclass
class ForwardKinematicsJointsToEEObservation(ObservationProcessorStep):
    """
    Computes the end-effector pose from joint positions using forward kinematics (FK).

    This step is typically used to add the robot's Cartesian pose to the observation space,
    which can be useful for visualization or as an input to a policy.

    Attributes:
        kinematics: The robot's kinematic model.
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def observation(self, observation: RobotObservation) -> RobotObservation:
        return compute_forward_kinematics_joints_to_ee(observation, self.kinematics, self.motor_names)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We only use the ee pose in the dataset, so we don't need the joint positions
        for n in self.motor_names:
            features[PipelineFeatureType.OBSERVATION].pop(f"{n}.pos", None)
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.OBSERVATION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_action")
@dataclass
class ForwardKinematicsJointsToEEAction(RobotActionProcessorStep):
    """
    Computes the end-effector pose from joint positions using forward kinematics (FK).

    This step is typically used to add the robot's Cartesian pose to the observation space,
    which can be useful for visualization or as an input to a policy.

    Attributes:
        kinematics: The robot's kinematic model.
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def action(self, action: RobotAction) -> RobotAction:
        return compute_forward_kinematics_joints_to_ee(action, self.kinematics, self.motor_names)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We only use the ee pose in the dataset, so we don't need the joint positions
        for n in self.motor_names:
            features[PipelineFeatureType.ACTION].pop(f"{n}.pos", None)
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register(name="forward_kinematics_joints_to_ee")
@dataclass
class ForwardKinematicsJointsToEE(ProcessorStep):
    kinematics: RobotKinematics
    motor_names: list[str]

    def __post_init__(self):
        self.joints_to_ee_action_processor = ForwardKinematicsJointsToEEAction(
            kinematics=self.kinematics, motor_names=self.motor_names
        )
        self.joints_to_ee_observation_processor = ForwardKinematicsJointsToEEObservation(
            kinematics=self.kinematics, motor_names=self.motor_names
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if transition.get(TransitionKey.ACTION) is not None:
            transition = self.joints_to_ee_action_processor(transition)
        if transition.get(TransitionKey.OBSERVATION) is not None:
            transition = self.joints_to_ee_observation_processor(transition)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if features[PipelineFeatureType.ACTION] is not None:
            features = self.joints_to_ee_action_processor.transform_features(features)
        if features[PipelineFeatureType.OBSERVATION] is not None:
            features = self.joints_to_ee_observation_processor.transform_features(features)
        return features


@ProcessorStepRegistry.register("inverse_kinematics_rl_step")
@dataclass
class InverseKinematicsRLStep(ProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using inverse kinematics (IK).

    This is modified from the InverseKinematicsEEToJoints step to be used in the RL pipeline.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = dict(transition)
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("Action is required for InverseKinematicsEEToJoints")
        action = dict(action)

        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = new_transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = joint_position_array_from_observation(observation, self.motor_names)
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.initial_guess_current_joints:  # Use current joints as initial guess
            self.q_curr = q_raw
        else:  # Use previous ik solution as initial guess
            if self.q_curr is None:
                self.q_curr = q_raw

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute inverse kinematics
        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        new_transition[TransitionKey.ACTION] = action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data["IK_solution"] = q_target
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """Resets the initial guess for the IK solver."""
        self.q_curr = None
