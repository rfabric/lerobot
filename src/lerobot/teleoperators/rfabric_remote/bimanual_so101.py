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
"""Bimanual SO-101 entry point for the rFabric remote bridge.

This is the production-shaped runner used by ``rfabric-arms.service``
and the local ``make run-arms`` target. It glues four pieces together:

* A :class:`BiSOFollower` controlling two SO-101 arms over their own
  serial buses.
* A :class:`RFabricRemoteTeleop` bound to the Unix-domain socket the
  rFabric agent connects into. It is configured with
  ``arms = ["left", "right"]`` and ``action_mode = "joint_velocity"``
  so every operator-side ``joint_velocity`` payload arrives here as a
  per-arm-prefixed dict of normalised servo velocities.
* Two parallel processor pipelines (one per arm), each a single
  :class:`JointVelocityServo` step. There is no IK and no FK in the
  control loop — every operator key maps directly to one motor with a
  fixed sign. Per-tick motion is ``velocity * step``, with
  ``step_per_tick_deg`` for actuators and ``gripper_step_per_tick`` for
  the gripper.
* A loop that splits the bimanual observation / action by ``left_`` /
  ``right_`` prefix, runs each through its arm's servo step,
  re-prefixes the resulting joint targets, and dispatches the merged
  dict back to the bimanual follower.

The URDF is still loaded for the optional ``--show-rest-pose`` and
``--probe-joints`` diagnostics, but the live control loop never reads
it: per-servo teleop needs only motor names and live joint readings.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from lerobot.motors.motors_bus import MotorCalibration
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollower, SOFollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    JointVelocityServo,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from .configuration_rfabric_remote import RFabricRemoteTeleopConfig
from .teleop_rfabric_remote import (
    ACTION_MODE_JOINT_VELOCITY,
    RFabricRemoteTeleop,
)

LOG = logging.getLogger(__name__)

ARMS: tuple[str, str] = ("left", "right")

# Teleop defaults: the operator UI emits a normalised velocity per
# servo each tick. ``joint_step_deg`` is the per-tick angular gain at
# full input — combined with the loop rate it sets the peak slew
# (deg/s = step * fps). At the defaults below that is 120°/s on each
# servo, which feels responsive on the SO-101 without overshooting.
# A 60 Hz tick is also where the Feetech STS3215's internal
# trajectory generator stops looking notchy — at 30 Hz the discrete
# goal updates are visible as small jolts.
DEFAULT_TARGET_FRAME = "gripper_frame_link"
DEFAULT_JOINT_STEP_DEG = 1.0
DEFAULT_GRIPPER_STEP = 3.0
DEFAULT_FPS = 60
# How far the integrating actuator target may lead the live joint
# reading (in degrees). The Feetech STS3215 PID derives torque from
# position error; with the SO-101's lerobot-default ``P_Coefficient =
# 16`` a 2°/tick step alone produces a 2° error which is below the
# stiction band on the gravity-loaded elbow. Letting the target run
# ahead by up to this many degrees gives the PID enough error to
# break free, while bounding wind-up against mechanical hard stops.
DEFAULT_TARGET_LEAD_CAP_DEG = 12.0
# Lerobot's SO-101 calibration burns the recorded ``range_min`` /
# ``range_max`` directly into each motor's EEPROM as
# ``Min/Max_Position_Limit``. If during the range-of-motion phase the
# operator did not move a joint past the homing pose in both
# directions, the burned-in limits exclude the rest position and the
# motor refuses to follow goals that would cross either edge. We
# detect this on connect (homing pose within this many degrees of an
# edge, or outside the range entirely) and reopen the motor's EEPROM
# limits to the full encoder span — the arm's mechanical hard stops
# remain the real safety envelope.
DEFAULT_REST_POSE_MARGIN_DEG = 18.0
# STS3215 / STS3250 / SM8512BL all have a 4096-count encoder. We use
# this for the deg ↔ count conversions that drive the auto-widening
# logic; the bridge intentionally only supports SO-101 so it is safe
# to hard-code here rather than reach into the per-motor model table.
_ENCODER_RESOLUTION = 4096
# Joints whose calibration has range_min == 0 and range_max == 4095
# (full-turn motors like the SO-101 wrist roll) are skipped — they
# already cover the full encoder space and have no homing limit to
# widen.
_FULL_TURN_JOINT_NAMES = frozenset({"wrist_roll"})
DEFAULT_SOCKET = Path("/run/rfabric/control.sock")
# Stable id so each SOFollower arm gets its own calibration JSON
# (``<id>_left.json`` / ``<id>_right.json`` under the so_follower cache).
# If BiSOFollowerConfig.id is omitted, both arms share ``None.json`` and
# the second bus loads the first arm's homing limits — Feetech then
# returns incorrect status packets on ``Torque_Enable``.
DEFAULT_ROBOT_ID = "rfabric_bimanual"


# Joint names that bypass the URDF/IK alignment layer. The gripper is
# normalised through a different range (RANGE_0_100), is downstream of
# the fixed gripper_frame_joint, and goes motor → motor without ever
# entering the IK joint vector — so sign / zero-offset overrides do
# not apply to it.
_ALIGNMENT_BYPASS_JOINTS = frozenset({"gripper"})


@dataclass(frozen=True)
class JointAlignment:
    """Per-joint motor-frame ↔ URDF-frame reconciliation knobs.

    Lerobot's calibration normalises ``Present_Position`` into degrees
    measured from the **calibrated mid-of-range** of each joint, with
    ``drive_mode=0`` hard-coded. That is only physically aligned with
    the URDF if (a) the arm sat at the URDF rest pose during the
    half-turn-homing prompt and (b) every motor is mounted with the
    same positive direction as its URDF ``<axis>``. Mirrored bimanual
    mounts almost always violate (b) on at least one joint.

    This class lets the operator inject corrections without re-running
    the lerobot calibration ritual:

    * ``sign``: ``+1`` if motor positive direction matches the URDF
      ``<axis>`` for that joint, ``-1`` if physically reversed.
    * ``zero_offset_deg``: what ``Present_Position`` reads (in degrees)
      when the arm is at the URDF rest pose. ``0`` if calibration mid-
      of-range already coincides with URDF zero.

    Conversions:

        q_urdf  = sign * (q_motor - zero_offset_deg)
        q_motor = sign *  q_urdf + zero_offset_deg
    """

    sign: dict[str, int]
    zero_offset_deg: dict[str, float]

    @classmethod
    def identity(cls) -> JointAlignment:
        return cls(sign={}, zero_offset_deg={})

    def motor_to_urdf(self, joint_name: str, motor_deg: float) -> float:
        return self._sign(joint_name) * (motor_deg - self._offset(joint_name))

    def urdf_to_motor(self, joint_name: str, urdf_deg: float) -> float:
        return self._sign(joint_name) * urdf_deg + self._offset(joint_name)

    def _sign(self, joint_name: str) -> int:
        return self.sign.get(joint_name, 1)

    def _offset(self, joint_name: str) -> float:
        return self.zero_offset_deg.get(joint_name, 0.0)


@dataclass(frozen=True)
class ArmContext:
    """Per-arm runtime bundle: prefix, joint names, IK pipeline, alignment."""

    arm: str
    motor_names: list[str]
    pipeline: RobotProcessorPipeline
    alignment: JointAlignment


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    init_logging()

    if args.show_rest_pose:
        _show_rest_pose(args)
        return

    robot = _build_robot(args)

    if args.probe_joints:
        _run_joint_probe(robot=robot, args=args)
        return

    if args.diagnose_signs:
        _diagnose_signs(robot=robot, args=args)
        return

    teleop = _build_teleop(args)

    robot.connect()
    _ensure_rest_pose_inside_motor_limits(robot, float(args.rest_pose_margin_deg))
    teleop.connect()
    try:
        contexts = [_build_arm_context(robot, arm, args) for arm in ARMS]
        _log_axis_convention(args)
        _log_alignment(contexts)
        _run_loop(robot=robot, teleop=teleop, contexts=contexts, fps=args.fps, args=args)
    except KeyboardInterrupt:
        LOG.info("rfabric_remote: stopping bimanual SO-101 loop")
    finally:
        teleop.disconnect()
        robot.disconnect()


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lerobot.teleoperators.rfabric_remote.bimanual_so101",
        description="Bimanual SO-101 follower driven by the rFabric remote UDS bridge.",
    )
    parser.add_argument(
        "--left-port",
        required=True,
        help="Serial port of the left SO-101 follower (e.g. /dev/tty.usbmodem...).",
    )
    parser.add_argument(
        "--right-port",
        required=True,
        help="Serial port of the right SO-101 follower.",
    )
    parser.add_argument(
        "--urdf",
        required=True,
        type=Path,
        help="Path to the SO-101 URDF file used by both arms' kinematic solvers.",
    )
    parser.add_argument(
        "--socket-path",
        type=Path,
        default=DEFAULT_SOCKET,
        help=f"UDS path the agent connects to (default: {DEFAULT_SOCKET}).",
    )
    parser.add_argument(
        "--robot-id",
        default=DEFAULT_ROBOT_ID,
        help=(
            "BiSOFollower config id; each arm stores calibration as "
            "``<robot-id>_left.json`` / ``<robot-id>_right.json``. Must be "
            "non-empty so the two serial buses never share one file."
        ),
    )
    parser.add_argument(
        "--target-frame",
        default=DEFAULT_TARGET_FRAME,
        help=(
            "URDF frame used by the diagnostic helpers (--show-rest-pose, "
            f"--probe-joints). Default: {DEFAULT_TARGET_FRAME}."
        ),
    )
    parser.add_argument(
        "--joint-step-deg",
        type=float,
        default=DEFAULT_JOINT_STEP_DEG,
        help=(
            "Per-tick angular gain (deg) for non-gripper joints at full "
            "operator input (velocity = ±1). At the default 30 Hz tick this "
            "yields ~%.0f°/s peak slew on each servo."
        )
        % (DEFAULT_JOINT_STEP_DEG * DEFAULT_FPS),
    )
    parser.add_argument(
        "--gripper-step",
        type=float,
        default=DEFAULT_GRIPPER_STEP,
        help=(
            "Per-tick gain for the gripper joint at full input. The "
            "gripper position is normalised to [0, 100] and clamped on "
            "every tick."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Control loop target frequency (default: {DEFAULT_FPS} Hz).",
    )
    parser.add_argument(
        "--target-lead-cap-deg",
        type=float,
        default=DEFAULT_TARGET_LEAD_CAP_DEG,
        help=(
            "Maximum lead (deg) the integrating per-joint target may "
            "hold over the live reading on actuators. Larger values let "
            "the Feetech PID develop more torque on stalled / heavily "
            "loaded joints (the SO-101 elbow under gravity is the "
            "worst case) at the cost of more wind-up torque against "
            f"mechanical hard stops. Default: {DEFAULT_TARGET_LEAD_CAP_DEG:g}°."
        ),
    )
    parser.add_argument(
        "--rest-pose-margin-deg",
        type=float,
        default=DEFAULT_REST_POSE_MARGIN_DEG,
        help=(
            "Tolerance band around each calibrated range edge (in degrees). "
            "If the homing pose falls within this band of either edge — or "
            "outside the range entirely — the bridge treats the calibration "
            "as malformed and reopens the motor's EEPROM limits to the full "
            "encoder span for this session. Compensates for lerobot "
            "calibrations whose recorded range-of-motion doesn't bracket "
            "the homing pose with adequate room. Set to 0 to keep the "
            f"calibrated range as-is (default: {DEFAULT_REST_POSE_MARGIN_DEG:g}°)."
        ),
    )
    parser.add_argument(
        "--watchdog-ms",
        type=int,
        default=200,
        help="Watchdog window applied to each arm independently.",
    )
    parser.add_argument(
        "--use-degrees",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the SO-101 motors should report joint angles in degrees.",
    )
    # Per-arm calibration ↔ URDF reconciliation. See JointAlignment.
    for arm in ARMS:
        parser.add_argument(
            f"--{arm}-joint-sign",
            type=_parse_joint_signs,
            default={},
            metavar="JOINT=±1[,JOINT=±1...]",
            help=(
                f"Per-joint sign overrides for the {arm} arm. "
                "Use -1 when a motor is mounted opposite the URDF axis "
                "(common on mirrored mounts). Example: "
                "shoulder_pan=-1,wrist_roll=-1"
            ),
        )
        parser.add_argument(
            f"--{arm}-joint-zero-offset-deg",
            type=_parse_joint_floats,
            default={},
            metavar="JOINT=DEG[,JOINT=DEG...]",
            help=(
                f"Per-joint zero-offset (degrees) for the {arm} arm — what "
                "Present_Position reads when the arm is at the URDF rest "
                "pose. Use this to trim mid-of-range vs URDF-zero mismatch "
                "without re-running lerobot calibration."
            ),
        )
    parser.add_argument(
        "--probe-joints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Connect without running interactive calibration, read each "
            "arm's joint readings (motor frame and, if alignment is set, "
            "URDF frame), print, and exit. Requires existing calibration "
            "files for both arms (use a normal run once, or lerobot-calibrate "
            "per port). If you ever ran without ``--robot-id``, delete stale "
            "``None.json`` under the so_follower calibration cache manually."
        ),
    )
    parser.add_argument(
        "--show-rest-pose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Open a native MuJoCo window rendering so101_new_calib.xml at "
            "q=0 (the URDF rest pose, == lerobot's calibrated mid-of-range). "
            "Use this to visually match the physical arm before running "
            "--probe-joints. Exit by closing the window or Ctrl-C."
        ),
    )
    parser.add_argument(
        "--diagnose-signs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Interactive per-joint sign diagnostic. Disables torque on each "
            "arm in turn, asks you to hand-move every non-gripper joint in "
            "the URDF-positive direction, infers the sign of the motor "
            "reading, and prints the exact ``--<arm>-joint-sign=...`` flag "
            "to use. Run with both arms posed at the URDF rest pose."
        ),
    )
    args = parser.parse_args(argv)
    if not str(args.robot_id).strip():
        parser.error("--robot-id must be a non-empty string.")
    urdf_path = args.urdf.expanduser()
    if not urdf_path.is_file():
        parser.error(
            "--urdf must be an existing file path to so101_new_calib.urdf (or equivalent). "
            f"Not found: {urdf_path}. In the lerobot fork the default is "
            "<lerobot>/assets/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
        )
    args.urdf = urdf_path.resolve()
    return args


def _parse_joint_floats(value: str) -> dict[str, float]:
    """Parse ``shoulder_pan=15.5,elbow_flex=-3.0`` into a dict."""

    result: dict[str, float] = {}
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            continue
        joint, separator, magnitude = token.partition("=")
        if not separator or not joint or not magnitude:
            raise argparse.ArgumentTypeError(f"expected JOINT=NUMBER, got {raw_token!r}")
        try:
            result[joint.strip()] = float(magnitude)
        except ValueError as err:
            raise argparse.ArgumentTypeError(f"could not parse {raw_token!r} as JOINT=NUMBER") from err
    return result


def _parse_joint_signs(value: str) -> dict[str, int]:
    """Parse ``shoulder_pan=-1,wrist_roll=-1`` into a dict; values must be ±1."""

    result: dict[str, int] = {}
    for joint, magnitude in _parse_joint_floats(value).items():
        sign = int(magnitude)
        if sign not in (1, -1):
            raise argparse.ArgumentTypeError(f"joint sign for {joint!r} must be 1 or -1, got {magnitude}")
        result[joint] = sign
    return result


# ─────────────────────────────────────────────────────────────────────
# Construction helpers
# ─────────────────────────────────────────────────────────────────────


def _build_robot(args: argparse.Namespace) -> BiSOFollower:
    # Defense-in-depth safety cap at the bus layer: bound any single-tick
    # |goal - present| produced by the servo. The integrating actuator
    # target may lead the live reading by up to ``lead_cap_deg`` (so
    # the PID develops enough error to overcome static friction on
    # loaded joints) — we add a 2x headroom on top of that so rounding
    # at the cap edge doesn't tip ``ensure_safe_goal_position`` into
    # scaling every joint down. The gripper path is bounded
    # independently by its own clamp so we just keep its 4x envelope.
    joint_step_cap = max(
        2.0 * float(args.target_lead_cap_deg),
        4.0 * float(args.gripper_step),
    )
    config = BiSOFollowerConfig(
        id=str(args.robot_id).strip(),
        left_arm_config=SOFollowerConfig(
            port=args.left_port,
            use_degrees=args.use_degrees,
            max_relative_target=joint_step_cap,
        ),
        right_arm_config=SOFollowerConfig(
            port=args.right_port,
            use_degrees=args.use_degrees,
            max_relative_target=joint_step_cap,
        ),
    )
    return BiSOFollower(config)


def _ensure_rest_pose_inside_motor_limits(
    robot: BiSOFollower,
    margin_deg: float,
) -> None:
    """Reset bad EEPROM Min/Max_Position_Limit windows to the full motor range.

    Lerobot's SO-101 calibration writes the recorded ``range_min`` /
    ``range_max`` directly into each Feetech motor's EEPROM as a hard
    travel limit. If during the range-of-motion phase the operator
    did not move a joint past the homing pose in both directions,
    the burned-in window excludes the rest position — the motor then
    silently clamps any goal that would cross either edge, which
    manifests as a joint that refuses to leave the homing region.
    Whenever we detect this on connect, we throw away the malformed
    EEPROM window for that joint and reopen it to the full ``[0,
    encoder_max]`` so the bridge can drive the arm through whatever
    range its mechanical hard stops permit. The calibration JSON on
    disk is left untouched — only the motor's runtime EEPROM and the
    in-memory bus calibration are mutated for this session.
    """

    if margin_deg <= 0.0:
        return
    margin_counts = max(1, int(round(margin_deg * _ENCODER_RESOLUTION / 360.0)))
    for arm_label, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
        _reset_arm_motor_limits_when_calibration_excludes_rest(arm, arm_label, margin_counts)


def _reset_arm_motor_limits_when_calibration_excludes_rest(
    arm: SOFollower,
    arm_label: str,
    margin_counts: int,
) -> None:
    if not arm.calibration:
        return
    rest_center = _ENCODER_RESOLUTION // 2
    encoder_max = _ENCODER_RESOLUTION - 1
    widened: dict[str, MotorCalibration] = {}
    adjustments: list[tuple[str, int, int, int]] = []
    for motor, calibration in arm.calibration.items():
        if motor in _FULL_TURN_JOINT_NAMES:
            widened[motor] = calibration
            continue
        encoder_at_rest = calibration.homing_offset + rest_center
        below_window = encoder_at_rest < calibration.range_min + margin_counts
        above_window = encoder_at_rest > calibration.range_max - margin_counts
        if not below_window and not above_window:
            widened[motor] = calibration
            continue
        widened[motor] = MotorCalibration(
            id=calibration.id,
            drive_mode=calibration.drive_mode,
            homing_offset=calibration.homing_offset,
            range_min=0,
            range_max=encoder_max,
        )
        adjustments.append((motor, encoder_at_rest, calibration.range_min, calibration.range_max))

    if not adjustments:
        return

    with arm.bus.torque_disabled():
        arm.bus.write_calibration(widened)

    for motor, rest, old_min, old_max in adjustments:
        LOG.warning(
            "rfabric_remote: %s arm %-14s homing pose (encoder=%d) is on/outside calibrated "
            "range [%d, %d]; reopened motor EEPROM limit to [0, %d] for this session — "
            "recalibrate range-of-motion through the home pose to restore a safety envelope.",
            arm_label,
            motor,
            rest,
            old_min,
            old_max,
            encoder_max,
        )


def _build_teleop(args: argparse.Namespace) -> RFabricRemoteTeleop:
    config = RFabricRemoteTeleopConfig(
        socket_path=args.socket_path,
        watchdog_ms=args.watchdog_ms,
        accepts=["joint_velocity", "stop", "home"],
        arms=list(ARMS),
        action_mode=ACTION_MODE_JOINT_VELOCITY,
    )
    return RFabricRemoteTeleop(config)


def _build_arm_context(
    robot: BiSOFollower,
    arm: str,
    args: argparse.Namespace,
) -> ArmContext:
    """Build the per-arm joint-velocity servo pipeline."""

    sub_robot = robot.left_arm if arm == "left" else robot.right_arm
    motor_names = list(sub_robot.bus.motors.keys())
    pipeline = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            JointVelocityServo(
                motor_names=motor_names,
                step_per_tick_deg=float(args.joint_step_deg),
                gripper_step_per_tick=float(args.gripper_step),
                lead_cap_deg=float(args.target_lead_cap_deg),
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    alignment = _build_alignment(arm, motor_names, args)
    return ArmContext(arm=arm, motor_names=motor_names, pipeline=pipeline, alignment=alignment)


def _build_alignment(
    arm: str,
    motor_names: Sequence[str],
    args: argparse.Namespace,
) -> JointAlignment:
    sign_overrides: dict[str, int] = getattr(args, f"{arm}_joint_sign")
    offset_overrides: dict[str, float] = getattr(args, f"{arm}_joint_zero_offset_deg")
    motor_set = set(motor_names)
    _validate_alignment_keys(arm, "sign", sign_overrides, motor_set)
    _validate_alignment_keys(arm, "zero-offset", offset_overrides, motor_set)
    return JointAlignment(sign=dict(sign_overrides), zero_offset_deg=dict(offset_overrides))


def _validate_alignment_keys(
    arm: str,
    label: str,
    overrides: dict[str, object],
    valid_motor_names: set[str],
) -> None:
    unknown = sorted(set(overrides) - valid_motor_names)
    if unknown:
        raise SystemExit(
            f"--{arm}-joint-{label} references unknown joints {unknown}; "
            f"valid joints: {sorted(valid_motor_names)}"
        )


# ─────────────────────────────────────────────────────────────────────
# Loop
# ─────────────────────────────────────────────────────────────────────


def _run_loop(
    *,
    robot: BiSOFollower,
    teleop: RFabricRemoteTeleop,
    contexts: list[ArmContext],
    fps: int,
    args: argparse.Namespace,
) -> None:
    period = 1.0 / max(fps, 1)
    LOG.info("rfabric_remote: bimanual SO-101 loop running at %d Hz", fps)
    while True:
        loop_start = time.perf_counter()

        observation = robot.get_observation()
        teleop.send_feedback(observation)

        teleop_action = teleop.get_action()

        joint_action: RobotAction = {}
        for context in contexts:
            arm_action = _strip_prefix(teleop_action, context.arm)
            arm_observation = _align_observation_to_urdf(
                _strip_prefix(observation, context.arm), context.alignment
            )
            arm_joint_targets = context.pipeline((arm_action, arm_observation))
            arm_joint_targets = _align_action_to_motor(arm_joint_targets, context.alignment)
            joint_action.update(_apply_prefix(arm_joint_targets, context.arm))

        robot.send_action(joint_action)

        # Drain the one-shot teleop event flags so the recording layer
        # can react if the operator presses terminate / success /
        # rerecord. We don't act on them here — the entry point is
        # teleop-only — but draining keeps the flags from re-firing.
        teleop.get_teleop_events()

        precise_sleep(max(period - (time.perf_counter() - loop_start), 0.0))


def _strip_prefix(values: dict[str, object], arm: str) -> dict[str, object]:
    prefix = f"{arm}_"
    return {key[len(prefix) :]: value for key, value in values.items() if key.startswith(prefix)}


def _apply_prefix(values: dict[str, object], arm: str) -> dict[str, object]:
    prefix = f"{arm}_"
    return {f"{prefix}{key}": value for key, value in values.items()}


# ─────────────────────────────────────────────────────────────────────
# Alignment + diagnostics
# ─────────────────────────────────────────────────────────────────────


def _align_observation_to_urdf(
    arm_observation: dict[str, object],
    alignment: JointAlignment,
) -> dict[str, object]:
    """Rewrite ``*.pos`` keys from motor frame into URDF frame in-place-safely.

    All non-``.pos`` entries (images, velocity, gripper) are forwarded
    untouched, and gripper is bypass-listed so its 0..100 scale is not
    mangled by the degree-domain alignment.
    """

    aligned = dict(arm_observation)
    for joint_name, motor_deg in _iter_joint_positions(arm_observation):
        if joint_name in _ALIGNMENT_BYPASS_JOINTS:
            continue
        aligned[f"{joint_name}.pos"] = alignment.motor_to_urdf(joint_name, float(motor_deg))
    return aligned


def _align_action_to_motor(
    arm_action: dict[str, object],
    alignment: JointAlignment,
) -> dict[str, object]:
    """Rewrite IK-emitted ``*.pos`` joint targets back into motor frame."""

    aligned = dict(arm_action)
    for joint_name, urdf_deg in _iter_joint_positions(arm_action):
        if joint_name in _ALIGNMENT_BYPASS_JOINTS:
            continue
        aligned[f"{joint_name}.pos"] = alignment.urdf_to_motor(joint_name, float(urdf_deg))
    return aligned


def _iter_joint_positions(values: dict[str, object]):
    """Yield ``(joint_name, value)`` pairs for every ``<joint>.pos`` entry."""

    for key, value in values.items():
        if not isinstance(key, str) or not key.endswith(".pos"):
            continue
        joint_name = key[: -len(".pos")]
        if value is None:
            continue
        yield joint_name, value


def _log_alignment(contexts: Sequence[ArmContext]) -> None:
    for context in contexts:
        if context.alignment.sign or context.alignment.zero_offset_deg:
            LOG.info(
                "rfabric_remote: %s arm joint alignment — signs=%s zero_offsets_deg=%s",
                context.arm,
                dict(sorted(context.alignment.sign.items())),
                dict(sorted(context.alignment.zero_offset_deg.items())),
            )
        else:
            LOG.info(
                "rfabric_remote: %s arm joint alignment — identity (no overrides)",
                context.arm,
            )


def _log_axis_convention(args: argparse.Namespace) -> None:
    """Print the active control gains on startup.

    There is no IK in the control path: each operator key/button is
    bound to one motor with a fixed sign, and the per-tick servo step
    is a simple ``q_target = q_present + velocity * step``.
    """

    LOG.info(
        "rfabric_remote: per-servo joint-velocity teleop "
        "(joint_step=%.2f deg/tick, gripper_step=%.1f/tick, "
        "lead_cap=%.1f° @ %d Hz)",
        float(args.joint_step_deg),
        float(args.gripper_step),
        float(args.target_lead_cap_deg),
        int(args.fps),
    )


def _show_rest_pose(args: argparse.Namespace) -> None:
    """Render the URDF at ``q=0`` to a PNG and open it natively.

    For ``so101_new_calib.urdf`` (the bridge default) ``q=0`` is the
    *mechanical mid-of-range* pose — the same configuration lerobot's
    calibration ritual registers as zero. Pose each physical arm to
    match the rendered image, then run ``--probe-joints`` to read off
    any residual offsets.

    A static three-quarter render is intentional: a live MuJoCo viewer
    needs ``mjpython`` on macOS (and a browser-served meshcat is
    flakey behind aggressive HTTP filters). A PNG opens reliably in
    every OS image viewer with zero extra runtime requirements.
    """

    try:
        import mujoco
        import numpy as np
    except ImportError as err:
        raise SystemExit(
            "--show-rest-pose needs the `mujoco` package. Install it "
            "into the lerobot venv: `pip install mujoco>=3.0`."
        ) from err

    urdf_path = Path(args.urdf).resolve()
    mjcf_path = urdf_path.with_suffix(".xml")
    if not mjcf_path.is_file():
        raise SystemExit(
            f"Expected MuJoCo description next to the URDF (so the rest "
            f"pose render uses the same kinematics): {mjcf_path}\n"
            "Use the SO-ARM100 fork that ships both files together."
        )

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    try:
        pixels = _render_rest_pose(mujoco, np, model, data)
    except Exception as err:  # mujoco.cgl.cgl.CGLError, OSError, etc.
        # Headless / SSH / sandboxed shells can't open a CoreGraphics
        # context. Fall back to a numerical summary so the operator can
        # still reason about the pose without an image.
        LOG.warning("rfabric_remote: offscreen render failed (%s); printing FK summary instead", err)
        _print_rest_pose_summary(mujoco, model, data)
        return

    output_path = Path(tempfile.gettempdir()) / "rfabric_so101_rest_pose.png"
    _write_png(output_path, pixels)
    _open_with_default_app(output_path)

    print(
        f"rfabric_remote: rendered {mjcf_path.name} at q=0 (URDF rest pose) "
        f"to {output_path}\n"
        "  • This is the pose lerobot's half-turn-homing calibrates to.\n"
        "  • Match each physical arm to the rendered model, then run\n"
        '    `make run-arms ARGS="--probe-joints"` to read residual '
        "offsets per joint."
    )


def _render_rest_pose(mujoco_mod, numpy_mod, model, data):
    """Render q=0 from a fixed three-quarter camera. Caller handles errors.

    The render target dimensions must not exceed the model's offscreen
    framebuffer (``vis.global_.offwidth/offheight``). Without overriding
    those, MuJoCo's default offscreen buffer is 640x480 — caps the
    resulting image quality. We bump the model-side limits here so we
    can render at a reasonable size without modifying the on-disk XML.
    """

    width, height = 1280, 960
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height

    renderer = mujoco_mod.Renderer(model, height=height, width=width)
    try:
        camera = mujoco_mod.MjvCamera()
        mujoco_mod.mjv_defaultCamera(camera)
        # Three-quarter front-right-above view so all six joints are
        # visually distinguishable at once.
        camera.lookat = numpy_mod.array([0.0, 0.0, 0.10])
        camera.distance = 0.55
        camera.azimuth = 135.0
        camera.elevation = -25.0
        renderer.update_scene(data, camera=camera)
        return renderer.render()
    finally:
        try:
            renderer.close()
        except AttributeError:
            # ``Renderer`` raises during ``__del__`` if construction
            # bailed before private attrs were set; nothing to clean up.
            pass


def _print_rest_pose_summary(mujoco_mod, model, data) -> None:
    """Headless-friendly fallback: dump joint axes + body positions at q=0."""

    print("rfabric_remote: q=0 FK summary (URDF rest pose, all joints at 0 rad)")
    print("  joint angles (rad):")
    for joint_index in range(model.njnt):
        name = (
            mujoco_mod.mj_id2name(model, mujoco_mod.mjtObj.mjOBJ_JOINT, joint_index) or f"joint_{joint_index}"
        )
        if name in _ALIGNMENT_BYPASS_JOINTS:
            continue
        print(f"    {name:>14s} = 0.000")
    print("  body world positions (m):")
    for body_index in range(model.nbody):
        name = mujoco_mod.mj_id2name(model, mujoco_mod.mjtObj.mjOBJ_BODY, body_index) or f"body_{body_index}"
        x, y, z = data.xpos[body_index]
        print(f"    {name:>20s}  x={x:+.4f}  y={y:+.4f}  z={z:+.4f}")
    print(
        "(Open the SO-ARM100 README at "
        "https://github.com/TheRobotStudio/SO-ARM100 for a labelled photo "
        "of this pose if you can't render an image here.)"
    )


def _write_png(path: Path, pixels) -> None:
    """Write an HxWx3 RGB ndarray to ``path`` without an extra image dep.

    Pillow is already pulled in by the `lerobot[kinematics]` dependency
    chain (matplotlib → Pillow), so this stays inside the existing
    install footprint instead of adding ``imageio``.
    """

    from PIL import Image

    Image.fromarray(pixels).save(path, format="PNG")


def _open_with_default_app(path: Path) -> None:
    """Open ``path`` in the OS default app, no-op if the platform tool
    is missing (CI / headless servers)."""

    import platform
    import subprocess

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        elif system == "Linux":
            subprocess.run(["xdg-open", str(path)], check=False)
        elif system == "Windows":
            subprocess.run(["cmd", "/c", "start", "", str(path)], check=False)
    except FileNotFoundError:
        # Caller already prints the path — operator can open it manually.
        pass


def _run_joint_probe(*, robot: BiSOFollower, args: argparse.Namespace) -> None:
    """Connect, dump motor-frame and URDF-frame joint readings, exit.

    Use this *after* a calibration run to verify each joint reads ~0°
    at the URDF rest pose, and to read off the offsets you'd plug into
    ``--<arm>-joint-zero-offset-deg``.
    """

    robot.connect(calibrate=False)
    try:
        if not robot.is_calibrated:
            left_path = robot.left_arm.calibration_fpath
            right_path = robot.right_arm.calibration_fpath
            raise SystemExit(
                "Both arms must already be calibrated for --probe-joints "
                f"(expected files missing or out of sync):\n  {left_path}\n  {right_path}\n"
                "Run the sidecar once without --probe-joints (answer the calibration "
                "prompts), or calibrate each follower with lerobot-calibrate, then "
                "re-run the probe. If you ever ran without --robot-id, remove stale "
                f"``{left_path.parent / 'None.json'}`` so it is not mistaken for a valid file."
            )
        observation = robot.get_observation()
        for arm in ARMS:
            sub_robot = robot.left_arm if arm == "left" else robot.right_arm
            motor_names = list(sub_robot.bus.motors.keys())
            alignment = _build_alignment(arm, motor_names, args)
            arm_observation = _strip_prefix(observation, arm)
            print(f"\n=== {arm} arm — motor frame (Present_Position, deg) ===")
            for motor in motor_names:
                value = arm_observation.get(f"{motor}.pos")
                print(
                    f"  {motor:>14s}: {float(value):+8.3f}"
                    if value is not None
                    else f"  {motor:>14s}:   <missing>"
                )
            if alignment.sign or alignment.zero_offset_deg:
                aligned = _align_observation_to_urdf(arm_observation, alignment)
                print(f"--- {arm} arm — URDF frame after alignment (deg) ---")
                for motor in motor_names:
                    if motor in _ALIGNMENT_BYPASS_JOINTS:
                        continue
                    value = aligned.get(f"{motor}.pos")
                    print(
                        f"  {motor:>14s}: {float(value):+8.3f}"
                        if value is not None
                        else f"  {motor:>14s}:   <missing>"
                    )
    finally:
        robot.disconnect()


# ─────────────────────────────────────────────────────────────────────
# Sign diagnostic
# ─────────────────────────────────────────────────────────────────────
#
# At URDF q=0 (the calibrated rest pose, gripper extended ≈+X), each
# non-gripper joint's URDF-positive direction has a known dominant
# physical effect on the gripper. Computed once via FK; used here to
# tell the operator exactly which way to hand-move every joint.
#
#   shoulder_pan  +20°  -> gripper to operator's RIGHT  (dy=-121mm)
#   shoulder_lift +20°  -> gripper DOWN                 (dz=-117mm)
#   elbow_flex    +20°  -> gripper DOWN                 (dz=-100mm)
#   wrist_flex    +20°  -> gripper DOWN                 (dz= -54mm)
#   wrist_roll    +20°  -> moving jaw LEFT + DOWN       (dy=+6, dz=-7mm)

# Each test is (joint, English instruction, Polish instruction). Both are
# printed; Polish below the English line so the operator has zero
# ambiguity but logs/screenshots stay readable to non-Polish reviewers.
_SIGN_TESTS: list[tuple[str, str, str]] = [
    (
        "shoulder_pan",
        "Rotate the WHOLE ARM (only the shoulder_pan / first joint) so the gripper swings to YOUR RIGHT.",
        "Obróć CAŁE ramię (tylko shoulder_pan / pierwszy przegub przy podstawie) tak, aby chwytak (końcówka) skręcił w PRAWO z Twojej perspektywy. Pozostałe przeguby trzymaj nieruchomo.",
    ),
    (
        "shoulder_lift",
        "Pitch the UPPER ARM at the shoulder so the gripper drops DOWN toward the desk.",
        "Pochyl GÓRNE ramię w stawie barkowym (drugi przegub od podstawy) tak, aby chwytak opadł W DÓŁ, w stronę blatu biurka. Łokieć i nadgarstek trzymaj nieruchomo.",
    ),
    (
        "elbow_flex",
        "Bend ONLY the elbow so the lower arm rotates DOWN and the gripper drops DOWN.",
        "Zegnij TYLKO przegub łokciowy (trzeci przegub) tak, aby przedramię obróciło się W DÓŁ i chwytak opadł W DÓŁ. Bark i nadgarstek trzymaj nieruchomo.",
    ),
    (
        "wrist_flex",
        "Pitch ONLY the wrist so the gripper tip rotates DOWN (gripper points more toward the desk).",
        "Pochyl TYLKO przegub nadgarstka (czwarty przegub, tuż przed chwytakiem) tak, aby końcówka chwytaka skierowała się W DÓŁ (bardziej w stronę blatu). Bark i łokieć trzymaj nieruchomo.",
    ),
    (
        "wrist_roll",
        "Roll ONLY the wrist (5th joint) so the moving (top) jaw rotates DOWN AND TO YOUR LEFT.",
        "Obróć TYLKO oś obrotu nadgarstka (piąty przegub – obraca cały chwytak wokół jego długiej osi) tak, aby ruchoma szczęka chwytaka obróciła się W DÓŁ I W LEWO z Twojej perspektywy.",
    ),
]

_SIGN_MIN_DELTA_DEG = 5.0


def _diagnose_signs(*, robot: BiSOFollower, args: argparse.Namespace) -> None:
    """Walk the operator through a per-joint sign check and emit the CLI flag.

    Workflow per arm:

    1. Connect with ``calibrate=False`` (calibration must already exist).
    2. Disable torque on the arm's bus so joints move freely by hand.
    3. For each non-gripper joint, prompt the operator to move the joint
       in its URDF-positive direction (described by gripper motion).
    4. Read motor before/after; if the reading increased, the motor's
       direction matches the URDF axis (sign = +1); if it decreased,
       the motor is mounted reversed (sign = -1).
    5. Re-enable torque before moving on.

    Hands-on. No commands are ever sent to the motors during the
    diagnostic — only torque is toggled.
    """

    robot.connect(calibrate=False)
    try:
        if not robot.is_calibrated:
            left_path = robot.left_arm.calibration_fpath
            right_path = robot.right_arm.calibration_fpath
            raise SystemExit(
                "Both arms must already be calibrated for --diagnose-signs:\n"
                f"  {left_path}\n  {right_path}\n"
                "Run the sidecar once without --diagnose-signs to calibrate, then re-run."
            )

        print(_diagnose_intro())
        _wait_for_enter("Press Enter when both arms are at the calibration pose")

        per_arm_signs: dict[str, dict[str, int]] = {}
        for arm in ARMS:
            per_arm_signs[arm] = _diagnose_single_arm(robot=robot, arm=arm)
        _print_diagnose_summary(per_arm_signs)
    finally:
        robot.disconnect()


def _diagnose_intro() -> str:
    return (
        "\nrfabric_remote: per-joint sign diagnostic / diagnostyka znaków przegubów\n"
        "  EN  • Pose BOTH arms at the calibration pose (URDF rest pose) —\n"
        "         the same pose you just verified against `--show-rest-pose`.\n"
        "       • Torque is disabled per arm so you can hand-move every joint.\n"
        "       • Move only the named joint, in the direction the prompt describes.\n"
        "       • Reading goes UP   -> sign = +1 (motor matches URDF axis).\n"
        "       • Reading goes DOWN -> sign = -1 (motor mounted reversed).\n"
        "  PL  • Ustaw OBA ramiona w pozycji kalibracyjnej (taką jak na PNG\n"
        "         z `--show-rest-pose`).\n"
        "       • Moment obrotowy zostanie wyłączony osobno na każdym ramieniu,\n"
        "         więc będziesz mógł poruszać przegubami ręcznie.\n"
        "       • Poruszaj TYLKO wskazanym przegubem, w kierunku z opisu.\n"
        "       • Odczyt rośnie  -> znak = +1 (silnik zgodny z URDF).\n"
        "       • Odczyt maleje  -> znak = -1 (silnik zamontowany odwrotnie)."
    )


def _diagnose_single_arm(*, robot: BiSOFollower, arm: str) -> dict[str, int]:
    sub_robot = robot.left_arm if arm == "left" else robot.right_arm
    bus = sub_robot.bus

    arm_pl = "lewe" if arm == "left" else "prawe"
    print(f"\n========== {arm.upper()} ARM / {arm_pl.upper()} RAMIĘ ==========")
    print(f"  EN  Disabling torque on the {arm} arm — joints are now free to move by hand.")
    print(
        f"  PL  Wyłączam moment obrotowy na {arm_pl} ramieniu — przeguby są teraz luźne, możesz nimi poruszać."
    )
    bus.disable_torque()
    try:
        signs: dict[str, int] = {}
        for joint, instruction_en, instruction_pl in _SIGN_TESTS:
            signs[joint] = _diagnose_single_joint(
                sub_robot=sub_robot,
                arm=arm,
                joint=joint,
                instruction_en=instruction_en,
                instruction_pl=instruction_pl,
            )
        return signs
    finally:
        print(f"\n  EN  Re-enabling torque on the {arm} arm.")
        print(f"  PL  Włączam z powrotem moment obrotowy na {arm_pl} ramieniu.")
        bus.enable_torque()


def _diagnose_single_joint(
    *,
    sub_robot,
    arm: str,
    joint: str,
    instruction_en: str,
    instruction_pl: str,
) -> int:
    while True:
        before = float(sub_robot.get_observation()[f"{joint}.pos"])
        print(f"\n  [{arm} {joint}]")
        print(f"    EN  {instruction_en}")
        print(f"    PL  {instruction_pl}")
        print(f"    current reading / aktualny odczyt: {before:+.2f}°")
        _wait_for_enter("    Press Enter when done / Naciśnij Enter, gdy skończysz")
        after = float(sub_robot.get_observation()[f"{joint}.pos"])
        delta = after - before
        print(f"    new reading / nowy odczyt:        {after:+.2f}°  (delta {delta:+.2f}°)")
        if abs(delta) < _SIGN_MIN_DELTA_DEG:
            print(
                f"    EN  motion too small (need >= {_SIGN_MIN_DELTA_DEG}° change). Try again.\n"
                f"    PL  ruch zbyt mały (potrzeba co najmniej {_SIGN_MIN_DELTA_DEG}° zmiany). Spróbuj jeszcze raz."
            )
            continue
        sign = 1 if delta > 0 else -1
        if sign == 1:
            verdict = "OK (sign = +1)  /  OK – silnik zgodny z URDF"
        else:
            verdict = "REVERSED (sign = -1)  /  ODWRÓCONY – silnik zamontowany w przeciwną stronę"
        print(f"    -> {verdict}")
        return sign


def _print_diagnose_summary(per_arm_signs: dict[str, dict[str, int]]) -> None:
    print("\n========== SUMMARY / PODSUMOWANIE ==========")
    flags: list[str] = []
    for arm in ARMS:
        arm_pl = "lewe" if arm == "left" else "prawe"
        reversed_joints = [joint for joint, sign in per_arm_signs[arm].items() if sign == -1]
        if not reversed_joints:
            print(f"  {arm} / {arm_pl}: every joint matches the URDF axis — no override needed.")
            print(f"  {arm} / {arm_pl}: wszystkie przeguby zgodne z URDF — żadnych poprawek nie trzeba.")
            continue
        flag_value = ",".join(f"{joint}=-1" for joint in reversed_joints)
        flag = f"--{arm}-joint-sign={flag_value}"
        flags.append(flag)
        print(f"  {arm} / {arm_pl}: reversed joints / przeguby odwrócone -> {reversed_joints}")
        print(f"           flag:  {flag}")

    if flags:
        joined = " ".join(flags)
        print("\n  EN  Append these flags to your run command, e.g.:")
        print(f"        make run-arms ARGS='{joined}'")
        print("\n  PL  Dodaj te flagi do polecenia uruchomienia, np.:")
        print(f"        make run-arms ARGS='{joined}'")
    else:
        print(
            "\n  EN  No overrides necessary. The wildness must be elsewhere — try\n"
            "        --ik-orientation-weight=0.05 or raise --max-joint-step-deg.\n"
            "  PL  Żadnych poprawek znaków nie trzeba. Problem dziwnego ruchu jest\n"
            "        gdzie indziej — spróbuj --ik-orientation-weight=0.05 lub zwiększ\n"
            "        --max-joint-step-deg."
        )


def _wait_for_enter(prompt: str) -> None:
    """Block on stdin Enter; abort cleanly on Ctrl-C / Ctrl-D."""

    try:
        input(f"{prompt}: ")
    except (KeyboardInterrupt, EOFError):
        raise SystemExit("\nrfabric_remote: diagnostic aborted.") from None


if __name__ == "__main__":
    main()
