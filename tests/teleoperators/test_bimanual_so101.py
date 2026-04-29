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
"""Tests for the bimanual SO-101 entry point's alignment + parsing helpers.

These exercise the pure-math reconciliation layer without standing up
any placo solver, hardware bus, or UDS — everything in here is in-memory
data manipulation.
"""

import argparse

import pytest

from lerobot.utils.import_utils import is_package_available

if not is_package_available("cbor2"):
    pytest.skip("cbor2 not available; install lerobot[rfabric] extra", allow_module_level=True)

from lerobot.teleoperators.rfabric_remote.bimanual_so101 import (
    JointAlignment,
    _align_action_to_motor,
    _align_observation_to_urdf,
    _parse_joint_floats,
    _parse_joint_signs,
)


class TestJointAlignment:
    def test_identity_is_a_passthrough(self) -> None:
        identity = JointAlignment.identity()
        assert identity.motor_to_urdf("shoulder_pan", 12.5) == 12.5
        assert identity.urdf_to_motor("shoulder_pan", -7.0) == -7.0

    def test_motor_to_urdf_subtracts_offset_then_flips_sign(self) -> None:
        alignment = JointAlignment(
            sign={"shoulder_pan": -1},
            zero_offset_deg={"shoulder_pan": 5.0},
        )
        # Motor reads 12° at URDF rest +5°. After subtracting the
        # offset the URDF would expect +7°; the mounted-reverse motor
        # then negates it to -7°.
        assert alignment.motor_to_urdf("shoulder_pan", 12.0) == pytest.approx(-7.0)

    def test_motor_urdf_round_trip(self) -> None:
        alignment = JointAlignment(
            sign={"shoulder_pan": -1, "wrist_roll": -1},
            zero_offset_deg={"shoulder_pan": 5.0, "elbow_flex": -3.5},
        )
        for joint in ("shoulder_pan", "wrist_roll", "elbow_flex", "untouched"):
            for motor_deg in (-90.0, -10.0, 0.0, 10.0, 87.5):
                urdf_deg = alignment.motor_to_urdf(joint, motor_deg)
                assert alignment.urdf_to_motor(joint, urdf_deg) == pytest.approx(motor_deg)


class TestObservationAlignment:
    def test_gripper_is_bypassed(self) -> None:
        alignment = JointAlignment(
            sign={"gripper": -1},
            zero_offset_deg={"gripper": 50.0},
        )
        observation = {
            "shoulder_pan.pos": 10.0,
            "gripper.pos": 25.0,
            "image.front": object(),
        }
        result = _align_observation_to_urdf(observation, alignment)
        # Non-pos entries pass through untouched.
        assert result["image.front"] is observation["image.front"]
        # Gripper is intentionally bypass-listed, so its value is preserved.
        assert result["gripper.pos"] == 25.0

    def test_observation_and_action_round_trip_through_alignment(self) -> None:
        alignment = JointAlignment(
            sign={"shoulder_pan": -1},
            zero_offset_deg={"shoulder_pan": 5.0, "elbow_flex": -3.5},
        )
        motor_observation = {
            "shoulder_pan.pos": 12.0,
            "elbow_flex.pos": -10.0,
            "gripper.pos": 30.0,
            "shoulder_pan.vel": 0.1,  # non-pos entries should pass through
        }
        urdf_observation = _align_observation_to_urdf(motor_observation, alignment)
        assert urdf_observation["shoulder_pan.pos"] == pytest.approx(-7.0)
        assert urdf_observation["elbow_flex.pos"] == pytest.approx(-6.5)
        assert urdf_observation["gripper.pos"] == 30.0
        assert urdf_observation["shoulder_pan.vel"] == 0.1

        # Pretend the IK pipeline returns the same URDF-frame joint
        # targets (no motion). After mapping back to motor frame we
        # should recover the original motor-frame readings on the
        # joints we touched.
        action = {
            "shoulder_pan.pos": urdf_observation["shoulder_pan.pos"],
            "elbow_flex.pos": urdf_observation["elbow_flex.pos"],
            "gripper.pos": 42.0,
        }
        motor_action = _align_action_to_motor(action, alignment)
        assert motor_action["shoulder_pan.pos"] == pytest.approx(12.0)
        assert motor_action["elbow_flex.pos"] == pytest.approx(-10.0)
        # Gripper is bypass-listed both ways.
        assert motor_action["gripper.pos"] == 42.0


class TestArgumentParsers:
    def test_parse_joint_floats_handles_whitespace_and_signs(self) -> None:
        parsed = _parse_joint_floats(" shoulder_pan = 15.5 , elbow_flex=-3.0 ")
        assert parsed == {"shoulder_pan": 15.5, "elbow_flex": -3.0}

    def test_parse_joint_floats_rejects_malformed_tokens(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_joint_floats("not_a_pair")

    def test_parse_joint_floats_rejects_non_numeric_value(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_joint_floats("shoulder_pan=abc")

    def test_parse_joint_signs_only_accepts_plus_or_minus_one(self) -> None:
        parsed = _parse_joint_signs("shoulder_pan=-1,wrist_roll=1")
        assert parsed == {"shoulder_pan": -1, "wrist_roll": 1}

    def test_parse_joint_signs_rejects_two(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_joint_signs("shoulder_pan=2")
