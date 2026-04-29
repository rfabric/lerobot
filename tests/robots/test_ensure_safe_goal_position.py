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

import pytest

from lerobot.robots.utils import ensure_safe_goal_position


def test_no_clamp_when_within_cap():
    out = ensure_safe_goal_position(
        {"j1": (5.0, 0.0), "j2": (-3.0, 1.0)},
        max_relative_target=10.0,
    )
    assert out == {"j1": 5.0, "j2": -3.0}


def test_uniform_scale_preserves_joint_direction():
    """All joints shrink by the same factor when any joint exceeds its cap."""

    out = ensure_safe_goal_position(
        {
            "a": (100.0, 0.0),
            "b": (5.0, 0.0),
        },
        max_relative_target=10.0,
    )
    assert out["a"] == pytest.approx(10.0)
    assert out["b"] == pytest.approx(0.5)


def test_uniform_scale_with_negative_delta():
    out = ensure_safe_goal_position(
        {
            "a": (-30.0, 0.0),
            "b": (-6.0, 0.0),
        },
        max_relative_target=10.0,
    )
    assert out["a"] == pytest.approx(-10.0)
    assert out["b"] == pytest.approx(-2.0)


def test_per_joint_caps_dict():
    out = ensure_safe_goal_position(
        {"a": (50.0, 0.0), "b": (4.0, 0.0)},
        max_relative_target={"a": 10.0, "b": 2.0},
    )
    assert out["a"] == pytest.approx(10.0)
    assert out["b"] == pytest.approx(0.8)
