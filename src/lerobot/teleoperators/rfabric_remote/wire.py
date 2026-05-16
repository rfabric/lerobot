#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Compatibility shim: re-export the shared `rfabric-control-wire` surface.

The framing primitives previously defined inline in this module now live
in the standalone ``rfabric-control-wire`` package so the standalone
``rfabric-yam-bridge`` and any other follower can decode the same wire
without depending on lerobot. This module re-exports the names lerobot
internally uses so existing call sites keep working unchanged.
"""

try:
    from rfabric_control_wire import (
        ACK_KIND,
        CONTROL_KIND,
        LENGTH_PREFIX_FORMAT,
        LENGTH_PREFIX_SIZE,
        MAX_FRAME_BYTES,
        MODE_KIND,
        STATE_KIND,
        VALID_KINDS,
        Frame,
        FrameError,
        now_ts_ns,
        read_frame,
        write_frame,
    )
except ImportError as err:  # pragma: no cover - environment-dependent
    raise ImportError(
        "the rfabric_remote teleoperator requires the 'rfabric-control-wire' package; "
        "install with `pip install 'lerobot[rfabric]'` or "
        "`pip install rfabric-control-wire`."
    ) from err

__all__ = [
    "ACK_KIND",
    "CONTROL_KIND",
    "Frame",
    "FrameError",
    "LENGTH_PREFIX_FORMAT",
    "LENGTH_PREFIX_SIZE",
    "MAX_FRAME_BYTES",
    "MODE_KIND",
    "STATE_KIND",
    "VALID_KINDS",
    "now_ts_ns",
    "read_frame",
    "write_frame",
]
