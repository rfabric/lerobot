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
"""Wire-format helpers for the rFabric agent <-> lerobot UDS bridge.

The frame envelope mirrors ``agent-src/src/media/envelope.rs`` and
``app/src/frontend/components/intervention/envelope.ts`` in the rFabric
platform repo: a CBOR map keyed ``kind`` / ``seq`` / ``tsNs`` /
``payload``. On the wire each frame is preceded by a u32 little-endian
length prefix:

::

    [ u32 LE length ][ CBOR-encoded Frame ]

This module deliberately keeps the framing primitives separate from the
teleoperator implementation so the same helpers are usable by tests and
ad-hoc tooling.
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from typing import Any

try:
    import cbor2
except ImportError as err:  # pragma: no cover - environment-dependent
    raise ImportError(
        "the rfabric_remote teleoperator requires the 'cbor2' package; "
        "install with `pip install 'lerobot[rfabric]'` or `pip install cbor2`."
    ) from err

CONTROL_KIND = "control"
STATE_KIND = "state"
ACK_KIND = "ack"
MODE_KIND = "mode"

VALID_KINDS = frozenset({CONTROL_KIND, STATE_KIND, ACK_KIND, MODE_KIND})

LENGTH_PREFIX_FORMAT = "<I"  # u32 little-endian, matches the Rust agent
LENGTH_PREFIX_SIZE = 4
MAX_FRAME_BYTES = 64 * 1024


class FrameError(ValueError):
    """Raised when a frame cannot be parsed or violates contract bounds."""


@dataclass(frozen=True)
class Frame:
    """In-memory representation of a wire frame."""

    kind: str
    seq: int
    ts_ns: int
    payload: Any

    def encode(self) -> bytes:
        if self.kind not in VALID_KINDS:
            raise FrameError(f"unsupported frame kind {self.kind!r}")
        return cbor2.dumps(
            {
                "kind": self.kind,
                "seq": self.seq,
                "tsNs": self.ts_ns,
                "payload": self.payload,
            }
        )

    @classmethod
    def decode(cls, raw: bytes) -> Frame:
        if len(raw) > MAX_FRAME_BYTES:
            raise FrameError(f"frame size {len(raw)} exceeds maximum {MAX_FRAME_BYTES}")
        try:
            decoded = cbor2.loads(raw)
        except Exception as err:
            raise FrameError(f"CBOR decode failed: {err}") from err
        if not isinstance(decoded, dict):
            raise FrameError(f"frame must decode to a CBOR map; got {type(decoded).__name__}")
        kind = decoded.get("kind")
        if kind not in VALID_KINDS:
            raise FrameError(f"unsupported frame kind {kind!r}")
        seq = decoded.get("seq", 0)
        # Both the TypeScript operator and the Rust agent use camelCase
        # `tsNs` on the wire; older payloads sometimes ship snake_case.
        ts_ns = decoded.get("tsNs", decoded.get("ts_ns", 0))
        return cls(kind=kind, seq=int(seq), ts_ns=int(ts_ns), payload=decoded.get("payload"))


def now_ts_ns() -> int:
    """Wall-clock nanoseconds since the Unix epoch.

    The bridge tags every emitted frame with this so the receiver can
    measure end-to-end latency.
    """

    return time.time_ns()


def write_frame(sock: socket.socket, frame: Frame) -> None:
    """Write a length-prefixed CBOR frame onto a connected UDS."""

    raw = frame.encode()
    if len(raw) == 0 or len(raw) > MAX_FRAME_BYTES:
        raise FrameError(f"refusing to write frame of size {len(raw)}")
    sock.sendall(struct.pack(LENGTH_PREFIX_FORMAT, len(raw)) + raw)


def read_frame(sock: socket.socket) -> Frame | None:
    """Block until the next frame arrives. Returns ``None`` on clean EOF."""

    header = _recv_exact(sock, LENGTH_PREFIX_SIZE)
    if header is None:
        return None
    (length,) = struct.unpack(LENGTH_PREFIX_FORMAT, header)
    if length == 0 or length > MAX_FRAME_BYTES:
        raise FrameError(f"frame length {length} out of range")
    body = _recv_exact(sock, length)
    if body is None:
        raise FrameError("connection closed mid-frame")
    return Frame.decode(body)


def _recv_exact(sock: socket.socket, count: int) -> bytes | None:
    """Read exactly ``count`` bytes; return ``None`` on a clean EOF before any byte."""

    buffer = bytearray()
    while len(buffer) < count:
        try:
            chunk = sock.recv(count - len(buffer))
        except OSError:
            return None
        if not chunk:
            if not buffer:
                return None
            return None
        buffer.extend(chunk)
    return bytes(buffer)
