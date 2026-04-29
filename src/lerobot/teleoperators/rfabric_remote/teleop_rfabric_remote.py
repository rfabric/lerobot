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
"""Teleoperator that bridges the rFabric agent's UDS control channel
into ``lerobot-teleoperate``.

Topology (see ``REAL_TIME_CONTROL_BRIDGE.md``):

::

    operator UI ──▶ LiveKit ──▶ rfabric-remote agent (Rust) ──▶ UDS socket
                                                                      │
                                                                      ▼
                                                       RFabricRemoteTeleop
                                                                      │
                                                                      ▼
                                                       Robot.send_action(action)

This class is the *server* side of the UDS — the agent connects in as
the client. The teleoperator binds the socket, accepts at most one
client at a time, decodes inbound CBOR ``Frame``s into lerobot-shaped
action dicts, and mirrors observations back as ``state`` frames at
``state_publish_hz``. All socket I/O lives on background daemon threads
so ``get_action`` / ``send_feedback`` stay non-blocking and fit cleanly
inside ``lerobot-teleoperate``'s synchronous loop.

Bimanual semantics: every ``control`` frame may carry an ``arm`` tag
(``"left"`` / ``"right"``); the teleoperator keeps a separate latest-frame
slot per arm and merges them at ``get_action`` time, so an alternating
left/right operator stream never starves either arm.
"""

from __future__ import annotations

import logging
import os
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Any

from lerobot.types import RobotAction, RobotObservation

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .action_mapping import (
    PAYLOAD_HOME,
    PAYLOAD_STOP,
    MappingState,
    hold_ee_pose_delta,
    hold_joint_action,
    hold_joint_velocity,
    map_payload_to_ee_pose_delta,
    map_payload_to_joint_action,
    map_payload_to_joint_velocity,
)
from .configuration_rfabric_remote import RFabricRemoteTeleopConfig
from .wire import (
    ACK_KIND,
    CONTROL_KIND,
    MODE_KIND,
    STATE_KIND,
    Frame,
    FrameError,
    now_ts_ns,
    read_frame,
    write_frame,
)

LOG = logging.getLogger(__name__)

ACTION_MODE_EE_POSE_DELTA = "ee_pose_delta"
ACTION_MODE_JOINT_DIRECT = "joint_direct"
ACTION_MODE_JOINT_VELOCITY = "joint_velocity"
_VALID_ACTION_MODES = frozenset(
    {
        ACTION_MODE_EE_POSE_DELTA,
        ACTION_MODE_JOINT_DIRECT,
        ACTION_MODE_JOINT_VELOCITY,
    }
)

# Single-arm setups use the empty-string arm tag in the per-arm slot
# map, so the bimanual and single-arm code paths stay identical.
_DEFAULT_ARM_TAG = ""


class RFabricRemoteTeleop(Teleoperator):
    """Adapter teleoperator for the rFabric remote intervention bridge."""

    config_class = RFabricRemoteTeleopConfig
    name = "rfabric_remote"

    def __init__(self, config: RFabricRemoteTeleopConfig) -> None:
        super().__init__(config)
        self.config = config
        if config.action_mode not in _VALID_ACTION_MODES:
            raise ValueError(
                f"unknown action_mode {config.action_mode!r}; expected one of {sorted(_VALID_ACTION_MODES)}"
            )
        self._listener: socket.socket | None = None
        self._client_lock = threading.Lock()
        self._client_socket: socket.socket | None = None
        self._stop_event = threading.Event()
        self._accept_thread: threading.Thread | None = None
        self._state_thread: threading.Thread | None = None

        # Per-arm latest control frame + monotonic arrival timestamp.
        # The watchdog runs independently per arm so an alternating
        # left/right operator stream never starves either arm.
        self._control_lock = threading.Lock()
        self._latest_control: dict[str, tuple[Frame, float]] = {}

        self._observation_lock = threading.Lock()
        self._latest_observation: RobotObservation | None = None
        self._observation_seq = 0

        self._mode_events: queue.Queue[Frame] = queue.Queue()

        self._mapping_state = MappingState.empty()

        # Sticky teleop-event flags drained by ``get_teleop_events``.
        # ``terminate / success / rerecord`` are one-shot edges;
        # ``is_intervention`` is recomputed dynamically from the
        # freshness of any per-arm control frame.
        self._terminate_episode = False
        self._success = False
        self._rerecord_episode = False

    # ------------------------------------------------------------------
    # Teleoperator interface
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict:
        # Generic frame relay — the action shape depends on the
        # configured action_mode and the operator-emitted payloads.
        return {"dtype": "float32", "shape": (1,), "names": {"action": 0}}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._listener is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002 - parameter required by base class
        if self.is_connected:
            return
        self._stop_event.clear()
        self._open_listener()
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="rfabric_remote_accept",
            daemon=True,
        )
        self._accept_thread.start()
        self._state_thread = threading.Thread(
            target=self._state_publish_loop,
            name="rfabric_remote_state",
            daemon=True,
        )
        self._state_thread.start()
        LOG.info(
            "rfabric_remote: listening on %s (action_mode=%s, watchdog=%dms, state_hz=%.1f)",
            self.config.socket_path,
            self.config.action_mode,
            self.config.watchdog_ms,
            self.config.state_publish_hz,
        )

    def disconnect(self) -> None:
        if not self.is_connected and not self._stop_event.is_set():
            return
        self._stop_event.set()
        self._close_client()
        listener = self._listener
        self._listener = None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
            socket_path = Path(self.config.socket_path)
            try:
                socket_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as err:
                LOG.warning("rfabric_remote: failed to remove socket file %s: %s", socket_path, err)
        for thread in (self._accept_thread, self._state_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)
        self._accept_thread = None
        self._state_thread = None

    def get_action(self) -> RobotAction:
        # Snapshot the per-arm slots while holding the lock; do the
        # mapping work without it so the network thread is never
        # blocked on a heavy translation step.
        with self._control_lock:
            snapshot = dict(self._latest_control)
        active_arms = self._iter_arms()
        action: RobotAction = {}
        for arm in active_arms:
            entry = snapshot.get(arm)
            if entry is None:
                action.update(self._hold_action_for(arm))
                continue
            frame, received_at = entry
            age_ms = (time.monotonic() - received_at) * 1000.0
            if age_ms > self.config.watchdog_ms:
                action.update(self._hold_action_for(arm))
                continue
            action.update(self._map_frame(frame, arm))
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        with self._observation_lock:
            self._latest_observation = feedback

    def get_teleop_events(self) -> dict[str, Any]:
        with self._control_lock:
            snapshot = dict(self._latest_control)
        is_intervention = False
        now = time.monotonic()
        for entry in snapshot.values():
            _, received_at = entry
            age_ms = (now - received_at) * 1000.0
            if age_ms <= self.config.watchdog_ms:
                is_intervention = True
                break

        events = {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: self._terminate_episode,
            TeleopEvents.SUCCESS: self._success,
            TeleopEvents.RERECORD_EPISODE: self._rerecord_episode,
        }
        # One-shot flags — clear after a single read so the recording
        # pipeline can treat them as edges.
        self._terminate_episode = False
        self._success = False
        self._rerecord_episode = False
        return events

    # ------------------------------------------------------------------
    # Action shape dispatch
    # ------------------------------------------------------------------

    def _iter_arms(self) -> list[str]:
        if self.config.arms:
            return list(self.config.arms)
        return [_DEFAULT_ARM_TAG]

    def _map_frame(self, frame: Frame, arm: str) -> RobotAction:
        if self.config.action_mode == ACTION_MODE_EE_POSE_DELTA:
            return map_payload_to_ee_pose_delta(frame.payload, arm)
        if self.config.action_mode == ACTION_MODE_JOINT_VELOCITY:
            return map_payload_to_joint_velocity(frame.payload, arm)
        return map_payload_to_joint_action(
            frame.payload,
            self._mapping_state,
            self.config.motion_scale,
            arm,
        )

    def _hold_action_for(self, arm: str) -> RobotAction:
        if self.config.action_mode == ACTION_MODE_EE_POSE_DELTA:
            return hold_ee_pose_delta(arm)
        if self.config.action_mode == ACTION_MODE_JOINT_VELOCITY:
            return hold_joint_velocity(arm)
        return hold_joint_action(self._mapping_state, arm)

    # ------------------------------------------------------------------
    # UDS lifecycle
    # ------------------------------------------------------------------

    def _open_listener(self) -> None:
        socket_path = Path(self.config.socket_path)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        # The previous run may have crashed before unlinking; clean up
        # so ``bind`` does not return EADDRINUSE.
        if socket_path.exists():
            try:
                socket_path.unlink()
            except OSError as err:
                LOG.warning("rfabric_remote: cannot remove stale socket %s: %s", socket_path, err)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.setblocking(True)
        listener.bind(str(socket_path))
        os.chmod(str(socket_path), self.config.socket_mode)
        listener.listen(1)
        self._listener = listener

    def _accept_loop(self) -> None:
        listener = self._listener
        if listener is None:
            return
        backoff_ms = self.config.accept_min_backoff_ms
        while not self._stop_event.is_set():
            try:
                client_sock, _ = listener.accept()
            except OSError as err:
                if self._stop_event.is_set():
                    return
                LOG.warning("rfabric_remote: accept failed (%s); backing off %dms", err, backoff_ms)
                self._stop_event.wait(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 2, self.config.accept_max_backoff_ms)
                continue
            backoff_ms = self.config.accept_min_backoff_ms
            self._handle_client(client_sock)

    def _handle_client(self, client_sock: socket.socket) -> None:
        with self._client_lock:
            self._client_socket = client_sock
        client_sock.settimeout(None)
        LOG.info("rfabric_remote: agent connected")
        try:
            self._send_capabilities(client_sock)
            while not self._stop_event.is_set():
                try:
                    frame = read_frame(client_sock)
                except FrameError as err:
                    LOG.warning("rfabric_remote: dropping malformed frame: %s", err)
                    break
                if frame is None:
                    LOG.info("rfabric_remote: agent disconnected (clean EOF)")
                    break
                self._on_inbound_frame(frame)
        finally:
            self._close_client()

    def _close_client(self) -> None:
        with self._client_lock:
            client_sock = self._client_socket
            self._client_socket = None
        if client_sock is not None:
            try:
                client_sock.close()
            except OSError:
                pass

    def _send_capabilities(self, client_sock: socket.socket) -> None:
        payload = {
            "kind": "capabilities",
            "accepts": list(self.config.accepts),
            "arms": list(self.config.arms),
        }
        frame = Frame(kind=MODE_KIND, seq=0, ts_ns=now_ts_ns(), payload=payload)
        try:
            write_frame(client_sock, frame)
        except (OSError, FrameError) as err:
            LOG.warning("rfabric_remote: failed to publish capability frame: %s", err)

    # ------------------------------------------------------------------
    # Inbound frame handling
    # ------------------------------------------------------------------

    def _on_inbound_frame(self, frame: Frame) -> None:
        if frame.kind == CONTROL_KIND:
            self._on_control_frame(frame)
        elif frame.kind in (MODE_KIND, ACK_KIND):
            self._on_mode_frame(frame)
        elif frame.kind == STATE_KIND:
            LOG.warning("rfabric_remote: unexpected `state` frame on inbound path")

    def _on_control_frame(self, frame: Frame) -> None:
        arm = _arm_tag_from_payload(frame.payload)
        with self._control_lock:
            self._latest_control[arm] = (frame, time.monotonic())
        # ``stop`` and ``home`` payloads are handled by the action
        # mapping on the next ``get_action`` tick — no episode-boundary
        # signal needed here.
        payload = frame.payload if isinstance(frame.payload, dict) else {}
        kind = payload.get("kind")
        if kind in (PAYLOAD_STOP, PAYLOAD_HOME):
            return

    def _on_mode_frame(self, frame: Frame) -> None:
        # Mode frames originating from the *operator* carry session
        # commands like episode terminate / success / rerecord. The
        # heartbeat the agent emits looks the same on the wire so we
        # filter on `payload.kind`.
        payload = frame.payload if isinstance(frame.payload, dict) else {}
        kind = payload.get("kind")
        if kind == "heartbeat":
            return  # noisy by design; nothing to do
        if kind == "terminate":
            self._terminate_episode = True
        elif kind == "success":
            self._terminate_episode = True
            self._success = True
        elif kind == "rerecord":
            self._terminate_episode = True
            self._rerecord_episode = True
        else:
            self._mode_events.put(frame)

    # ------------------------------------------------------------------
    # Outbound state publication
    # ------------------------------------------------------------------

    def _state_publish_loop(self) -> None:
        period = 1.0 / max(self.config.state_publish_hz, 0.1)
        while not self._stop_event.is_set():
            self._stop_event.wait(period)
            if self._stop_event.is_set():
                return
            with self._observation_lock:
                observation = self._latest_observation
            if observation is None:
                continue
            with self._client_lock:
                client_sock = self._client_socket
            if client_sock is None:
                continue
            self._observation_seq += 1
            frame = Frame(
                kind=STATE_KIND,
                seq=self._observation_seq,
                ts_ns=now_ts_ns(),
                payload=_observation_to_payload(observation),
            )
            try:
                write_frame(client_sock, frame)
            except (OSError, FrameError) as err:
                LOG.debug("rfabric_remote: state publish failed (%s); waiting for reconnect", err)
                self._close_client()


def _arm_tag_from_payload(payload: Any) -> str:
    """Extract the arm tag from a control payload.

    Returns ``""`` for un-tagged payloads so the single-arm path lands
    in the same per-arm slot as ``arms = []`` configurations.
    """

    if isinstance(payload, dict):
        arm = payload.get("arm")
        if isinstance(arm, str):
            return arm
    return _DEFAULT_ARM_TAG


def _observation_to_payload(observation: RobotObservation) -> dict[str, Any]:
    """Reduce an observation dict to a CBOR-friendly payload.

    Camera frames and other large tensors that travel through other
    surfaces (LiveKit video tracks, the ingest pipeline) are dropped
    here — only the proprioceptive scalar fields are mirrored back so
    the operator UI can render a live joint / pose overlay without the
    state channel becoming a bandwidth hotspot.
    """

    if not isinstance(observation, dict):
        return {"kind": "snapshot"}
    scalars: dict[str, float] = {}
    for key, value in observation.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            scalars[str(key)] = float(value)
    return {"kind": "snapshot", "values": scalars}
