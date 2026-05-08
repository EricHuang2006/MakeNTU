#!/usr/bin/env python3
"""
MakeNTU board-side demo server.

Runs on the i.MX93 / camera device.
- TCP video stream on port 9999: [8-byte !Q length][JPEG bytes]
- TCP command server on port 10000: one JSON command per line

This version keeps photo storage on the PC web bridge. The board only streams
frames and logs/acknowledges commands from the client.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import threading
import time
from typing import Any

import cv2


VIDEO_PORT = 9999
COMMAND_PORT = 10000
HEADER_FORMAT = "!Q"  # network byte order, unsigned 8-byte integer
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.capture_running = False
        self.photo_interval_sec = 5.0
        self.mode = "full_body"
        self.gesture_mode = False
        self.servo = {"pan": 0.0, "tilt": 0.0, "height": 0.0}
        self.quality = 75
        self.status = "idle"
        self.command_count = 0
        self.last_command: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "capture_running": self.capture_running,
                "photo_interval_sec": self.photo_interval_sec,
                "mode": self.mode,
                "gesture_mode": self.gesture_mode,
                "servo": dict(self.servo),
                "quality": self.quality,
                "status": self.status,
                "has_frame": self.latest_jpeg is not None,
                "command_count": self.command_count,
                "last_command": self.last_command,
                "photo_storage": "PC bridge, not board",
            }


def log_command(addr: tuple[str, int], command: dict[str, Any]) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[command] {timestamp} from {addr[0]}:{addr[1]} -> {json.dumps(command, sort_keys=True)}")


def camera_loop(state: SharedState, camera_index: int, width: int, height: int, fps: float) -> None:
    cap = cv2.VideoCapture(camera_index)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    frame_delay = 1.0 / fps if fps > 0 else 0.03
    print(f"[camera] started camera={camera_index}, size={width}x{height}, fps={fps}")

    while True:
        ok, frame = cap.read()
        if not ok:
            with state.lock:
                state.status = "camera_read_failed"
            time.sleep(0.2)
            continue

        with state.lock:
            quality = int(state.quality)
            capture_running = state.capture_running
            mode = state.mode
            gesture_mode = state.gesture_mode

        # Draw a small overlay so the demo state is visible in the stream.
        overlay = f"mode={mode} gesture={gesture_mode} pc_auto={capture_running}"
        cv2.putText(frame, overlay, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            continue

        with state.lock:
            state.latest_jpeg = encoded.tobytes()
            if state.status in ("idle", "camera_read_failed"):
                state.status = "streaming"

        time.sleep(frame_delay)


def video_client_thread(conn: socket.socket, addr: tuple[str, int], state: SharedState) -> None:
    print(f"[video] client connected: {addr}")
    try:
        while True:
            with state.lock:
                jpeg = state.latest_jpeg
            if jpeg is None:
                time.sleep(0.05)
                continue
            packet = struct.pack(HEADER_FORMAT, len(jpeg)) + jpeg
            conn.sendall(packet)
            time.sleep(0.01)
    except (ConnectionError, OSError):
        print(f"[video] client disconnected: {addr}")
    finally:
        conn.close()


def video_server(state: SharedState, host: str, port: int) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(4)
    print(f"[video] listening on {host}:{port}")
    while True:
        conn, addr = server.accept()
        threading.Thread(target=video_client_thread, args=(conn, addr, state), daemon=True).start()


def apply_settings(settings: dict[str, Any], state: SharedState) -> None:
    with state.lock:
        if "photo_interval_sec" in settings:
            state.photo_interval_sec = float(settings["photo_interval_sec"])
        if "mode" in settings:
            state.mode = str(settings["mode"])
        if "quality" in settings:
            state.quality = int(settings["quality"])
        if "gesture_mode" in settings:
            state.gesture_mode = bool(settings["gesture_mode"])
        if "servo" in settings and isinstance(settings["servo"], dict):
            for key in ("pan", "tilt", "height"):
                if key in settings["servo"]:
                    state.servo[key] = float(settings["servo"][key])


def handle_command(command: dict[str, Any], state: SharedState) -> dict[str, Any]:
    cmd_type = command.get("type")
    with state.lock:
        state.command_count += 1
        state.last_command = command

    if cmd_type == "start_capture":
        settings = command.get("settings", {}) or {}
        apply_settings(settings, state)
        with state.lock:
            state.capture_running = True
            state.status = "pc_auto_capture_running"
        return {
            "ok": True,
            "message": "start acknowledged; photos are saved by the PC bridge",
            "state": state.snapshot(),
        }

    if cmd_type == "stop_capture":
        with state.lock:
            state.capture_running = False
            state.status = "pc_auto_capture_stopped"
        return {"ok": True, "message": "stop acknowledged", "state": state.snapshot()}

    if cmd_type == "take_photo":
        with state.lock:
            state.status = "manual_photo_request_acknowledged"
        return {
            "ok": True,
            "message": "photo request acknowledged; PC bridge saves the latest streamed frame",
            "state": state.snapshot(),
        }

    if cmd_type == "set_settings":
        settings = command.get("settings", {}) or {}
        apply_settings(settings, state)
        with state.lock:
            state.status = "settings_updated"
        return {"ok": True, "message": "settings updated", "state": state.snapshot()}

    if cmd_type == "get_status":
        return {"ok": True, "state": state.snapshot()}

    return {"ok": False, "error": f"unknown command type: {cmd_type}"}


def command_client_thread(conn: socket.socket, addr: tuple[str, int], state: SharedState) -> None:
    file = conn.makefile("rwb")
    try:
        for raw_line in file:
            try:
                command = json.loads(raw_line.decode("utf-8"))
                log_command(addr, command)
                response = handle_command(command, state)
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            file.write((json.dumps(response) + "\n").encode("utf-8"))
            file.flush()
    except (ConnectionError, OSError):
        pass
    finally:
        conn.close()


def command_server(state: SharedState, host: str, port: int) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(8)
    print(f"[command] listening on {host}:{port}")
    print("[command] logging each JSON request from the PC bridge")
    while True:
        conn, addr = server.accept()
        threading.Thread(target=command_client_thread, args=(conn, addr, state), daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="MakeNTU board-side video/control demo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--video-port", type=int, default=VIDEO_PORT)
    parser.add_argument("--command-port", type=int, default=COMMAND_PORT)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    state = SharedState()

    threading.Thread(target=video_server, args=(state, args.host, args.video_port), daemon=True).start()
    threading.Thread(target=command_server, args=(state, args.host, args.command_port), daemon=True).start()

    # Keep camera loop on the main thread so camera errors are obvious.
    camera_loop(state, args.camera, args.width, args.height, args.fps)


if __name__ == "__main__":
    main()