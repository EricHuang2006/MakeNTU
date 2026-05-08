"""
Standalone MakeNTU video + control demo server.

It uses TWO TCP ports:
  9999  = actual camera video stream, server -> client
  10000 = JSON command channel, client -> server

Run on server/i.MX93 side:
  python3 demo_video_control_server.py

Optional:
  python3 demo_video_control_server.py --camera 0 --width 640 --height 480 --quality 70 --fps 15

This does NOT import or modify your main MakeNTU code.
"""

import argparse
import json
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2

HOST = "0.0.0.0"
VIDEO_PORT = 9999
COMMAND_PORT = 10000

state_lock = threading.Lock()
state: Dict[str, Any] = {
    "gesture_mode": False,
    "pose_mode": "full_body",
    "capture_running": False,
    "photo_interval_sec": 5.0,
    "save_dir": "captured_photos",
    "take_photo_requested": False,
    "manual_servo": None,
    "last_photo_path": None,
    "last_command": None,
    "running": True,
}

latest_frame_lock = threading.Lock()
latest_frame: Optional[Any] = None
latest_frame_time = 0.0


def send_length_prefixed(sock: socket.socket, payload: bytes) -> None:
    """Send one framed payload: 8-byte unsigned length + payload bytes."""
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def draw_overlay(frame, snapshot: Dict[str, Any]):
    """Draw demo status text onto preview frames."""
    overlay = frame.copy()
    lines = [
        f"capture_running: {snapshot['capture_running']}",
        f"gesture_mode: {snapshot['gesture_mode']}",
        f"pose_mode: {snapshot['pose_mode']}",
        f"photo_interval_sec: {snapshot['photo_interval_sec']}",
        f"last_photo: {snapshot['last_photo_path']}",
    ]
    y = 24
    for line in lines:
        cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        y += 24
    return overlay


def save_photo(frame, save_dir: str) -> str:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = time.strftime("photo_%Y%m%d_%H%M%S.jpg")
    path = str(Path(save_dir) / filename)
    cv2.imwrite(path, frame)
    return path


def camera_loop(camera_index: int, width: int, height: int, target_fps: float) -> None:
    """Owns the physical camera. Updates latest_frame and handles auto photo capture."""
    global latest_frame, latest_frame_time

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[camera] ERROR: could not open camera index {camera_index}")
        with state_lock:
            state["running"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    print(f"[camera] opened camera={camera_index}, requested={width}x{height}@{target_fps}fps")
    last_auto_photo_time = 0.0
    min_frame_delay = 1.0 / max(target_fps, 1.0)

    try:
        while True:
            loop_start = time.time()
            with state_lock:
                if not state["running"]:
                    break
                snapshot = dict(state)

            ok, frame = cap.read()
            if not ok or frame is None:
                print("[camera] WARNING: failed to read frame")
                time.sleep(0.05)
                continue

            now = time.time()

            # One-shot photo request from command channel.
            if snapshot["take_photo_requested"]:
                path = save_photo(frame, snapshot["save_dir"])
                with state_lock:
                    state["take_photo_requested"] = False
                    state["last_photo_path"] = path
                print(f"[camera] saved requested photo: {path}")

            # Automatic photo loop, started/stopped by UI/client command.
            if snapshot["capture_running"]:
                interval = max(float(snapshot["photo_interval_sec"]), 0.1)
                if now - last_auto_photo_time >= interval:
                    path = save_photo(frame, snapshot["save_dir"])
                    with state_lock:
                        state["last_photo_path"] = path
                    last_auto_photo_time = now
                    print(f"[camera] saved automatic photo: {path}")

            # Latest-frame-only preview. Video clients read this; old frames are dropped.
            with state_lock:
                overlay_snapshot = dict(state)
            preview = draw_overlay(frame, overlay_snapshot)
            with latest_frame_lock:
                latest_frame = preview
                latest_frame_time = now

            elapsed = time.time() - loop_start
            if elapsed < min_frame_delay:
                time.sleep(min_frame_delay - elapsed)
    finally:
        cap.release()
        print("[camera] stopped")


def handle_video_client(client: socket.socket, addr, jpeg_quality: int, stream_fps: float) -> None:
    print(f"[video] client connected: {addr}")
    min_frame_delay = 1.0 / max(stream_fps, 1.0)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    try:
        while True:
            start = time.time()
            with state_lock:
                if not state["running"]:
                    break

            with latest_frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()

            if frame is None:
                time.sleep(0.05)
                continue

            ok, encoded = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                print("[video] WARNING: JPEG encode failed")
                continue

            send_length_prefixed(client, encoded.tobytes())

            elapsed = time.time() - start
            if elapsed < min_frame_delay:
                time.sleep(min_frame_delay - elapsed)
    except (BrokenPipeError, ConnectionResetError, ConnectionError):
        print(f"[video] client disconnected: {addr}")
    finally:
        client.close()


def video_server(jpeg_quality: int, stream_fps: float) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, VIDEO_PORT))
        server.listen(5)
        print(f"[video] listening on {HOST}:{VIDEO_PORT}")
        while True:
            with state_lock:
                if not state["running"]:
                    break
            client, addr = server.accept()
            threading.Thread(
                target=handle_video_client,
                args=(client, addr, jpeg_quality, stream_fps),
                daemon=True,
            ).start()


def apply_command(command: Dict[str, Any]) -> Dict[str, Any]:
    cmd_type = command.get("type")

    with state_lock:
        state["last_command"] = command

        if cmd_type == "start_capture":
            settings = command.get("settings", {}) or {}
            if "photo_interval_sec" in settings:
                state["photo_interval_sec"] = float(settings["photo_interval_sec"])
            if "save_dir" in settings:
                state["save_dir"] = str(settings["save_dir"])
            if "gesture_mode" in settings:
                state["gesture_mode"] = bool(settings["gesture_mode"])
            if "pose_mode" in settings:
                state["pose_mode"] = str(settings["pose_mode"])
            state["capture_running"] = True
            return {"ok": True, "message": "automatic capture started", "state": dict(state)}

        if cmd_type == "stop_capture":
            state["capture_running"] = False
            return {"ok": True, "message": "automatic capture stopped", "state": dict(state)}

        if cmd_type == "set_gesture_mode":
            state["gesture_mode"] = bool(command.get("enabled", False))
            return {"ok": True, "gesture_mode": state["gesture_mode"]}

        if cmd_type == "set_pose_mode":
            mode = command.get("mode", "full_body")
            state["pose_mode"] = mode
            return {"ok": True, "pose_mode": mode}

        if cmd_type == "take_photo":
            state["take_photo_requested"] = True
            return {"ok": True, "message": "photo trigger queued"}

        if cmd_type == "set_servo":
            state["manual_servo"] = {
                "pan": command.get("pan"),
                "tilt": command.get("tilt"),
                "height": command.get("height"),
            }
            return {"ok": True, "manual_servo": state["manual_servo"]}

        if cmd_type == "get_state":
            return {"ok": True, "state": dict(state), "latest_frame_time": latest_frame_time}

        if cmd_type == "shutdown_demo":
            state["running"] = False
            return {"ok": True, "message": "demo server shutting down"}

        return {"ok": False, "error": f"unknown command type: {cmd_type}"}


def handle_command_client(client: socket.socket, addr) -> None:
    print(f"[command] client connected: {addr}")
    with client:
        file = client.makefile("rwb")
        while True:
            line = file.readline()
            if not line:
                break
            try:
                command = json.loads(line.decode("utf-8"))
                response = apply_command(command)
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            file.write((json.dumps(response) + "\n").encode("utf-8"))
            file.flush()
    print(f"[command] client disconnected: {addr}")


def command_server() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, COMMAND_PORT))
        server.listen(5)
        print(f"[command] listening on {HOST}:{COMMAND_PORT}")
        while True:
            with state_lock:
                if not state["running"]:
                    break
            client, addr = server.accept()
            threading.Thread(target=handle_command_client, args=(client, addr), daemon=True).start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MakeNTU actual video stream + command socket demo")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index, usually 0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15.0, help="Camera read target FPS")
    parser.add_argument("--stream-fps", type=float, default=15.0, help="Video socket send FPS")
    parser.add_argument("--quality", type=int, default=70, help="JPEG quality, 1-100")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.quality = max(1, min(100, args.quality))

    threading.Thread(
        target=camera_loop,
        args=(args.camera, args.width, args.height, args.fps),
        daemon=True,
    ).start()
    threading.Thread(target=video_server, args=(args.quality, args.stream_fps), daemon=True).start()
    command_server()
    print("[server] demo stopped")


if __name__ == "__main__":
    main()
