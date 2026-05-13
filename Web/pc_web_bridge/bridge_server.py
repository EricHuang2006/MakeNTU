#!/usr/bin/env python3
"""
MakeNTU PC web bridge.

Runs on your PC.
- Connects to the board video socket: board_ip:9999
- Connects to the board command socket: board_ip:10000
- Serves a browser UI at http://PC_IP:8000
- Converts board's raw JPEG stream into browser-friendly MJPEG at /video
- Saves manual and automatic captures on the PC, not on the board
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, jsonify, render_template, request, send_file


HEADER_FORMAT = "!Q"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class BridgeState:
    def __init__(
        self,
        board_ip: str,
        video_port: int,
        command_port: int,
        capture_dir: Path,
    ) -> None:
        self.board_ip = str(board_ip).strip()
        self.video_port = video_port
        self.command_port = command_port
        self.capture_dir = capture_dir
        self.capture_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.video_connected = False
        self.video_error: str | None = None
        self.last_frame_time = 0.0

        self.capture_running = False
        self.photo_interval_sec = 5.0
        self.last_capture_time = 0.0
        self.saved_count = 0
        self.last_saved_path: str | None = None
        self.last_settings: dict[str, Any] = {}

    def update_board_ip(self, board_ip: str) -> None:
        next_ip = str(board_ip).strip()
        if not next_ip:
            abort_json(400, "board_ip is required")
        with self.lock:
            self.board_ip = next_ip
            self.video_connected = False
            self.video_error = f"board IP updated to {next_ip}; reconnecting stream"

    def connection_target(self) -> tuple[str, int, int]:
        with self.lock:
            return self.board_ip, self.video_port, self.command_port

    def set_frame(self, jpeg: bytes) -> None:
        with self.lock:
            self.latest_jpeg = jpeg
            self.video_connected = True
            self.video_error = None
            self.last_frame_time = time.time()

    def set_video_error(self, message: str) -> None:
        with self.lock:
            self.video_connected = False
            self.video_error = message

    def save_latest_jpeg(self, reason: str) -> dict[str, Any]:
        with self.lock:
            jpeg = self.latest_jpeg
            frame_age = None if self.last_frame_time == 0 else time.time() - self.last_frame_time
            if jpeg is None:
                abort_json(409, "no video frame received yet")
            self.saved_count += 1
            count = self.saved_count

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}_{count:04d}_{reason}.jpg"
        path = self.capture_dir / filename
        path.write_bytes(jpeg)

        with self.lock:
            self.last_saved_path = str(path)
            self.last_capture_time = time.time()

        print(f"[bridge/photo] saved {path}")
        return {
            "ok": True,
            "message": "photo saved on PC bridge",
            "path": str(path),
            "filename": filename,
            "frame_age_sec": None if frame_age is None else round(frame_age, 3),
            "saved_count": count,
        }

    def list_captures(self, limit: int = 100) -> list[dict[str, Any]]:
        files = sorted(
            self.capture_dir.glob("*.jpg"),
            key=lambda file_path: file_path.stat().st_mtime,
            reverse=True,
        )
        items: list[dict[str, Any]] = []
        for file_path in files[:limit]:
            stat = file_path.stat()
            items.append(
                {
                    "filename": file_path.name,
                    "size_bytes": stat.st_size,
                    "modified_ts": stat.st_mtime,
                    "thumbnail_url": f"/captures/{file_path.name}",
                    "download_url": f"/captures/{file_path.name}?download=1",
                    "delete_url": f"/api/captures/{file_path.name}",
                }
            )
        return items

    def resolve_capture_path(self, filename: str) -> Path:
        path = (self.capture_dir / filename).resolve()
        capture_root = self.capture_dir.resolve()
        if path.parent != capture_root or not path.is_file():
            abort_json(404, "capture not found")
        return path

    def delete_capture(self, filename: str) -> dict[str, Any]:
        path = self.resolve_capture_path(filename)
        path.unlink()
        with self.lock:
            if self.last_saved_path == str(path):
                self.last_saved_path = None
        return {
            "ok": True,
            "message": "capture deleted",
            "filename": filename,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            age = None if self.last_frame_time == 0 else round(time.time() - self.last_frame_time, 2)
            return {
                "board_ip": self.board_ip,
                "video_port": self.video_port,
                "command_port": self.command_port,
                "video_connected": self.video_connected,
                "video_error": self.video_error,
                "has_frame": self.latest_jpeg is not None,
                "latest_frame_age_sec": age,
                "capture_running": self.capture_running,
                "photo_interval_sec": self.photo_interval_sec,
                "capture_dir": str(self.capture_dir),
                "saved_count": self.saved_count,
                "last_saved_path": self.last_saved_path,
                "last_settings": self.last_settings,
            }


def abort_json(status_code: int, message: str):
    response = jsonify({"ok": False, "detail": message})
    response.status_code = status_code
    abort(response)


def clamp_numeric(value: Any, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if minimum is not None:
        number = max(number, minimum)
    if maximum is not None:
        number = min(number, maximum)
    return number


def parse_settings_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    return {
        "mode": str(data.get("mode", "full_body")),
        "photo_interval_sec": clamp_numeric(data.get("photo_interval_sec"), 5.0, 0.5, 3600.0),
        "quality": int(clamp_numeric(data.get("quality"), 75, 10, 95)),
        "gesture_mode": bool(data.get("gesture_mode", False)),
        "pan": clamp_numeric(data.get("pan"), 0.0),
        "tilt": clamp_numeric(data.get("tilt"), 0.0),
        "height": clamp_numeric(data.get("height"), 0.0),
    }


def to_board_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": settings["mode"],
        "photo_interval_sec": settings["photo_interval_sec"],
        "quality": settings["quality"],
        "gesture_mode": settings["gesture_mode"],
        "servo": {
            "pan": settings["pan"],
            "tilt": settings["tilt"],
            "height": settings["height"],
        },
    }


def recvall(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def video_reader_loop(state: BridgeState) -> None:
    while True:
        try:
            board_ip, video_port, _command_port = state.connection_target()
            if not board_ip:
                state.set_video_error("board IP not set")
                time.sleep(0.5)
                continue
            print(f"[bridge/video] connecting to {board_ip}:{video_port}")
            with socket.create_connection((board_ip, video_port), timeout=5) as sock:
                sock.settimeout(10)
                print("[bridge/video] connected")
                while True:
                    header = recvall(sock, HEADER_SIZE)
                    frame_size = struct.unpack(HEADER_FORMAT, header)[0]
                    if frame_size <= 0 or frame_size > 20_000_000:
                        raise ValueError(f"invalid frame size: {frame_size}")
                    jpeg = recvall(sock, frame_size)
                    state.set_frame(jpeg)
        except Exception as exc:
            message = f"video connection error: {exc}"
            print(f"[bridge/video] {message}")
            state.set_video_error(message)
            time.sleep(1.0)


def auto_capture_loop(state: BridgeState) -> None:
    while True:
        should_capture = False
        with state.lock:
            if state.capture_running:
                interval = float(state.photo_interval_sec)
                should_capture = time.time() - state.last_capture_time >= interval
        if should_capture:
            try:
                state.save_latest_jpeg("auto")
            except Exception as exc:
                print(f"[bridge/photo] auto capture skipped: {exc}")
                time.sleep(0.5)
        time.sleep(0.1)


def send_board_command(state: BridgeState, command: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
    try:
        board_ip, _video_port, command_port = state.connection_target()
        if not board_ip:
            abort_json(400, "board IP not set")
        print(f"[bridge/command] -> board: {json.dumps(command, sort_keys=True)}")
        with socket.create_connection((board_ip, command_port), timeout=timeout) as sock:
            file = sock.makefile("rwb")
            file.write((json.dumps(command) + "\n").encode("utf-8"))
            file.flush()
            raw_response = file.readline()
            if not raw_response:
                raise ConnectionError("board command socket closed without response")
            response = json.loads(raw_response.decode("utf-8"))
            print(f"[bridge/command] <- board: {json.dumps(response, sort_keys=True)}")
            return response
    except Exception as exc:
        abort_json(502, f"board command failed: {exc}")


def create_app(state: BridgeState) -> Flask:
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/video")
    def video() -> Response:
        def mjpeg_generator():
            boundary = b"--frame\r\n"
            last_sent_id = 0
            while True:
                with state.lock:
                    jpeg = state.latest_jpeg
                    frame_id = int(state.last_frame_time * 1000)
                if jpeg is None or frame_id == last_sent_id:
                    time.sleep(0.03)
                    continue
                last_sent_id = frame_id
                yield (
                    boundary
                    + b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
                time.sleep(0.01)

        return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/bridge_status")
    def bridge_status():
        return jsonify(state.snapshot())

    @app.post("/api/connection")
    def update_connection():
        payload = request.get_json(silent=True)
        board_ip = ""
        if isinstance(payload, dict):
            board_ip = str(payload.get("board_ip", ""))
        state.update_board_ip(board_ip)
        return jsonify({"ok": True, "message": "board IP updated", "bridge": state.snapshot()})

    @app.get("/api/status")
    def board_status():
        response = send_board_command(state, {"type": "get_status"})
        return jsonify({"bridge": state.snapshot(), "board": response})

    @app.get("/api/captures")
    def captures():
        return jsonify(
            {
                "capture_dir": str(state.capture_dir),
                "files": state.list_captures(),
            }
        )

    @app.get("/captures/<path:filename>")
    def capture_file(filename: str):
        path = state.resolve_capture_path(filename)
        download = request.args.get("download", "0")
        return send_file(path, mimetype="image/jpeg", as_attachment=(download == "1"), download_name=path.name)

    @app.delete("/api/captures/<path:filename>")
    def delete_capture(filename: str):
        return jsonify(state.delete_capture(filename))

    @app.post("/api/settings")
    def settings():
        parsed_settings = parse_settings_payload(request.get_json(silent=True))
        board_settings = to_board_settings(parsed_settings)
        response = send_board_command(state, {"type": "set_settings", "settings": board_settings})
        with state.lock:
            state.photo_interval_sec = float(parsed_settings["photo_interval_sec"])
            state.last_settings = board_settings
        return jsonify({"ok": True, "saved_on": "board settings only", "board": response, "bridge": state.snapshot()})

    @app.post("/api/start")
    def start():
        parsed_settings = parse_settings_payload(request.get_json(silent=True))
        board_settings = to_board_settings(parsed_settings)
        response = send_board_command(state, {"type": "start_capture", "settings": board_settings})
        with state.lock:
            state.capture_running = True
            state.photo_interval_sec = float(parsed_settings["photo_interval_sec"])
            state.last_settings = board_settings
            state.last_capture_time = 0.0
        return jsonify({"ok": True, "message": "PC auto capture started", "board": response, "bridge": state.snapshot()})

    @app.post("/api/stop")
    def stop():
        response = send_board_command(state, {"type": "stop_capture"})
        with state.lock:
            state.capture_running = False
        return jsonify({"ok": True, "message": "PC auto capture stopped", "board": response, "bridge": state.snapshot()})

    @app.post("/api/photo")
    def photo():
        board_response = send_board_command(state, {"type": "take_photo"})
        save_response = state.save_latest_jpeg("manual")
        return jsonify({"ok": True, "board": board_response, "photo": save_response, "bridge": state.snapshot()})

    return app


def main() -> None:
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description="MakeNTU PC-hosted web UI bridge")
    parser.add_argument("board_ip", nargs="?", default="", help="IP address of the i.MX93 board")
    parser.add_argument("--video-port", type=int, default=9999)
    parser.add_argument("--command-port", type=int, default=10000)
    parser.add_argument("--host", default="0.0.0.0", help="PC web server bind address")
    parser.add_argument("--port", type=int, default=8000, help="PC web server port")
    parser.add_argument("--capture-dir", default="captures", help="where photos are saved on the PC")
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)
    if not capture_dir.is_absolute():
        capture_dir = (script_dir / capture_dir).resolve()

    state = BridgeState(
        args.board_ip,
        args.video_port,
        args.command_port,
        capture_dir,
    )
    threading.Thread(target=video_reader_loop, args=(state,), daemon=True).start()
    threading.Thread(target=auto_capture_loop, args=(state,), daemon=True).start()

    app = create_app(state)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
