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
import asyncio
import json
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


HEADER_FORMAT = "!Q"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class BridgeState:
    def __init__(self, board_ip: str, video_port: int, command_port: int, capture_dir: Path) -> None:
        self.board_ip = board_ip
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
                raise HTTPException(status_code=409, detail="no video frame received yet")
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
            print(f"[bridge/video] connecting to {state.board_ip}:{state.video_port}")
            with socket.create_connection((state.board_ip, state.video_port), timeout=5) as sock:
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
        print(f"[bridge/command] -> board: {json.dumps(command, sort_keys=True)}")
        with socket.create_connection((state.board_ip, state.command_port), timeout=timeout) as sock:
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
        raise HTTPException(status_code=502, detail=f"board command failed: {exc}") from exc


class Settings(BaseModel):
    mode: str = Field(default="full_body")
    photo_interval_sec: float = Field(default=5.0, ge=0.5, le=3600)
    quality: int = Field(default=75, ge=10, le=95)
    gesture_mode: bool = False
    pan: float = 0.0
    tilt: float = 0.0
    height: float = 0.0

    def to_board_settings(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "photo_interval_sec": self.photo_interval_sec,
            "quality": self.quality,
            "gesture_mode": self.gesture_mode,
            "servo": {
                "pan": self.pan,
                "tilt": self.tilt,
                "height": self.height,
            },
        }


def create_app(state: BridgeState) -> FastAPI:
    app = FastAPI(title="MakeNTU PC Web Bridge")
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html")

    @app.get("/video")
    async def video() -> StreamingResponse:
        async def mjpeg_generator():
            boundary = b"--frame\r\n"
            last_sent_id = 0
            while True:
                with state.lock:
                    jpeg = state.latest_jpeg
                    frame_id = int(state.last_frame_time * 1000)
                if jpeg is None or frame_id == last_sent_id:
                    await asyncio.sleep(0.03)
                    continue
                last_sent_id = frame_id
                yield (
                    boundary
                    + b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
                await asyncio.sleep(0.01)

        return StreamingResponse(
            mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/bridge_status")
    async def bridge_status() -> JSONResponse:
        return JSONResponse(state.snapshot())

    @app.get("/api/status")
    async def board_status() -> JSONResponse:
        response = send_board_command(state, {"type": "get_status"})
        return JSONResponse({"bridge": state.snapshot(), "board": response})

    @app.get("/api/captures")
    async def captures() -> JSONResponse:
        files = sorted(state.capture_dir.glob("*.jpg"), reverse=True)
        return JSONResponse({"capture_dir": str(state.capture_dir), "files": [p.name for p in files[:100]]})

    @app.post("/api/settings")
    async def settings(settings: Settings) -> JSONResponse:
        board_settings = settings.to_board_settings()
        response = send_board_command(state, {"type": "set_settings", "settings": board_settings})
        with state.lock:
            state.photo_interval_sec = float(settings.photo_interval_sec)
            state.last_settings = board_settings
        return JSONResponse({"ok": True, "saved_on": "board settings only", "board": response, "bridge": state.snapshot()})

    @app.post("/api/start")
    async def start(settings: Settings) -> JSONResponse:
        board_settings = settings.to_board_settings()
        response = send_board_command(state, {"type": "start_capture", "settings": board_settings})
        with state.lock:
            state.capture_running = True
            state.photo_interval_sec = float(settings.photo_interval_sec)
            state.last_settings = board_settings
            state.last_capture_time = 0.0
        return JSONResponse({"ok": True, "message": "PC auto capture started", "board": response, "bridge": state.snapshot()})

    @app.post("/api/stop")
    async def stop() -> JSONResponse:
        response = send_board_command(state, {"type": "stop_capture"})
        with state.lock:
            state.capture_running = False
        return JSONResponse({"ok": True, "message": "PC auto capture stopped", "board": response, "bridge": state.snapshot()})

    @app.post("/api/photo")
    async def photo() -> JSONResponse:
        board_response = send_board_command(state, {"type": "take_photo"})
        save_response = state.save_latest_jpeg("manual")
        return JSONResponse({"ok": True, "board": board_response, "photo": save_response, "bridge": state.snapshot()})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="MakeNTU PC-hosted web UI bridge")
    parser.add_argument("board_ip", help="IP address of the i.MX93 board")
    parser.add_argument("--video-port", type=int, default=9999)
    parser.add_argument("--command-port", type=int, default=10000)
    parser.add_argument("--host", default="0.0.0.0", help="PC web server bind address")
    parser.add_argument("--port", type=int, default=8000, help="PC web server port")
    parser.add_argument("--capture-dir", default="captures", help="where photos are saved on the PC")
    args = parser.parse_args()

    state = BridgeState(args.board_ip, args.video_port, args.command_port, Path(args.capture_dir))
    threading.Thread(target=video_reader_loop, args=(state,), daemon=True).start()
    threading.Thread(target=auto_capture_loop, args=(state,), daemon=True).start()

    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()