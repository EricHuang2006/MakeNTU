#!/usr/bin/env python3
"""
Live gesture detector diagnostic.

This script runs the pose model on a video source and displays the gesture
recognized by pose_logic.classify_manual_gesture(). It does not start the FSM
and does not send any motor, stepper, LED, Discord, or UART commands.
"""

from __future__ import annotations

import argparse
import socket
import struct
import time
from pathlib import Path

import cv2
import numpy as np

from config import CONF_THRESHOLD, IMG_SIZE, MODEL_PATH, NMS_THRESHOLD
from drawing import draw_debug_view
from pose_logic import classify_manual_gesture
from vision import (
    apply_nms,
    decode_pose_output,
    load_pose_model,
    preprocess_frame,
    run_inference,
)


HEADER_FORMAT = "Q"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
WINDOW_NAME = "MakeNTU Gesture Stream Test"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run live video through pose gesture detection only."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--camera",
        default="0",
        help="OpenCV camera index or video path. Default: 0.",
    )
    source.add_argument(
        "--board-stream",
        help="Read MakeNTU JPEG socket stream as HOST:PORT, for example 10.27.106.155:9999.",
    )
    parser.add_argument("--model", default=MODEL_PATH, help="Path to YOLO pose TFLite model.")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Person confidence threshold.")
    parser.add_argument("--nms", type=float, default=NMS_THRESHOLD, help="NMS threshold.")
    parser.add_argument("--keypoint-conf", type=float, default=0.3, help="Keypoint visibility threshold.")
    parser.add_argument("--no-window", action="store_true", help="Print detections without opening a GUI window.")
    parser.add_argument(
        "--stream-host",
        default=None,
        help="Bind address for annotated output stream, for example 0.0.0.0.",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=9998,
        help="TCP port for annotated output stream. Default: 9998.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=80, help="Output stream JPEG quality.")
    return parser.parse_args()


def resolve_model_path(model_path):
    path = Path(model_path)
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parent / path).resolve())


def parse_camera_source(source):
    try:
        return int(source)
    except ValueError:
        return source


def recvall(sock, length):
    chunks = bytearray()
    while len(chunks) < length:
        packet = sock.recv(length - len(chunks))
        if not packet:
            raise ConnectionError("stream socket closed")
        chunks.extend(packet)
    return bytes(chunks)


def board_stream_frames(address):
    host, port_text = address.rsplit(":", 1)
    port = int(port_text)
    while True:
        print(f"[stream] connecting to {host}:{port}")
        try:
            with socket.create_connection((host, port), timeout=5.0) as sock:
                sock.settimeout(10.0)
                print("[stream] connected")
                while True:
                    header = recvall(sock, HEADER_SIZE)
                    frame_size = struct.unpack(HEADER_FORMAT, header)[0]
                    if frame_size <= 0 or frame_size > 20_000_000:
                        raise ValueError(f"invalid frame size: {frame_size}")
                    data = recvall(sock, frame_size)
                    frame_array = np.frombuffer(data, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    if frame is None:
                        print("[stream] skipped undecodable JPEG frame")
                        continue
                    yield frame
        except Exception as exc:
            print(f"[stream] disconnected: {exc}; retrying in 1s")
            time.sleep(1.0)


def camera_frames(source):
    cap = cv2.VideoCapture(parse_camera_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video source: {source}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


class AnnotatedStreamServer:
    def __init__(self, host, port, jpeg_quality):
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.clients = []
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(4)
        self.server_socket.setblocking(False)
        print(f"[output] annotated stream listening on {self.host}:{self.port}")

    def accept_pending_clients(self):
        while True:
            try:
                client, address = self.server_socket.accept()
            except BlockingIOError:
                return
            client.settimeout(0.5)
            self.clients.append(client)
            print(f"[output] client connected from {address}")

    def send_frame(self, frame):
        self.accept_pending_clients()
        if not self.clients:
            return

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            print("[output] skipped frame: JPEG encoding failed")
            return

        payload = encoded.tobytes()
        packet = struct.pack(HEADER_FORMAT, len(payload)) + payload
        connected_clients = []
        for client in self.clients:
            try:
                client.sendall(packet)
                connected_clients.append(client)
            except OSError as exc:
                print(f"[output] client disconnected: {exc}")
                client.close()
        self.clients = connected_clients

    def close(self):
        for client in self.clients:
            client.close()
        self.server_socket.close()


def draw_gesture_label(display_img, gesture, people_count):
    label = gesture or "none"
    color = (0, 255, 0) if gesture else (0, 0, 255)
    cv2.rectangle(display_img, (8, 154), (390, 216), (0, 0, 0), -1)
    cv2.putText(
        display_img,
        f"Gesture: {label}",
        (16, 184),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display_img,
        f"People: {people_count}",
        (16, 208),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def run_stream(args):
    interpreter, model_info = load_pose_model(resolve_model_path(args.model))
    frames = board_stream_frames(args.board_stream) if args.board_stream else camera_frames(args.camera)
    output_stream = None
    if args.stream_host is not None:
        output_stream = AnnotatedStreamServer(
            host=args.stream_host,
            port=args.stream_port,
            jpeg_quality=args.jpeg_quality,
        )
    last_gesture = None
    frame_count = 0
    started_at = time.monotonic()

    try:
        for frame in frames:
            input_data, ai_img = preprocess_frame(frame, model_info, IMG_SIZE)
            output_data = run_inference(interpreter, model_info, input_data)
            boxes, scores, all_keypoints = decode_pose_output(
                output_data,
                model_info,
                IMG_SIZE,
                args.conf,
            )
            indices = apply_nms(boxes, scores, args.conf, args.nms)
            gesture = classify_manual_gesture(
                indices=indices,
                all_keypoints=all_keypoints,
                keypoint_conf=args.keypoint_conf,
            )

            if gesture != last_gesture:
                print(f"[gesture] {gesture or 'none'}")
                last_gesture = gesture

            frame_count += 1
            elapsed = max(0.001, time.monotonic() - started_at)
            fps = frame_count / elapsed

            display_img = draw_debug_view(
                img=ai_img,
                indices=indices,
                all_keypoints=all_keypoints,
                face_boxes=[],
                photo_good=gesture is not None,
                quality_score=min(100, len(indices) * 30),
                quality_problems=[f"gesture={gesture or 'none'} fps={fps:.1f}"],
                adjustment={
                    "pan_dir": "none",
                    "pan_amount_deg": 0.0,
                    "tilt_dir": "none",
                    "tilt_amount_deg": 0.0,
                    "size_status": "test",
                },
            )
            draw_gesture_label(display_img, gesture, len(indices))
            if output_stream is not None:
                output_stream.send_frame(display_img)
            if not args.no_window:
                cv2.imshow(WINDOW_NAME, display_img)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        if output_stream is not None:
            output_stream.close()


def main():
    args = parse_args()
    try:
        run_stream(args)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
