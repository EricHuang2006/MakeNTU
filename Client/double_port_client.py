"""
Standalone MakeNTU video + control demo client.

Run on PC side:
  python3 demo_video_control_client.py 192.168.0.73

Commands you can type:
  start 5
  stop
  gesture on
  gesture off
  pose full_body
  pose upper_body
  photo
  servo 10 -5 20
  state
  quit

Requires:
  pip install opencv-python
"""

import json
import socket
import struct
import sys
import threading
from typing import Dict, Any

import cv2
import numpy as np

SERVER_IP = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
VIDEO_PORT = 9999
COMMAND_PORT = 10000

stop_event = threading.Event()


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data += chunk
    return data


def video_receiver() -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((SERVER_IP, VIDEO_PORT))
            print(f"[video] connected to {SERVER_IP}:{VIDEO_PORT}")
            while not stop_event.is_set():
                size_bytes = recv_exact(sock, struct.calcsize("!Q"))
                payload_size = struct.unpack("!Q", size_bytes)[0]
                jpeg_bytes = recv_exact(sock, payload_size)

                array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
                if frame is None:
                    print("[video] warning: could not decode frame")
                    continue

                cv2.imshow("MakeNTU demo video stream", frame)
                # Press q in the video window to close the client.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break
    except Exception as exc:
        if not stop_event.is_set():
            print(f"[video] stopped: {exc}")
    finally:
        cv2.destroyAllWindows()


def send_command(file, command: Dict[str, Any]) -> None:
    file.write((json.dumps(command) + "\n").encode("utf-8"))
    file.flush()
    response = file.readline().decode("utf-8").strip()
    print("[response]", response)


def main() -> None:
    threading.Thread(target=video_receiver, daemon=True).start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cmd_sock:
        cmd_sock.connect((SERVER_IP, COMMAND_PORT))
        file = cmd_sock.makefile("rwb")
        print(f"[command] connected to {SERVER_IP}:{COMMAND_PORT}")
        print("Type: start <interval_sec>, stop, gesture on/off, pose <mode>, photo, servo <pan> <tilt> <height>, state, quit")

        while not stop_event.is_set():
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            parts = text.split()

            if parts[0] == "quit":
                break
            elif parts[0] == "start":
                interval = float(parts[1]) if len(parts) >= 2 else 5.0
                send_command(file, {
                    "type": "start_capture",
                    "settings": {
                        "photo_interval_sec": interval,
                        "save_dir": "captured_photos",
                    },
                })
            elif parts[0] == "stop":
                send_command(file, {"type": "stop_capture"})
            elif parts[:2] == ["gesture", "on"]:
                send_command(file, {"type": "set_gesture_mode", "enabled": True})
            elif parts[:2] == ["gesture", "off"]:
                send_command(file, {"type": "set_gesture_mode", "enabled": False})
            elif parts[0] == "pose" and len(parts) >= 2:
                send_command(file, {"type": "set_pose_mode", "mode": parts[1]})
            elif parts[0] == "photo":
                send_command(file, {"type": "take_photo"})
            elif parts[0] == "servo" and len(parts) == 4:
                pan, tilt, height = map(float, parts[1:])
                send_command(file, {"type": "set_servo", "pan": pan, "tilt": tilt, "height": height})
            elif parts[0] == "state":
                send_command(file, {"type": "get_state"})
            else:
                print("unknown command")

    stop_event.set()


if __name__ == "__main__":
    main()