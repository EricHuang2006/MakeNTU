import socket
import ssl
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np


IMX93_IP = "172.20.10.4"
PORT = 9999
CA_CERT_PATH = Path(__file__).with_name("server.crt")
SERVER_HOSTNAME = "makentu-imx93"
RECONNECT_DELAY_SECONDS = 1.0
SOCKET_TIMEOUT_SECONDS = 5.0
HEADER_SIZE = struct.calcsize("Q")
WINDOW_NAME = "MakeNTU - i.MX 93 Live View"


def connect_to_server():
    context = ssl.create_default_context(cafile=str(CA_CERT_PATH))
    print(f"Connecting to i.MX 93 at {IMX93_IP}:{PORT}...")
    raw_socket = socket.create_connection((IMX93_IP, PORT), timeout=SOCKET_TIMEOUT_SECONDS)
    client_socket = context.wrap_socket(raw_socket, server_hostname=SERVER_HOSTNAME)
    client_socket.settimeout(SOCKET_TIMEOUT_SECONDS)
    print("Connected! Receiving video stream...")
    return client_socket


def recv_exact(client_socket, expected_size):
    chunks = bytearray()
    while len(chunks) < expected_size:
        packet = client_socket.recv(min(4096, expected_size - len(chunks)))
        if not packet:
            raise ConnectionError("Server closed the connection.")
        chunks.extend(packet)
    return bytes(chunks)


def receive_frame(client_socket):
    packed_msg_size = recv_exact(client_socket, HEADER_SIZE)
    msg_size = struct.unpack("Q", packed_msg_size)[0]
    if msg_size <= 0:
        raise ValueError(f"Invalid frame size: {msg_size}")

    frame_data = recv_exact(client_socket, msg_size)
    img_np = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG frame.")
    return frame


def main():
    while True:
        client_socket = None
        try:
            client_socket = connect_to_server()

            while True:
                frame = receive_frame(client_socket)
                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stream closed by user.")
                    return

        except KeyboardInterrupt:
            print("Stream closed by user.")
            return
        except Exception as exc:
            print(f"Stream interrupted: {exc}")
            print(f"Reconnecting in {RECONNECT_DELAY_SECONDS:.1f} second(s)...")
            time.sleep(RECONNECT_DELAY_SECONDS)
        finally:
            if client_socket is not None:
                client_socket.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        cv2.destroyAllWindows()
