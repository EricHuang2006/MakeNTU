# MakeNTU UI Bridge Demo

This demo uses the architecture:

```text
i.MX93 board
  ├── port 9999: raw JPEG video stream, [8-byte !Q length][JPEG bytes]
  └── port 10000: JSON command socket

PC
  ├── connects to board:9999 and board:10000
  ├── hosts the web UI at http://PC_IP:8000
  └── converts raw JPEG stream into browser MJPEG at /video

Browser
  └── opens the PC-hosted UI
```

## 1. Run the board-side demo

On the i.MX93 / camera device:

```bash
cd board_device
python3 -m pip install -r requirements.txt
python3 board_server_demo.py --camera 0 --width 640 --height 480 --fps 15
```

The board listens on:

```text
9999  video stream
10000 command socket
```

## 2. Run the PC-hosted web bridge

On your PC:

```bash
cd pc_web_bridge
python3 -m pip install -r requirements.txt
python3 bridge_server.py 192.168.0.73
```

Replace `192.168.0.73` with your i.MX93 board IP.

## 3. Open the UI

From your PC or another device on the same network:

```text
http://PC_IP:8000
```

The UI supports:

- live video preview
- mode selection
- photo interval
- JPEG quality
- servo values
- Start automatic capture
- Stop automatic capture
- Take Photo Now
- Save Settings

## How this maps to your real MakeNTU code

Your current code already has a board-side video stream using:

```python
size = struct.pack("Q", len(data))
client_socket.sendall(size + data)
```

To integrate this design into the real project later:

1. Keep the board's video stream on port 9999.
2. Add a command socket on port 10000.
3. Let commands update shared state such as `running`, `photo_interval_sec`, `pose_mode`, `gesture_mode`, and servo targets.
4. Run the PC web bridge on your development laptop.
5. Let users use the browser UI hosted by the PC.

## Notes

This is intentionally a demo. For a hackathon/local network it is fine, but for real deployment you should add authentication or restrict access to trusted devices.
