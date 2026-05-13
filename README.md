# MakeNTU Repo

This is a camera rig project for the FRDM i.MX93 board. It combines pose detection, finite-state camera behavior, servo/stepper motor control, LED feedback, and PC-side video visualization for operating the rig on a local network.

## Repository Layout

```text
Client/                 PC-side stream clients and experiments
Server/
  MakeNTU_iMX93/        Main i.MX93 runtime, FSM, vision, motor, and hardware control
  Hardware/unit_test/   Hardware bring-up and unit test scripts
  models/               TensorFlow Lite pose models
  test/                 Socket/upload/SSL experiments
  utils/                Utility scripts and Python requirements
Web/
  board_device/         Demo board-side video/command server
  pc_web_bridge/        Browser bridge from board demo sockets to web UI
```

## Default Run Flow

Run the real camera rig runtime on the i.MX93 / server side:

```bash
cd Server/MakeNTU_iMX93
python3 main.py
```

It opens the camera, runs the pose model, updates the camera rig FSM, drives configured hardware outputs, and serves a plain TCP JPEG frame stream on `PORT` from `config.py` (`9999` by default).

On the PC / client side, use the OpenCV client to visualize that stream:

```bash
cd Client
python3 pc_client.py <IMX93_IP> 9999
```

Press `q` in the OpenCV window to quit.

Important configuration lives in:

```text
Server/MakeNTU_iMX93/config.py
```

Before running on hardware, review the UART, GPIO, model path, motor output, stepper output, and LED output settings in that file.

## Python Setup

Install the project dependencies from the repo root:

```bash
python3 -m pip install -r requirements.txt
```

For board-only installs that do not need the web bridge or PC preview tools, use the smaller server requirements:

```bash
cd Server/utils
python3 -m pip install -r requirements.txt
```

The web bridge and board demo also keep local requirement files for isolated setup:

```bash
cd Web/pc_web_bridge
python3 -m pip install -r requirements.txt

cd ../board_device
python3 -m pip install -r requirements.txt
```

## Web UI Bridge Demo

The `Web/` folder contains a demo-ready browser UI bridge. It is a separate demo flow from the real `Server/MakeNTU_iMX93/main.py` runtime because the web UI expects both a video socket and a command socket.

Run the board-side demo on the i.MX93 / camera device:

```bash
cd Web/board_device
python3 -m pip install -r requirements.txt
python3 board_server_demo.py --camera 0 --width 640 --height 480 --fps 15
```

Run the PC-hosted web bridge on your PC:

```bash
cd Web/pc_web_bridge
python3 -m pip install -r requirements.txt
python3 bridge_server.py <BOARD_IP>
```

Then open:

```text
http://<PC_IP>:8000
```

Use `Client/pc_client.py` for the main live-view workflow with `Server/MakeNTU_iMX93/main.py`. Use the web bridge when you want the browser demo with live preview, settings, manual capture, and PC-side saved captures.

## Notes

- The pose models are stored under `Server/models/`.
- Hardware scripts under `Server/Hardware/unit_test/` are intended for bring-up and isolated checks.
- Set `DISCORD_WEBHOOK_URL` in the environment if event logging should post to Discord.
