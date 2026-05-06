import os
import time
import fcntl
import ctypes
import signal
import sys

# ============================================================
# FRDM-IMX93 P11 GPIO
# ============================================================
# P11 pin 3 = GPIO_IO02 -> A4988 STEP
# P11 pin 5 = GPIO_IO03 -> A4988 DIR
#
# gpiodetect:
# gpiochip0 [43810000.gpio] = GPIO2 bank
#
# Therefore:
# GPIO_IO02 = /dev/gpiochip0 line 2
# GPIO_IO03 = /dev/gpiochip0 line 3
# ============================================================

GPIOCHIP = "/dev/gpiochip0"

STEP_LINE = 2
DIR_LINE = 3

# Arduino original:
# delayMicroseconds(500)
# delayMicroseconds(1000)
STEP_HIGH_TIME = 0.0005
STEP_LOW_TIME = 0.0010

DEFAULT_PULSES = 2500
PAUSE_TIME = 1.0


# ============================================================
# Linux GPIO character device ABI v1
# ============================================================

GPIOHANDLES_MAX = 64

GPIOHANDLE_REQUEST_INPUT = 1 << 0
GPIOHANDLE_REQUEST_OUTPUT = 1 << 1

GPIO_GET_LINEHANDLE_IOCTL = 0xC16CB403
GPIOHANDLE_SET_LINE_VALUES_IOCTL = 0xC040B409


class gpiohandle_request(ctypes.Structure):
    _fields_ = [
        ("lineoffsets", ctypes.c_uint32 * GPIOHANDLES_MAX),
        ("flags", ctypes.c_uint32),
        ("default_values", ctypes.c_uint8 * GPIOHANDLES_MAX),
        ("consumer_label", ctypes.c_char * 32),
        ("lines", ctypes.c_uint32),
        ("fd", ctypes.c_int),
    ]


class gpiohandle_data(ctypes.Structure):
    _fields_ = [
        ("values", ctypes.c_uint8 * GPIOHANDLES_MAX),
    ]


chip_fd = None
handle_fd = None
current_step = 0
current_dir = 0


def request_gpio_lines():
    global chip_fd, handle_fd

    chip_fd = os.open(GPIOCHIP, os.O_RDONLY)

    req = gpiohandle_request()
    req.lineoffsets[0] = STEP_LINE
    req.lineoffsets[1] = DIR_LINE
    req.flags = GPIOHANDLE_REQUEST_OUTPUT
    req.default_values[0] = 0
    req.default_values[1] = 0
    req.consumer_label = b"a4988-stepper"
    req.lines = 2

    fcntl.ioctl(chip_fd, GPIO_GET_LINEHANDLE_IOCTL, req)

    handle_fd = req.fd
    print(f"Requested GPIO lines successfully:")
    print(f"  STEP = {GPIOCHIP} line {STEP_LINE}")
    print(f"  DIR  = {GPIOCHIP} line {DIR_LINE}")


def write_lines(step_value, dir_value):
    global current_step, current_dir

    current_step = 1 if step_value else 0
    current_dir = 1 if dir_value else 0

    data = gpiohandle_data()
    data.values[0] = current_step
    data.values[1] = current_dir

    fcntl.ioctl(handle_fd, GPIOHANDLE_SET_LINE_VALUES_IOCTL, data)


def set_dir(dir_value):
    write_lines(current_step, dir_value)


def set_step(step_value):
    write_lines(step_value, current_dir)


def step_once():
    set_step(1)
    time.sleep(STEP_HIGH_TIME)

    set_step(0)
    time.sleep(STEP_LOW_TIME)


def move_steps(steps, direction):
    set_dir(direction)
    time.sleep(0.001)

    for _ in range(steps):
        step_once()


def cleanup(signum=None, frame=None):
    print("\nStopping stepper output...")

    try:
        if handle_fd is not None:
            write_lines(0, current_dir)
            os.close(handle_fd)
    except Exception:
        pass

    try:
        if chip_fd is not None:
            os.close(chip_fd)
    except Exception:
        pass

    print("Done.")
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def arduino_like_loop():
    print("Arduino-like loop started.")
    print("Press Ctrl+C to stop.")

    while True:
        print("DIR = LOW")
        move_steps(DEFAULT_PULSES, direction=0)
        time.sleep(PAUSE_TIME)

        print("DIR = HIGH")
        move_steps(DEFAULT_PULSES, direction=1)
        time.sleep(PAUSE_TIME)


def interactive_mode():
    print("A4988 stepper controller")
    print("Commands:")
    print("  steps <count> <dir>    example: steps 2500 0")
    print("  loop                   run Arduino-like loop")
    print("  q                      quit")
    print()

    while True:
        cmd = input("stepper> ").strip()
        parts = cmd.split()

        if cmd.lower() in ["q", "quit", "exit"]:
            cleanup()

        if cmd.lower() == "loop":
            arduino_like_loop()

        if len(parts) == 3 and parts[0].lower() == "steps":
            try:
                count = int(parts[1])
                direction = int(parts[2])

                if direction not in [0, 1]:
                    print("dir must be 0 or 1")
                    continue

                move_steps(count, direction)

            except ValueError:
                print("Usage: steps 2500 0")
            continue

        print("Unknown command. Example: steps 2500 0")


request_gpio_lines()
write_lines(0, 0)

interactive_mode()
