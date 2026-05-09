import argparse
import ctypes
import fcntl
import os
import signal
import sys
import time


GPIOHANDLES_MAX = 64
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


class StepDirAxisNoSwitch:
    def __init__(
        self,
        gpiochip="/dev/gpiochip0",
        step_line=2,
        dir_line=3,
        steps_per_cm=1000.0,
        up_direction=0,
        step_high_time=0.0005,
        step_low_time=0.0010,
    ):
        if steps_per_cm <= 0:
            raise ValueError("steps_per_cm must be greater than 0")

        self.gpiochip = gpiochip
        self.step_line = int(step_line)
        self.dir_line = int(dir_line)
        self.steps_per_cm = float(steps_per_cm)
        self.up_direction = 1 if int(up_direction) else 0
        self.step_high_time = float(step_high_time)
        self.step_low_time = float(step_low_time)
        self.position_cm = 0.0
        self.current_step = 0
        self.current_dir = 0
        self.chip_fd = None
        self.handle_fd = None
        self.open()

    def open(self):
        self.chip_fd = os.open(self.gpiochip, os.O_RDONLY)
        req = gpiohandle_request()
        req.lineoffsets[0] = self.step_line
        req.lineoffsets[1] = self.dir_line
        req.flags = GPIOHANDLE_REQUEST_OUTPUT
        req.default_values[0] = 0
        req.default_values[1] = 0
        req.consumer_label = b"a4988-no-switch-test"
        req.lines = 2
        try:
            fcntl.ioctl(self.chip_fd, GPIO_GET_LINEHANDLE_IOCTL, req)
        except Exception:
            os.close(self.chip_fd)
            self.chip_fd = None
            raise
        self.handle_fd = req.fd
        self.write_lines(0, 0)

    def close(self):
        try:
            if self.handle_fd is not None:
                self.write_lines(0, self.current_dir)
                os.close(self.handle_fd)
        finally:
            self.handle_fd = None
        try:
            if self.chip_fd is not None:
                os.close(self.chip_fd)
        finally:
            self.chip_fd = None

    def write_lines(self, step_value, dir_value):
        if self.handle_fd is None:
            raise RuntimeError("stepper axis is closed")

        self.current_step = 1 if step_value else 0
        self.current_dir = 1 if dir_value else 0
        data = gpiohandle_data()
        data.values[0] = self.current_step
        data.values[1] = self.current_dir
        fcntl.ioctl(self.handle_fd, GPIOHANDLE_SET_LINE_VALUES_IOCTL, data)

    def step_once(self):
        self.write_lines(1, self.current_dir)
        time.sleep(self.step_high_time)
        self.write_lines(0, self.current_dir)
        time.sleep(self.step_low_time)

    def move_steps(self, steps, direction):
        steps = int(round(abs(steps)))
        if steps == 0:
            return 0

        self.write_lines(0, direction)
        time.sleep(0.001)
        for _ in range(steps):
            self.step_once()
        return steps

    def move_to_x_cm(self, target_cm):
        target_cm = float(target_cm)
        delta_cm = target_cm - self.position_cm
        steps = int(round(abs(delta_cm) * self.steps_per_cm))
        if steps == 0:
            return {
                "target_cm": target_cm,
                "moved_cm": 0.0,
                "steps": 0,
                "direction": self.current_dir,
                "position_cm": self.position_cm,
            }

        direction = self.up_direction if delta_cm > 0 else 1 - self.up_direction
        moved_steps = self.move_steps(steps, direction)
        moved_cm = moved_steps / self.steps_per_cm
        if delta_cm < 0:
            moved_cm = -moved_cm
        self.position_cm += moved_cm
        return {
            "target_cm": target_cm,
            "moved_cm": moved_cm,
            "steps": moved_steps,
            "direction": direction,
            "position_cm": self.position_cm,
        }


axis = None


def cleanup(_signum=None, _frame=None):
    if axis is not None:
        axis.close()
    print("closed")
    raise SystemExit(0)


def wait_for_enter(message):
    input(f"{message}\nPress Enter when ready, or Ctrl+C to abort. ")


def move_to_position(target_cm, pause_s):
    before = axis.position_cm
    result = axis.move_to_x_cm(target_cm)
    print(
        "move: "
        f"{before:.2f}cm -> {target_cm:.2f}cm, "
        f"moved={result['moved_cm']:.2f}cm, "
        f"steps={result['steps']}, "
        f"dir={result['direction']}, "
        f"position={result['position_cm']:.2f}cm"
    )
    time.sleep(pause_s)


def parse_args():
    parser = argparse.ArgumentParser(
        description="No-switch unit test: move stepper through photo positions and return to 0cm."
    )
    parser.add_argument("--gpiochip", default="/dev/gpiochip0")
    parser.add_argument("--step-line", type=int, default=2)
    parser.add_argument("--dir-line", type=int, default=3)
    parser.add_argument("--steps-per-cm", type=float, default=1000.0)
    parser.add_argument("--rod-length-cm", type=float, default=21.0)
    parser.add_argument("--photo-count", type=int, default=3)
    parser.add_argument("--up-direction", type=int, choices=(0, 1), default=0)
    parser.add_argument("--step-high-time", type=float, default=0.0005)
    parser.add_argument("--step-low-time", type=float, default=0.0010)
    parser.add_argument("--pause-s", type=float, default=0.5)
    parser.add_argument("--yes", action="store_true", help="Run without interactive confirmation.")
    return parser.parse_args()


def main():
    global axis

    args = parse_args()
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    if args.photo_count <= 0:
        raise SystemExit("photo-count must be greater than 0")

    step_cm = args.rod_length_cm / args.photo_count
    positions = [idx * step_cm for idx in range(args.photo_count)]

    print("Stepper photo-position no-switch test")
    print("This assumes the physical axis is already at 0cm.")
    print("The micro switch is not opened or used by this test.")
    print(f"positions_cm={', '.join(f'{pos:.2f}' for pos in positions)}")
    print("return_cm=0.00")
    print(f"steps_per_cm={args.steps_per_cm}, up_direction={args.up_direction}")

    if not args.yes:
        wait_for_enter("Manually place the stepper axis at the 0cm/bottom starting position.")

    axis = StepDirAxisNoSwitch(
        gpiochip=args.gpiochip,
        step_line=args.step_line,
        dir_line=args.dir_line,
        steps_per_cm=args.steps_per_cm,
        up_direction=args.up_direction,
        step_high_time=args.step_high_time,
        step_low_time=args.step_low_time,
    )

    try:
        for idx, target_cm in enumerate(positions):
            print(f"\nphoto position {idx + 1}/{args.photo_count}")
            move_to_position(target_cm, args.pause_s)

        print("\nreturning to 0cm")
        move_to_position(0.0, args.pause_s)
        print("\nPASS: photo positions reached and returned to 0cm.")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
