import ctypes
import fcntl
import os
import time


GPIOHANDLES_MAX = 64
GPIOHANDLE_REQUEST_INPUT = 1 << 0
GPIOHANDLE_REQUEST_OUTPUT = 1 << 1
GPIO_GET_LINEHANDLE_IOCTL = 0xC16CB403
GPIOHANDLE_SET_LINE_VALUES_IOCTL = 0xC040B409
GPIOHANDLE_GET_LINE_VALUES_IOCTL = 0xC040B408


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


class A4988Axis:
    def __init__(
        self,
        gpiochip="/dev/gpiochip0",
        step_line=18,
        dir_line=17,
        bottom_switch_line=15,
        bottom_switch_active_low=True,
        step_high_time=0.0005,
        step_low_time=0.0010,
        steps_per_cm=1000.0,
        up_direction=1,
        home_direction=0,
        max_home_cm=24.0,
    ):
        if steps_per_cm <= 0:
            raise ValueError("steps_per_cm must be greater than 0")

        self.gpiochip = gpiochip
        self.step_line = int(step_line)
        self.dir_line = int(dir_line)
        self.bottom_switch_line = int(bottom_switch_line)
        self.bottom_switch_active_low = bool(bottom_switch_active_low)
        self.step_high_time = float(step_high_time)
        self.step_low_time = float(step_low_time)
        self.steps_per_cm = float(steps_per_cm)
        self.up_direction = 1 if int(up_direction) else 0
        self.home_direction = 1 if int(home_direction) else 0
        self.max_home_cm = float(max_home_cm)
        self.position_cm = 0.0
        self.homed = False

        self.motion_chip_fd = None
        self.motion_fd = None
        self.switch_chip_fd = None
        self.switch_fd = None
        self.current_step = 0
        self.current_dir = 0

        self.open()

    def open(self):
        if self.motion_fd is not None:
            return

        self.motion_chip_fd = os.open(self.gpiochip, os.O_RDONLY)
        self.switch_chip_fd = os.open(self.gpiochip, os.O_RDONLY)

        motion_req = gpiohandle_request()
        motion_req.lineoffsets[0] = self.step_line
        motion_req.lineoffsets[1] = self.dir_line
        motion_req.flags = GPIOHANDLE_REQUEST_OUTPUT
        motion_req.default_values[0] = 0
        motion_req.default_values[1] = 0
        motion_req.consumer_label = b"a4988-axis"
        motion_req.lines = 2

        switch_req = gpiohandle_request()
        switch_req.lineoffsets[0] = self.bottom_switch_line
        switch_req.flags = GPIOHANDLE_REQUEST_INPUT
        switch_req.consumer_label = b"a4988-bottom-switch"
        switch_req.lines = 1

        try:
            fcntl.ioctl(self.motion_chip_fd, GPIO_GET_LINEHANDLE_IOCTL, motion_req)
            fcntl.ioctl(self.switch_chip_fd, GPIO_GET_LINEHANDLE_IOCTL, switch_req)
        except Exception:
            self.close()
            raise

        self.motion_fd = motion_req.fd
        self.switch_fd = switch_req.fd
        self.write_motion_lines(0, 0)

    def close(self):
        for fd_name in ("motion_fd", "switch_fd", "motion_chip_fd", "switch_chip_fd"):
            fd = getattr(self, fd_name)
            if fd is not None:
                try:
                    if fd_name == "motion_fd":
                        self.write_motion_lines(0, self.current_dir)
                    os.close(fd)
                except Exception:
                    pass
                setattr(self, fd_name, None)

    def write_motion_lines(self, step_value, dir_value):
        if self.motion_fd is None:
            raise RuntimeError("A4988Axis is closed")

        self.current_step = 1 if step_value else 0
        self.current_dir = 1 if dir_value else 0
        data = gpiohandle_data()
        data.values[0] = self.current_step
        data.values[1] = self.current_dir
        fcntl.ioctl(self.motion_fd, GPIOHANDLE_SET_LINE_VALUES_IOCTL, data)

    def read_bottom_switch_raw(self):
        if self.switch_fd is None:
            raise RuntimeError("A4988Axis is closed")

        data = gpiohandle_data()
        fcntl.ioctl(self.switch_fd, GPIOHANDLE_GET_LINE_VALUES_IOCTL, data)
        return int(data.values[0])

    def bottom_switch_pressed(self):
        raw = self.read_bottom_switch_raw()
        if self.bottom_switch_active_low:
            return raw == 0
        return raw == 1

    def set_dir(self, dir_value):
        self.write_motion_lines(self.current_step, dir_value)

    def set_step(self, step_value):
        self.write_motion_lines(step_value, self.current_dir)

    def step_once(self):
        self.set_step(1)
        time.sleep(self.step_high_time)
        self.set_step(0)
        time.sleep(self.step_low_time)

    def move_steps(self, steps, direction, stop_on_bottom_switch=False):
        steps = int(round(abs(steps)))
        if steps == 0:
            return 0

        self.set_dir(direction)
        time.sleep(0.001)

        moved = 0
        for _ in range(steps):
            if stop_on_bottom_switch and self.bottom_switch_pressed():
                break
            self.step_once()
            moved += 1

        return moved

    def cm_to_steps(self, distance_cm):
        return int(round(abs(float(distance_cm)) * self.steps_per_cm))

    def adjust_x_cm(self, distance_cm):
        distance_cm = float(distance_cm)
        steps = self.cm_to_steps(distance_cm)
        if steps == 0:
            return self._result(distance_cm, 0, self.current_dir, 0.0)

        direction = self.up_direction if distance_cm > 0 else 1 - self.up_direction
        moved = self.move_steps(steps, direction)
        moved_cm = moved / self.steps_per_cm
        if distance_cm < 0:
            moved_cm = -moved_cm
        self.position_cm += moved_cm
        return self._result(distance_cm, moved, direction, moved_cm)

    def move_to_x_cm(self, target_cm):
        return self.adjust_x_cm(float(target_cm) - self.position_cm)

    def home_bottom(self):
        max_steps = self.cm_to_steps(self.max_home_cm)
        if self.bottom_switch_pressed():
            moved = 0
        else:
            moved = self.move_steps(
                max_steps,
                self.home_direction,
                stop_on_bottom_switch=True,
            )

        pressed = self.bottom_switch_pressed()
        if not pressed:
            raise RuntimeError("Bottom micro switch was not reached during homing")

        self.position_cm = 0.0
        self.homed = True
        return {
            "homed": True,
            "bottom_switch_pressed": pressed,
            "steps": moved,
            "position_cm": self.position_cm,
        }

    def _result(self, requested_cm, steps, direction, moved_cm):
        return {
            "requested_cm": requested_cm,
            "moved_cm": moved_cm,
            "steps": steps,
            "direction": direction,
            "position_cm": self.position_cm,
            "homed": self.homed,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
