import fcntl
import os
import time

from config import MOTOR_I2C_ADDRESS, MOTOR_I2C_BUS
from event_logger import log_event


I2C_SLAVE = 0x0703
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

DEFAULT_PAN_ANGLE = 85.0
DEFAULT_TILT_ANGLE = 94.0
DEFAULT_TILT_PAIR_ANGLE = 81.0
DEFAULT_HEIGHT_ANGLE = 90.0

PAN_MAX_DELTA = 85.0
TILT_MAX_DELTA = 45.0


class PCA9685:
    def __init__(self, bus=0, address=0x40, init_50hz=True):
        self.bus = bus
        self.address = address
        self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, address)

        if init_50hz:
            self.set_pwm_freq_50hz()

    def close(self):
        if self.fd is None:
            return
        os.close(self.fd)
        self.fd = None

    def write_reg(self, reg, value):
        if self.fd is None:
            raise RuntimeError("PCA9685 is closed")
        os.write(self.fd, bytes([reg & 0xFF, value & 0xFF]))

    def set_pwm_freq_50hz(self):
        self.write_reg(MODE1, 0x10)
        self.write_reg(PRESCALE, 0x79)
        self.write_reg(MODE1, 0x20)
        time.sleep(0.01)
        self.write_reg(MODE1, 0xA0)

    def validate_channel(self, channel):
        if not 0 <= channel <= 15:
            raise ValueError("PCA9685 channel must be in range 0..15")

    def set_pwm(self, channel, on_count, off_count):
        self.validate_channel(channel)
        base = LED0_ON_L + 4 * channel
        self.write_reg(base + 0, on_count & 0xFF)
        self.write_reg(base + 1, (on_count >> 8) & 0x0F)
        self.write_reg(base + 2, off_count & 0xFF)
        self.write_reg(base + 3, (off_count >> 8) & 0x0F)

    def full_off(self, channel):
        self.validate_channel(channel)
        base = LED0_ON_L + 4 * channel
        self.write_reg(base + 0, 0x00)
        self.write_reg(base + 1, 0x00)
        self.write_reg(base + 2, 0x00)
        self.write_reg(base + 3, 0x10)


class Servo:
    def __init__(
        self,
        board,
        channel,
        min_us=500,
        max_us=2500,
        min_angle=0,
        max_angle=180,
        center=90,
    ):
        board.validate_channel(channel)

        if max_angle <= min_angle:
            raise ValueError("max_angle must be greater than min_angle")

        self.board = board
        self.channel = channel
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_us = min_us
        self.max_us = max_us
        self.center = center
        self.angle = None

    def angle_to_us(self, angle):
        angle = max(self.min_angle, min(self.max_angle, angle))
        ratio = (angle - self.min_angle) / (self.max_angle - self.min_angle)
        return self.min_us + (self.max_us - self.min_us) * ratio

    def set_us(self, pulse_us):
        ticks = int(pulse_us * 4096 / 20000)
        ticks = max(0, min(4095, ticks))
        self.board.set_pwm(self.channel, 0, ticks)

    def set_angle(self, angle):
        angle = max(self.min_angle, min(self.max_angle, angle))
        pulse_us = self.angle_to_us(angle)
        self.set_us(pulse_us)
        self.angle = angle
        return pulse_us

    def off(self):
        self.board.full_off(self.channel)


class CameraServoRig:
    def __init__(self):
        self.enabled = False
        self.board = None
        self.pan1 = None
        self.tilt1 = None
        self.tilt2 = None
        self.servos = []
        self.current = {
            "pan": DEFAULT_PAN_ANGLE,
            "tilt": DEFAULT_TILT_ANGLE,
            "height": DEFAULT_HEIGHT_ANGLE,
        }

        try:
            self.board = PCA9685(bus=MOTOR_I2C_BUS, address=MOTOR_I2C_ADDRESS)
            self.pan1 = Servo(self.board, 0, 500, 2500, 20, 160, DEFAULT_PAN_ANGLE)
            self.tilt1 = Servo(self.board, 4, 500, 2500, 20, 160, DEFAULT_TILT_ANGLE)
            self.tilt2 = Servo(self.board, 8, 500, 2441, 20, 160, DEFAULT_TILT_PAIR_ANGLE)
            self.servos = [self.pan1, self.tilt1, self.tilt2]
            self.enabled = True
            self.center()
            log_event("system", "CameraServoRig initialized.", throttle_seconds=0.0)
        except Exception as exc:
            log_event("error", f"Failed to initialize motor rig: {exc}", throttle_seconds=0.0)
            self.enabled = False
            self.board = None
            self.servos = []

    def _clamp_relative_angle(self, servo, angle, max_delta):
        lower = max(servo.min_angle, servo.center - max_delta)
        upper = min(servo.max_angle, servo.center + max_delta)
        return max(lower, min(upper, angle))

    def reset_all(self):
        if not self.enabled:
            return
        for servo in self.servos:
            servo.set_angle(servo.center)

    def center(self):
        if not self.enabled:
            return
        self.reset_all()
        self.current["pan"] = float(self.pan1.center)
        self.current["tilt"] = float(self.tilt1.center)
        self.current["height"] = DEFAULT_HEIGHT_ANGLE
        time.sleep(0.08)

    def pan(self, angle):
        if not self.enabled:
            return
        safe_angle = self._clamp_relative_angle(self.pan1, angle, PAN_MAX_DELTA)
        self.pan1.set_angle(safe_angle)
        self.current["pan"] = safe_angle

    def tilt(self, angle):
        if not self.enabled:
            return
        safe_angle = self._clamp_relative_angle(self.tilt1, angle, TILT_MAX_DELTA)
        delta = safe_angle - self.tilt1.center
        op = 0.9 if delta > 0 else 1.1
        self.tilt1.set_angle(safe_angle)
        paired_tilt = self.tilt2.center - (delta * op)
        paired_tilt = self._clamp_relative_angle(self.tilt2, paired_tilt, TILT_MAX_DELTA)
        self.tilt2.set_angle(paired_tilt)
        self.current["tilt"] = safe_angle

    def set_angles(self, pan=None, tilt=None, height=None):
        if pan is not None:
            self.pan(pan)
        if tilt is not None:
            self.tilt(tilt)
        if height is not None:
            self.current["height"] = max(0.0, min(180.0, float(height)))

    def shutdown(self):
        if not self.enabled:
            return
        for servo in self.servos:
            servo.off()
        if self.board is not None:
            self.board.close()
        self.enabled = False


# Backward-compatible alias for existing imports.
ServoControl = CameraServoRig
