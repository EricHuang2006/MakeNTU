import fcntl
import os
import time


I2C_SLAVE = 0x0703
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06


class PCA9685:
    """
    PCA9685 PWM board driver.

    One board controls up to 16 PWM channels. Servos should share one PCA9685
    instance instead of each opening its own I2C file descriptor.
    """

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
        """
        Configure PCA9685 for 50 Hz PWM, suitable for hobby servos.
        """

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class Servo:
    """
    One physical servo connected to one PCA9685 channel.
    """

    def __init__(
        self,
        board,
        channel,
        min_angle=0,
        max_angle=180,
        min_us=500,
        max_us=2500,
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

    def center(self):
        return self.set_angle((self.min_angle + self.max_angle) / 2)

    def off(self):
        self.board.full_off(self.channel)


# Backward-compatible name for older code that imported ServoController.
ServoController = PCA9685
