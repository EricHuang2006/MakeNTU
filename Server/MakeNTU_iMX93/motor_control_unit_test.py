import argparse
import fcntl
import os
import time

MOTOR_I2C_BUS = 0
MOTOR_I2C_ADDRESS = 0x40
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
        self.angle = center

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
    def __init__(self, bus=MOTOR_I2C_BUS, address=MOTOR_I2C_ADDRESS):
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
            self.board = PCA9685(bus=bus, address=address)
            self.pan1 = Servo(self.board, 0, 500, 2500, 20, 160, DEFAULT_PAN_ANGLE)
            self.tilt1 = Servo(self.board, 4, 500, 2500, 20, 160, DEFAULT_TILT_ANGLE)
            self.tilt2 = Servo(self.board, 8, 500, 2441, 20, 160, DEFAULT_TILT_PAIR_ANGLE)
            self.servos = [self.pan1, self.tilt1, self.tilt2]
            self.enabled = True
            self.center()
            print("CameraServoRig initialized.")
        except Exception as exc:
            print(f"Failed to initialize motor rig: {exc}")
            self.enabled = False
            self.board = None
            self.servos = []

    def _clamp_relative_angle(self, servo, angle, max_delta):
        lower = max(servo.min_angle, servo.center - max_delta)
        upper = min(servo.max_angle, servo.center + max_delta)
        return max(lower, min(upper, float(angle)))

    def reset_all(self):
        if not self.enabled:
            return
        self.pan(self.pan1.center)
        self.tilt(self.tilt1.center)

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
        angle = self._clamp_relative_angle(self.pan1, angle, PAN_MAX_DELTA)
        while self.current["pan"] != angle:
            delta = min(2.0, abs(self.current["pan"] - angle))
            delta = delta if angle > self.current["pan"] else -delta
            to = self.current["pan"] + delta
            self.pan1.set_angle(to)
            self.current["pan"] += delta
            time.sleep(0.)

    def tilt(self, angle):
        if not self.enabled:
            return
        angle = self._clamp_relative_angle(self.tilt1, angle, TILT_MAX_DELTA)
        while self.current["tilt"] != angle:
            delta = min(2.0, abs(self.current["tilt"] - angle))
            delta = delta if angle > self.current["tilt"] else -delta
            to = self.current["tilt"] + delta
            dc = to - self.tilt1.center
            self.tilt1.set_angle(to)
            angle2 = self.tilt2.center - (dc * (0.9 if dc > 0 else 1.1))
            self.tilt2.set_angle(angle2)
            self.current["tilt"] += delta
            time.sleep(abs(delta) / 4 * 0.05)

        #     def set_angle(self, angle):
        # angle = max(self.min_angle, min(self.max_angle, float(angle)))
        # print(f"set angle : {self.angle} -> {angle}")
        # while self.angle != angle:
        #     delta = min(5.0, abs(angle - self.angle))
        #     print(f"delta : {delta}")
        #     delta = delta if angle > self.angle else -delta
        #     print(f"aft_delta : {delta}")
        #     pulse_us = self.angle_to_us(self.angle + delta)
        #     self.set_us(pulse_us)
        #     self.angle += delta
        #     time.sleep(0.2)



    def set_angles(self, pan=None, tilt=None, height=None):
        if pan is not None:
            self.pan(pan)
        if tilt is not None:
            self.tilt(tilt)
        if height is not None:
            self.current["height"] = max(0.0, min(180.0, float(height)))

    def raw(self, channel, angle):
        for servo in self.servos:
            if servo.channel == int(channel):
                servo.set_angle(angle)
                return
        raise ValueError("raw channel must be one of 0, 4, or 8")

    def shutdown(self):
        if not self.enabled:
            return
        for servo in self.servos:
            servo.off()
        if self.board is not None:
            self.board.close()
        self.enabled = False


def print_state(rig):
    paired_angle = rig.tilt2.angle if rig.tilt2 is not None else None
    paired_text = f"{paired_angle:.1f}" if paired_angle is not None else "None"
    print(
        "state: "
        f"pan={rig.current['pan']:.1f} "
        f"tilt={rig.current['tilt']:.1f} "
        f"tilt_pair_ch8={paired_text} "
        f"height={rig.current['height']:.1f}"
    )


def print_help():
    print("Commands:")
    print("  center")
    print("  pan <angle>          move pan servo CH0")
    print("  tilt <angle>         move tilt pair CH4/CH8 in opposite directions")
    print("  set <pan> <tilt>     move pan and tilt together")
    print("  raw <ch> <angle>     direct CH0, CH4, or CH8 test")
    print("  sweep pan")
    print("  sweep tilt")
    print("  off")
    print("  status")
    print("  help")
    print("  quit")


def run_sweep(rig, target):
    if target == "pan":
        for angle in (60, 85, 110, 85):
            print(f"pan {angle}")
            rig.pan(angle)
            print_state(rig)
            time.sleep(0.4)
        return

    if target == "tilt":
        for angle in (74, 94, 114, 94):
            print(f"tilt {angle}")
            rig.tilt(angle)
            print_state(rig)
            time.sleep(0.4)
        return

    print("Usage: sweep pan OR sweep tilt")


def run_command(rig, command):
    parts = command.split()
    if not parts:
        return True

    name = parts[0].lower()

    if name in ("q", "quit", "exit"):
        return False

    if name == "help":
        print_help()
    elif name == "status":
        print_state(rig)
    elif name == "center":
        rig.center()
        print_state(rig)
    elif name == "off":
        rig.shutdown()
        print("PWM off and I2C closed")
        return False
    elif name == "pan" and len(parts) == 2:
        rig.pan(float(parts[1]))
        print_state(rig)
    elif name == "tilt" and len(parts) == 2:
        rig.tilt(float(parts[1]))
        print_state(rig)
    elif name == "set" and len(parts) == 3:
        rig.set_angles(pan=float(parts[1]), tilt=float(parts[2]))
        print_state(rig)
    elif name == "raw" and len(parts) == 3:
        rig.raw(int(parts[1]), float(parts[2]))
        print_state(rig)
    elif name == "sweep" and len(parts) == 2:
        run_sweep(rig, parts[1].lower())
    else:
        print("Unknown command. Type 'help'.")

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone board-side unit test for motor_control.py servo logic."
    )
    parser.add_argument("--bus", type=int, default=MOTOR_I2C_BUS)
    parser.add_argument("--address", type=lambda value: int(value, 0), default=MOTOR_I2C_ADDRESS)
    parser.add_argument("command", nargs="*", help="Optional one-shot command, e.g. pan 90")
    return parser.parse_args()


def main():
    args = parse_args()
    rig = CameraServoRig(bus=args.bus, address=args.address)

    if not rig.enabled:
        raise SystemExit("Motor unit test cannot start because the rig is disabled.")

    try:
        if args.command:
            run_command(rig, " ".join(args.command))
            return

        print_help()
        print()
        print_state(rig)

        while True:
            try:
                command = input("motor> ").strip()
            except EOFError:
                break

            try:
                keep_running = run_command(rig, command)
            except ValueError as exc:
                print(f"Invalid value: {exc}")
                keep_running = True

            if not keep_running:
                break
    finally:
        rig.shutdown()
        print("closed")


if __name__ == "__main__":
    main()
