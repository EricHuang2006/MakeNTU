import os
import time
import fcntl
import signal
import sys

I2C_BUS = 0
PCA9685_ADDR = 0x40

# 目前只開這兩個 channel
SERVO_CHANNELS = [0, 4]

# 之後要加第三個，例如 CH8，就改成：
# SERVO_CHANNELS = [0, 4, 8]

MIN_ANGLE = 0
MAX_ANGLE = 180

# 這裡用你目前比較可能需要的校正範圍
# 如果 2000 us 只到 90 度，可以先試 MAX_US = 2500 或更高
MIN_US = 500
MAX_US = 2500

I2C_SLAVE = 0x0703
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

fd = os.open(f"/dev/i2c-{I2C_BUS}", os.O_RDWR)
fcntl.ioctl(fd, I2C_SLAVE, PCA9685_ADDR)


def write_reg(reg, val):
    os.write(fd, bytes([reg & 0xFF, val & 0xFF]))


def set_pwm_freq_50hz():
    write_reg(MODE1, 0x10)
    write_reg(PRESCALE, 0x79)
    write_reg(MODE1, 0x20)
    time.sleep(0.01)
    write_reg(MODE1, 0xA0)


def set_pwm(channel, on_count, off_count):
    if channel not in SERVO_CHANNELS:
        raise ValueError(f"Only channels {SERVO_CHANNELS} are enabled")

    base = LED0_ON_L + 4 * channel

    write_reg(base + 0, on_count & 0xFF)
    write_reg(base + 1, (on_count >> 8) & 0x0F)
    write_reg(base + 2, off_count & 0xFF)
    write_reg(base + 3, (off_count >> 8) & 0x0F)


def set_servo_us(channel, pulse_us):
    ticks = int(pulse_us * 4096 / 20000)
    set_pwm(channel, 0, ticks)


def angle_to_us(angle):
    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
    return MIN_US + (MAX_US - MIN_US) * angle / (MAX_ANGLE - MIN_ANGLE)


def set_servo_angle(channel, angle):
    pulse_us = angle_to_us(angle)
    set_servo_us(channel, pulse_us)
    print(f"CH{channel}: angle={angle:.1f} deg, pulse={pulse_us:.1f} us")


def full_off(channel):
    base = LED0_ON_L + 4 * channel
    write_reg(base + 0, 0x00)
    write_reg(base + 1, 0x00)
    write_reg(base + 2, 0x00)
    write_reg(base + 3, 0x10)


def enabled_off():
    for ch in SERVO_CHANNELS:
        full_off(ch)


def cleanup(signum=None, frame=None):
    print("\nTurning off enabled channels...")
    try:
        enabled_off()
    finally:
        os.close(fd)
    print("Done.")
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

set_pwm_freq_50hz()

print("PCA9685 servo controller: CH0 and CH4")
print("Commands:")
print("  0 <angle>      example: 0 90")
print("  4 <angle>      example: 4 45")
print("  both <angle>   example: both 90")
print("  off 0")
print("  off 4")
print("  q")
print()

# 啟動時先把兩顆放到中間
for ch in SERVO_CHANNELS:
    set_servo_angle(ch, 90)

while True:
    cmd = input("servo> ").strip()
    parts = cmd.split()

    if cmd.lower() in ["q", "quit", "exit"]:
        cleanup()

    if len(parts) == 2 and parts[0].lower() == "both":
        try:
            angle = float(parts[1])
            for ch in SERVO_CHANNELS:
                set_servo_angle(ch, angle)
        except ValueError:
            print("Usage: both 90")
        continue

    if len(parts) == 2 and parts[0].lower() == "off":
        try:
            ch = int(parts[1])
            if ch not in SERVO_CHANNELS:
                print(f"Only channels {SERVO_CHANNELS} are enabled")
                continue
            full_off(ch)
            print(f"CH{ch} off")
        except ValueError:
            print("Usage: off 0")
        continue

    if len(parts) == 2:
        try:
            ch = int(parts[0])
            angle = float(parts[1])

            if ch not in SERVO_CHANNELS:
                print(f"Only channels {SERVO_CHANNELS} are enabled")
                continue

            set_servo_angle(ch, angle)

        except ValueError:
            print("Usage: 0 90 or 4 90")
        continue

    print("Unknown command. Example: 0 90")
