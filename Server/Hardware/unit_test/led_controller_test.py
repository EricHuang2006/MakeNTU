import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "MakeNTU_iMX93"))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from led_control import RgbLedController  # noqa: E402


def print_help():
    print("Commands:")
    print("  off")
    print("  red")
    print("  green")
    print("  blue")
    print("  yellow")
    print("  white")
    print("  blink <color> <seconds>")
    print("  q")


def main():
    led = RgbLedController()
    try:
        print_help()
        while True:
            command = input("led> ").strip().lower()
            parts = command.split()

            if command in ("q", "quit", "exit"):
                break

            if command == "help":
                print_help()
            elif command in ("off", "red", "green", "blue", "yellow", "white"):
                led.set_light(command)
            elif len(parts) == 3 and parts[0] == "blink":
                led.set_light(parts[1], pattern="blink", duration_s=float(parts[2]))
            else:
                print("Unknown command. Type 'help'.")
    finally:
        led.close()
        print("closed")


if __name__ == "__main__":
    main()
