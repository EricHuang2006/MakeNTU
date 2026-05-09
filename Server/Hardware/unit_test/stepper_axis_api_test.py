import argparse
import os
import signal
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "MakeNTU_iMX93"))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from stepper_a4988_api import A4988Axis  # noqa: E402


axis = None


def cleanup(_signum=None, _frame=None):
    if axis is not None:
        axis.close()
    print("closed")
    raise SystemExit(0)


def print_help():
    print("Commands:")
    print("  home")
    print("  cm <distance>        example: cm 7 or cm -7")
    print("  goto <position_cm>   example: goto 14")
    print("  switch")
    print("  status")
    print("  quit")


def print_result(result):
    print(
        "result: "
        f"homed={result.get('homed')} "
        f"requested_cm={result.get('requested_cm', 0):.2f} "
        f"moved_cm={result.get('moved_cm', 0):.2f} "
        f"steps={result.get('steps', 0)} "
        f"position_cm={result.get('position_cm', 0):.2f}"
    )


def run_command(command):
    parts = command.split()
    if not parts:
        return True

    name = parts[0].lower()
    if name in ("q", "quit", "exit"):
        return False

    if name == "help":
        print_help()
    elif name == "home":
        print_result(axis.home_bottom())
    elif name == "cm" and len(parts) == 2:
        print_result(axis.adjust_x_cm(float(parts[1])))
    elif name == "goto" and len(parts) == 2:
        print_result(axis.move_to_x_cm(float(parts[1])))
    elif name == "switch":
        print(f"bottom_switch_pressed={axis.bottom_switch_pressed()} raw={axis.read_bottom_switch_raw()}")
    elif name == "status":
        print(f"homed={axis.homed} position_cm={axis.position_cm:.2f}")
    else:
        print("Unknown command. Type 'help'.")

    return True


def parse_args():
    parser = argparse.ArgumentParser(description="A4988 axis API unit test.")
    parser.add_argument("--gpiochip", default="/dev/gpiochip0")
    parser.add_argument("--step-line", type=int, default=2)
    parser.add_argument("--dir-line", type=int, default=3)
    parser.add_argument("--bottom-switch-line", type=int, default=4)
    parser.add_argument("--bottom-switch-active-low", action="store_true", default=True)
    parser.add_argument("--bottom-switch-active-high", dest="bottom_switch_active_low", action="store_false")
    parser.add_argument("--steps-per-cm", type=float, default=1000.0)
    parser.add_argument("--up-direction", type=int, choices=(0, 1), default=1)
    parser.add_argument("--home-direction", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-home-cm", type=float, default=24.0)
    parser.add_argument("command", nargs="*", help="Optional one-shot command, e.g. home")
    return parser.parse_args()


def main():
    global axis

    args = parse_args()
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    axis = A4988Axis(
        gpiochip=args.gpiochip,
        step_line=args.step_line,
        dir_line=args.dir_line,
        bottom_switch_line=args.bottom_switch_line,
        bottom_switch_active_low=args.bottom_switch_active_low,
        steps_per_cm=args.steps_per_cm,
        up_direction=args.up_direction,
        home_direction=args.home_direction,
        max_home_cm=args.max_home_cm,
    )

    try:
        if args.command:
            run_command(" ".join(args.command))
            return

        print_help()
        while True:
            try:
                command = input("axis> ").strip()
            except EOFError:
                break

            try:
                keep_running = run_command(command)
            except ValueError as exc:
                print(f"Invalid value: {exc}")
                keep_running = True

            if not keep_running:
                break
    finally:
        cleanup()


if __name__ == "__main__":
    main()
