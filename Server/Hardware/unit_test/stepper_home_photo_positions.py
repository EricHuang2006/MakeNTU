import argparse
import os
import signal
import sys
import time


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "MakeNTU_iMX93"))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from stepper_a4988_api import A4988Axis  # noqa: E402


axis = None


def close_axis():
    if axis is not None:
        axis.close()
    print("closed")


def cleanup(_signum=None, _frame=None):
    close_axis()
    raise SystemExit(0)


def wait_for_enter(message, assume_yes):
    if assume_yes:
        return
    input(f"{message}\nPress Enter when ready, or Ctrl+C to abort. ")


def print_switch(label):
    raw = axis.read_bottom_switch_raw()
    pressed = axis.bottom_switch_pressed()
    level = "HIGH" if raw else "LOW"
    print(f"{label}: raw={raw}, level={level}, pressed={pressed}")


def print_result(label, result):
    print(
        f"{label}: "
        f"steps={result.get('steps', 0)}, "
        f"home_steps={result.get('home_steps', 0)}, "
        f"backoff_steps={result.get('backoff_steps', 0)}, "
        f"moved_cm={result.get('moved_cm', 0.0):.2f}, "
        f"direction={result.get('direction', result.get('backoff_direction', '-'))}, "
        f"position_cm={result.get('position_cm', 0.0):.2f}, "
        f"homed={result.get('homed')}"
    )


def run_home():
    print("\n[1/4] Home to bottom switch and back off")
    print_switch("before home")
    result = axis.home_bottom()
    print_result("home", result)
    print_switch("after backoff")
    if not result.get("homed"):
        raise RuntimeError("Axis did not report homed=True")


def move_to_photo_position(index, total, target_cm, pause_s):
    print(f"\n[{index + 2}/4] Photo position {index + 1}/{total}: target={target_cm:.2f}cm")
    result = axis.move_to_x_cm(target_cm)
    print_result("move", result)
    print_switch("switch")
    time.sleep(pause_s)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stepper + switch test: home with bottom switch/backoff, then move through "
            "the three photo positions."
        )
    )
    parser.add_argument("--gpiochip", default="/dev/gpiochip0")
    parser.add_argument("--step-line", type=int, default=18)
    parser.add_argument("--dir-line", type=int, default=17)
    parser.add_argument("--bottom-switch-line", type=int, default=15)
    parser.add_argument("--bottom-switch-active-low", action="store_true", default=True)
    parser.add_argument(
        "--bottom-switch-active-high",
        dest="bottom_switch_active_low",
        action="store_false",
    )
    parser.add_argument("--steps-per-cm", type=float, default=1000.0)
    parser.add_argument("--rod-length-cm", type=float, default=21.0)
    parser.add_argument("--photo-count", type=int, default=3)
    parser.add_argument("--up-direction", type=int, choices=(0, 1), default=0)
    parser.add_argument("--home-direction", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-home-cm", type=float, default=24.0)
    parser.add_argument("--home-backoff-steps", type=int, default=200)
    parser.add_argument("--step-high-time", type=float, default=0.0005)
    parser.add_argument("--step-low-time", type=float, default=0.0010)
    parser.add_argument("--pause-s", type=float, default=1.0)
    parser.add_argument("--return-home", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Run without interactive confirmations.")
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

    print("Stepper home + photo-position sanity test")
    print("This test uses only the A4988 stepper output and bottom micro switch input.")
    print(f"gpiochip={args.gpiochip}")
    print(f"step_line={args.step_line}, dir_line={args.dir_line}")
    print(
        f"bottom_switch_line={args.bottom_switch_line}, "
        f"active_low={args.bottom_switch_active_low}"
    )
    print(f"home_direction={args.home_direction}, up_direction={args.up_direction}")
    print(f"home_backoff_steps={args.home_backoff_steps}")
    print(f"positions_cm={', '.join(f'{pos:.2f}' for pos in positions)}")

    wait_for_enter(
        "Confirm the axis can safely move toward the bottom switch and then up the rod.",
        args.yes,
    )

    axis = A4988Axis(
        gpiochip=args.gpiochip,
        step_line=args.step_line,
        dir_line=args.dir_line,
        bottom_switch_line=args.bottom_switch_line,
        bottom_switch_active_low=args.bottom_switch_active_low,
        step_high_time=args.step_high_time,
        step_low_time=args.step_low_time,
        steps_per_cm=args.steps_per_cm,
        up_direction=args.up_direction,
        home_direction=args.home_direction,
        max_home_cm=args.max_home_cm,
        home_backoff_steps=args.home_backoff_steps,
    )

    try:
        run_home()

        for idx, target_cm in enumerate(positions):
            move_to_photo_position(idx, args.photo_count, target_cm, args.pause_s)

        if args.return_home:
            print("\n[return] Moving back to 0.00cm base point")
            result = axis.move_to_x_cm(0.0)
            print_result("return", result)

        print("\nPASS: homed and reached all photo positions.")
    finally:
        close_axis()


if __name__ == "__main__":
    main()
