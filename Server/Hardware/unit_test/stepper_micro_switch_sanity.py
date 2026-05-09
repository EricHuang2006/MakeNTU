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


def cleanup(_signum=None, _frame=None):
    if axis is not None:
        axis.close()
    print("closed")
    raise SystemExit(0)


def wait_for_enter(message):
    input(f"{message}\nPress Enter when ready, or Ctrl+C to abort. ")


def read_switch(label):
    raw = axis.read_bottom_switch_raw()
    pressed = axis.bottom_switch_pressed()
    print(f"{label}: raw={raw}, pressed={pressed}")
    return pressed


def require_switch_state(expected_pressed, label):
    pressed = read_switch(label)
    if pressed != expected_pressed:
        expected = "pressed" if expected_pressed else "released"
        actual = "pressed" if pressed else "released"
        raise RuntimeError(f"Expected switch {expected}, but it reads {actual}")


def run_switch_manual_test():
    print("\n[1/4] Micro switch manual sanity test")
    wait_for_enter("Make sure the bottom switch is NOT pressed.")
    require_switch_state(False, "released check")

    wait_for_enter("Press and hold the bottom switch by hand.")
    require_switch_state(True, "pressed check")

    wait_for_enter("Release the bottom switch.")
    require_switch_state(False, "released again check")


def run_tiny_motion_test(test_steps, direction):
    print("\n[2/4] Tiny motion test")
    print(f"About to move {test_steps} step(s), direction={direction}.")
    wait_for_enter("Keep one hand near power. Confirm the axis has room to move.")
    moved = axis.move_steps(test_steps, direction)
    print(f"moved_steps={moved}, current_dir={axis.current_dir}")


def run_switch_stop_test(test_steps, direction):
    print("\n[3/4] Stop-on-switch test")
    print(
        "This test moves slowly toward the switch direction and should stop when you press "
        "the switch by hand."
    )
    wait_for_enter("Prepare to press the switch by hand immediately after motion starts.")
    moved = axis.move_steps(test_steps, direction, stop_on_bottom_switch=True)
    print(f"stopped_after_steps={moved}, switch_pressed={axis.bottom_switch_pressed()}")
    if moved >= test_steps and not axis.bottom_switch_pressed():
        raise RuntimeError("Stepper did not stop on the switch during the bounded stop test")


def run_home_test():
    print("\n[4/4] Bounded home_bottom test")
    wait_for_enter(
        "Release the switch. The axis will home toward the bottom and stop at the switch."
    )
    result = axis.home_bottom()
    print(
        "home result: "
        f"homed={result['homed']} "
        f"steps={result['steps']} "
        f"position_cm={result['position_cm']:.2f} "
        f"switch_pressed={result['bottom_switch_pressed']}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conservative sanity test for A4988 stepper + bottom micro switch."
    )
    parser.add_argument("--gpiochip", default="/dev/gpiochip0")
    parser.add_argument("--step-line", type=int, default=2)
    parser.add_argument("--dir-line", type=int, default=3)
    parser.add_argument("--bottom-switch-line", type=int, default=4)
    parser.add_argument("--bottom-switch-active-low", action="store_true", default=True)
    parser.add_argument("--bottom-switch-active-high", dest="bottom_switch_active_low", action="store_false")
    parser.add_argument("--steps-per-cm", type=float, default=1000.0)
    parser.add_argument("--up-direction", type=int, choices=(0, 1), default=1)
    parser.add_argument("--home-direction", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-home-cm", type=float, default=2.0)
    parser.add_argument("--tiny-steps", type=int, default=50)
    parser.add_argument("--stop-test-steps", type=int, default=2000)
    parser.add_argument("--skip-motion", action="store_true")
    parser.add_argument("--skip-home", action="store_true")
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

    print("Stepper + micro switch sanity test")
    print(f"gpiochip={args.gpiochip}")
    print(f"step_line={args.step_line}, dir_line={args.dir_line}")
    print(
        f"bottom_switch_line={args.bottom_switch_line}, "
        f"active_low={args.bottom_switch_active_low}"
    )
    print(f"home_direction={args.home_direction}, up_direction={args.up_direction}")
    print(f"max_home_cm={args.max_home_cm}, steps_per_cm={args.steps_per_cm}")

    try:
        run_switch_manual_test()

        if not args.skip_motion:
            run_tiny_motion_test(args.tiny_steps, args.up_direction)
            run_switch_stop_test(args.stop_test_steps, args.home_direction)

        if not args.skip_home:
            run_home_test()

        print("\nPASS: stepper and bottom switch sanity checks completed.")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
