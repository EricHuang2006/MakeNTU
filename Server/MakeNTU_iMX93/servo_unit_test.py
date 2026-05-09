import argparse
import time

from motor_control import MirroredTiltTestRig


def print_state(state):
    print(
        "state: "
        f"tilt_left={state['tilt_left']:.1f} "
        f"tilt_right={state['tilt_right']:.1f} "
        f"left_right={state['left_right']:.1f}"
    )


def print_help():
    print("Commands:")
    print("  center              center CH0, CH4, CH8")
    print("  tilt <angle>        mirrored tilt; example: tilt 100 -> CH0=100, CH4=80")
    print("  delta <degrees>     mirrored tilt relative to center; example: delta -10")
    print("  lr <angle>          move left-right servo CH8")
    print("  raw <ch> <angle>    move CH0, CH4, or CH8 directly")
    print("  sweep tilt          quick mirrored tilt sweep")
    print("  sweep lr            quick CH8 sweep")
    print("  off                 turn PWM off for CH0, CH4, CH8")
    print("  status              show last commanded angles")
    print("  help")
    print("  quit")


def run_command(rig, command):
    parts = command.split()
    if not parts:
        return True

    name = parts[0].lower()

    if name in ("q", "quit", "exit"):
        return False

    if name == "help":
        print_help()
        return True

    if name == "status":
        print_state(rig.current)
        return True

    if name == "center":
        rig.center()
        print("centered")
        print_state(rig.current)
        return True

    if name == "off":
        rig.off()
        print("PWM off")
        print_state(rig.current)
        return True

    if name == "tilt" and len(parts) == 2:
        state = rig.set_tilt(float(parts[1]))
        print_state(state)
        return True

    if name == "delta" and len(parts) == 2:
        state = rig.set_tilt_delta(float(parts[1]))
        print_state(state)
        return True

    if name in ("lr", "left-right", "left_right") and len(parts) == 2:
        state = rig.set_left_right(float(parts[1]))
        print_state(state)
        return True

    if name == "raw" and len(parts) == 3:
        state = rig.set_raw(int(parts[1]), float(parts[2]))
        print_state(state)
        return True

    if name == "sweep" and len(parts) == 2:
        run_sweep(rig, parts[1].lower())
        return True

    print("Unknown command. Type 'help' for commands.")
    return True


def run_sweep(rig, target):
    if target == "tilt":
        print("sweeping mirrored tilt pair")
        for angle in (70, 90, 110, 90):
            print(f"tilt {angle}")
            print_state(rig.set_tilt(angle))
            time.sleep(0.5)
        return

    if target in ("lr", "left-right", "left_right"):
        print("sweeping left-right servo CH8")
        for angle in (70, 90, 110, 90):
            print(f"lr {angle}")
            print_state(rig.set_left_right(angle))
            time.sleep(0.5)
        return

    print("Usage: sweep tilt OR sweep lr")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Board-local CLI test for mirrored tilt servos CH0/CH4 and left-right servo CH8."
    )
    parser.add_argument("--center", type=float, default=90)
    parser.add_argument("--left-tilt-channel", type=int, default=0)
    parser.add_argument("--right-tilt-channel", type=int, default=4)
    parser.add_argument("--left-right-channel", type=int, default=8)
    parser.add_argument("command", nargs="*", help="Optional one-shot command, e.g. tilt 100")
    return parser.parse_args()


def main():
    args = parse_args()
    rig = MirroredTiltTestRig(
        left_tilt_channel=args.left_tilt_channel,
        right_tilt_channel=args.right_tilt_channel,
        left_right_channel=args.left_right_channel,
        center_angle=args.center,
    )

    if not rig.enabled:
        raise SystemExit("Servo unit test cannot start because the rig is disabled.")

    try:
        if args.command:
            try:
                run_command(rig, " ".join(args.command))
            except ValueError as exc:
                print(f"Invalid value: {exc}")
            return

        print_help()
        print()
        print_state(rig.current)

        while True:
            try:
                command = input("servo> ").strip()
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
