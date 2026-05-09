import queue
import sys
import threading

from config import ENABLE_CLI_MODEL_INPUT
from event_logger import log_event


class CliModelInput:
    def __init__(self):
        self.enabled = bool(ENABLE_CLI_MODEL_INPUT)
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = None

        if not self.enabled:
            return

        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print(
            "[CLI] Commands: mode:1, mode:2, mode:3, "
            "gesture:pan_left, gesture:pan_right, gesture:tilt_up, "
            "gesture:tilt_down, gesture:height_up, gesture:height_down, "
            "gesture:finish"
        )

    def pop_frame_inputs(self):
        inputs = {}
        if not self.enabled:
            return inputs

        while True:
            try:
                key, value = self.events.get_nowait()
            except queue.Empty:
                break
            inputs[key] = value

        return inputs

    def close(self):
        self.stop_event.set()

    def _read_loop(self):
        while not self.stop_event.is_set():
            line = sys.stdin.readline()
            if line == "":
                return
            self._handle_line(line.strip())

    def _handle_line(self, line):
        if not line:
            return

        if ":" not in line:
            log_event(
                "error",
                f"Invalid CLI model input '{line}'. Use key:value, for example mode:3.",
                throttle_seconds=0.0,
            )
            return

        key, value = [part.strip().lower() for part in line.split(":", 1)]
        if key in ("mode", "sign"):
            if value not in ("1", "2", "3"):
                log_event("error", f"Invalid mode '{value}'. Expected 1, 2, or 3.", throttle_seconds=0.0)
                return
            self.events.put(("hand_sign", int(value)))
            log_event("api", f"CLI injected hand_sign={value}.", throttle_seconds=0.0)
            return

        if key in ("gesture", "manual"):
            valid_gestures = {
                "pan_left",
                "pan_right",
                "tilt_up",
                "tilt_down",
                "height_up",
                "height_down",
                "finish",
            }
            if value not in valid_gestures:
                log_event("error", f"Invalid gesture '{value}'.", throttle_seconds=0.0)
                return
            self.events.put(("manual_gesture", value))
            log_event("api", f"CLI injected manual_gesture={value}.", throttle_seconds=0.0)
            return

        log_event("error", f"Unknown CLI model input key '{key}'.", throttle_seconds=0.0)
