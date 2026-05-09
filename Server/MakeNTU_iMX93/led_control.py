import threading
import time

from config import (
    ENABLE_LED_OUTPUT,
    LED_B_LINE,
    LED_BLINK_INTERVAL_SECONDS,
    LED_COMMON_ANODE,
    LED_G_LINE,
    LED_GPIOCHIP,
    LED_R_LINE,
)
from event_logger import log_event


try:
    import gpiod
    from gpiod.line import Direction, Value
except ImportError:
    gpiod = None
    Direction = None
    Value = None


COLOR_CHANNELS = {
    "off": (0, 0, 0),
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "yellow": (1, 1, 0),
    "white": (1, 1, 1),
}


class RgbLedController:
    def __init__(self):
        self.enabled = False
        self.lines = None
        self.line_ids = {
            "r": LED_R_LINE,
            "g": LED_G_LINE,
            "b": LED_B_LINE,
        }
        self.lock = threading.Lock()
        self.blink_stop = threading.Event()
        self.blink_thread = None

        if not ENABLE_LED_OUTPUT:
            log_event("system", "LED output disabled by config.", throttle_seconds=0.0)
            return

        if gpiod is None:
            log_event("error", "LED output unavailable: gpiod import failed.", throttle_seconds=0.0)
            return

        off_value = self._gpio_value(False)
        config = {
            LED_R_LINE: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=off_value),
            LED_G_LINE: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=off_value),
            LED_B_LINE: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=off_value),
        }

        try:
            self.lines = gpiod.request_lines(
                LED_GPIOCHIP,
                consumer="makentu-rgb-led",
                config=config,
            )
            self.enabled = True
            self.set_light("off")
            log_event("system", "RGB LED controller initialized.", throttle_seconds=0.0)
        except Exception as exc:
            self.lines = None
            log_event("error", f"Failed to initialize RGB LED: {exc}", throttle_seconds=0.0)

    def _gpio_value(self, on):
        if LED_COMMON_ANODE:
            return Value.INACTIVE if on else Value.ACTIVE
        return Value.ACTIVE if on else Value.INACTIVE

    def _color_values(self, color):
        channels = COLOR_CHANNELS.get(str(color).lower(), COLOR_CHANNELS["off"])
        return {
            LED_R_LINE: self._gpio_value(bool(channels[0])),
            LED_G_LINE: self._gpio_value(bool(channels[1])),
            LED_B_LINE: self._gpio_value(bool(channels[2])),
        }

    def _set_color_now(self, color):
        if not self.enabled or self.lines is None:
            return
        self.lines.set_values(self._color_values(color))

    def _stop_blink(self):
        self.blink_stop.set()
        thread = self.blink_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.2)
        self.blink_thread = None
        self.blink_stop.clear()

    def set_light(self, color, pattern="solid", duration_s=0.0, blink_interval_s=None):
        color = str(color).lower()
        pattern = str(pattern).lower()

        if color not in COLOR_CHANNELS:
            log_event("error", f"Unknown LED color={color}; using off.", throttle_seconds=0.0)
            color = "off"

        if not self.enabled:
            log_event(
                "api",
                f"Simulated LED color={color}, pattern={pattern}, duration={duration_s:.1f}s",
                throttle_seconds=0.0,
            )
            return

        with self.lock:
            self._stop_blink()
            if pattern == "blink":
                self._start_blink(color, float(duration_s), blink_interval_s)
            else:
                self._set_color_now(color)

    def _start_blink(self, color, duration_s, blink_interval_s=None):
        stop_at = time.monotonic() + duration_s if duration_s > 0 else None
        interval = LED_BLINK_INTERVAL_SECONDS if blink_interval_s is None else blink_interval_s
        interval = max(0.05, float(interval))

        def blink_loop():
            on = False
            while not self.blink_stop.is_set():
                if stop_at is not None and time.monotonic() >= stop_at:
                    break
                on = not on
                self._set_color_now(color if on else "off")
                time.sleep(interval)
            self._set_color_now("off")

        self.blink_thread = threading.Thread(target=blink_loop, daemon=True)
        self.blink_thread.start()

    def close(self):
        with self.lock:
            self._stop_blink()
            self._set_color_now("off")
            if self.lines is not None:
                try:
                    self.lines.release()
                except Exception:
                    pass
                self.lines = None
            self.enabled = False
