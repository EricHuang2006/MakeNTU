import time

from DC_sender import send_frame_to_discord
from event_logger import log_event
from led_control import RgbLedController


class DummyRigApi:
    def __init__(self):
        self.mode_selection = None
        self.led = RgbLedController()

    def request_mode_selection(self, mode_number):
        self.mode_selection = mode_number

    def consume_mode_selection(self, context):
        selected = context.get("mode_selection", None)
        if selected is None:
            selected = self.mode_selection
        self.mode_selection = None
        return selected

    def set_light(self, color, pattern="solid", duration_s=0.0, blink_interval_s=None):
        self.led.set_light(
            color,
            pattern=pattern,
            duration_s=duration_s,
            blink_interval_s=blink_interval_s,
        )
        log_event(
            "api",
            (
                f"Light set to color={color}, pattern={pattern}, "
                f"duration={duration_s:.1f}s"
            ),
            throttle_seconds=0.0,
        )

    def notify_capture_trigger(self, reason):
        log_event("api", f"Capture trigger reason={reason}", throttle_seconds=0.0)

    def take_photo(self, frame):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_event("api", f"Take photo timestamp={timestamp}", throttle_seconds=0.0)
        return frame

    def upload_photo(self, frame, webhook_url):
        log_event("api", "Uploading photo to Discord.", throttle_seconds=0.0)
        return send_frame_to_discord(frame, webhook_url)

    def close(self):
        self.led.close()
