import time

from DC_sender import send_frame_to_discord
from event_logger import log_event
from led_control import RgbLedController


class DummyRigApi:
    def __init__(self):
        self.capture_requested = False
        self.manual_mode_requested = False
        self.auto_mode_requested = False
        self.led = RgbLedController()

    def request_capture(self):
        self.capture_requested = True

    def consume_capture_request(self, context):
        requested = self.capture_requested or bool(context.get("api_capture_requested", False))
        self.capture_requested = False
        return requested

    def request_manual_mode(self):
        self.manual_mode_requested = True

    def consume_manual_mode_request(self, context):
        requested = self.manual_mode_requested or bool(context.get("api_manual_mode", False))
        self.manual_mode_requested = False
        return requested

    def request_auto_mode(self):
        self.auto_mode_requested = True

    def consume_auto_mode_request(self, context):
        requested = self.auto_mode_requested or bool(context.get("api_auto_mode", False))
        self.auto_mode_requested = False
        return requested

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
