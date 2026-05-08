import cv2
import requests
from event_logger import log_event


def send_frame_to_discord(frame, webhook_url):
    if not webhook_url:
        log_event("error", "DISCORD_WEBHOOK_URL not set. Skipping Discord upload.", throttle_seconds=0.0)
        return False

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        log_event("error", "Failed to encode frame for Discord.", throttle_seconds=0.0)
        return False

    try:
        response = requests.post(
            webhook_url,
            data={"content": "MakeNTU capture uploaded."},
            files={"file": ("capture.jpg", buffer.tobytes())},
            timeout=5,
        )

        if response.status_code in (200, 204):
            log_event("api", "Discord upload succeeded.", throttle_seconds=0.0)
            return True

        log_event("error", f"Discord upload failed: {response.status_code}", throttle_seconds=0.0)
        return False

    except Exception as exc:
        log_event("error", f"Discord upload error: {exc}", throttle_seconds=0.0)
        return False
