import time

from event_logger import log_event
from fsm_output import build_motor_command
from fsm_states import STATE_AUTO_CONTROL


def update_failure(fsm, _context):
    if time.monotonic() >= fsm.state_data["timeout_at"]:
        log_event("state", "Failure timeout completed. Returning to AUTO_CONTROL.", throttle_seconds=0.0)
        fsm.switch_state(STATE_AUTO_CONTROL)
        return fsm.last_command

    return build_motor_command(
        fsm.current_angles["pan"],
        fsm.current_angles["tilt"],
        fsm.current_angles["height"],
        "FAILURE: blinking red light and waiting 5 seconds",
        error="failure_timeout",
    )


def update_photo_capture(fsm, context):
    if time.monotonic() < fsm.state_data.get("flash_until", 0.0):
        fsm.debug_problems = ["PHOTO_CAPTURE: blinking light countdown before capture"]
        return build_motor_command(
            fsm.current_angles["pan"],
            fsm.current_angles["tilt"],
            fsm.current_angles["height"],
            "PHOTO_CAPTURE: blinking light countdown before capture",
        )

    if not fsm.state_data.get("captured", False):
        fsm.api.set_light("green", pattern="solid")
        log_event("api", "Taking photo.", throttle_seconds=0.0)
        photo = fsm.api.take_photo(context["frame"])
        uploaded = fsm.api.upload_photo(photo, context.get("DISCORD_WEBHOOK_URL"))
        fsm.state_data["captured"] = True
        fsm.debug_problems = ["PHOTO_CAPTURE: captured frame and upload requested"]
        if uploaded:
            log_event("api", "Photo upload succeeded.", throttle_seconds=0.0)
        else:
            log_event("error", "Photo upload failed or skipped.", throttle_seconds=0.0)
        return build_motor_command(
            fsm.current_angles["pan"],
            fsm.current_angles["tilt"],
            fsm.current_angles["height"],
            "PHOTO_CAPTURE: green light and upload photo",
        )

    fsm.switch_state(STATE_AUTO_CONTROL)
    return fsm.last_command
