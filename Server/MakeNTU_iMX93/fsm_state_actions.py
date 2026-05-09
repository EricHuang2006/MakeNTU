import time

from event_logger import log_event
from fsm_output import build_motor_command
from fsm_states import STATE_AUTO_CONTROL, STATE_STEPPER_POSITION


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
        sequence = getattr(fsm, "auto_sequence", {})
        if sequence.get("active"):
            photo_index = int(sequence.get("photo_index", 0)) + 1
            photo_count = int(sequence.get("photo_count", 1))
            log_event(
                "api",
                f"Taking auto sequence photo {photo_index}/{photo_count}.",
                throttle_seconds=0.0,
            )
        else:
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

    sequence = getattr(fsm, "auto_sequence", {})
    if sequence.get("active"):
        sequence["photo_index"] = int(sequence.get("photo_index", 0)) + 1
        if sequence["photo_index"] < int(sequence.get("photo_count", 1)):
            log_event(
                "state",
                (
                    "Photo captured; moving stepper to next auto sequence position "
                    f"({sequence['photo_index'] + 1}/{sequence.get('photo_count', 1)})."
                ),
                throttle_seconds=0.0,
            )
            fsm.switch_state(STATE_STEPPER_POSITION)
            return fsm.last_command

        sequence["active"] = False
        log_event("state", "Auto capture sequence completed.", throttle_seconds=0.0)

    fsm.switch_state(STATE_AUTO_CONTROL)
    return fsm.last_command
