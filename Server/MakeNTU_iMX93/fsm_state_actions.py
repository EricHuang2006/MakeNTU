import time

from config import LED_PHOTO_SUCCESS_SECONDS
from event_logger import log_event
from fsm_output import build_motor_command
from fsm_states import (
    STATE_FRAME_BALANCE,
    STATE_HORIZONTAL_FIX,
    STATE_HORIZONTAL_SWEEP,
    STATE_MODE_SELECT,
    STATE_STEPPER_POSITION,
    STATE_VERTICAL_FIX,
    STATE_VERTICAL_SWEEP,
)


HORIZONTAL_STATES = {STATE_HORIZONTAL_SWEEP, STATE_HORIZONTAL_FIX}
VERTICAL_STATES = {
    STATE_VERTICAL_SWEEP,
    STATE_VERTICAL_FIX,
    STATE_FRAME_BALANCE,
}


def update_failure(fsm, _context):
    if time.monotonic() >= fsm.state_data["timeout_at"]:
        sequence = getattr(fsm, "auto_sequence", {})
        if sequence.get("active"):
            _recover_auto_sequence_failure(fsm, sequence)
            return fsm.last_command

        log_event("state", "Failure timeout completed. Returning to MODE_SELECT.", throttle_seconds=0.0)
        fsm.switch_state(STATE_MODE_SELECT)
        return fsm.last_command

    return build_motor_command(
        fsm.current_angles["pan"],
        fsm.current_angles["tilt"],
        fsm.current_angles["height"],
        "FAILURE: blinking red light and waiting 5 seconds",
        error="failure_timeout",
    )


def _recover_auto_sequence_failure(fsm, sequence):
    photo_index = int(sequence.get("photo_index", 0))
    _ensure_recovery_data_for_photo(sequence, photo_index)

    failed_adjustment = _adjustment_for_state(fsm.failure_source_state)
    photo_count = int(sequence.get("photo_count", 1))

    if failed_adjustment is None:
        log_event(
            "state",
            (
                "Failure timeout completed from unknown adjustment. "
                f"Skipping auto sequence photo {photo_index + 1}/{photo_count}."
            ),
            throttle_seconds=0.0,
        )
        _skip_current_photo(fsm, sequence)
        return

    retries = sequence["adjustment_retry_used"]
    if not retries.get(failed_adjustment, False):
        retries[failed_adjustment] = True
        retry_state = _start_state_for_adjustment(failed_adjustment)
        log_event(
            "state",
            (
                f"Failure timeout completed. Retrying {failed_adjustment} adjustment "
                f"for auto sequence photo {photo_index + 1}/{photo_count} once."
            ),
            throttle_seconds=0.0,
        )
        fsm.switch_state(retry_state)
        return

    if failed_adjustment == "vertical" and not sequence.get("vertical_backed_to_horizontal", False):
        sequence["vertical_backed_to_horizontal"] = True
        sequence["reset_tilt_before_horizontal"] = True
        retries["horizontal"] = False
        retries["vertical"] = False
        log_event(
            "state",
            (
                "Vertical adjustment failed after retry. Backing up to horizontal "
                f"adjustment for photo {photo_index + 1}/{photo_count} after resetting tilt."
            ),
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_HORIZONTAL_SWEEP)
        return

    log_event(
        "state",
        (
            f"{failed_adjustment.capitalize()} adjustment failed after retry. "
            f"Skipping auto sequence photo {photo_index + 1}/{photo_count}."
        ),
        throttle_seconds=0.0,
    )
    _skip_current_photo(fsm, sequence, failed_adjustment)


def _adjustment_for_state(state):
    if state == STATE_STEPPER_POSITION:
        return "height"
    if state in HORIZONTAL_STATES:
        return "horizontal"
    if state in VERTICAL_STATES:
        return "vertical"
    return None


def _start_state_for_adjustment(adjustment):
    if adjustment == "height":
        return STATE_STEPPER_POSITION
    if adjustment == "horizontal":
        return STATE_HORIZONTAL_SWEEP
    if adjustment == "vertical":
        return STATE_VERTICAL_SWEEP
    return STATE_STEPPER_POSITION


def _skip_current_photo(fsm, sequence, failed_adjustment=None):
    photo_index = int(sequence.get("photo_index", 0))
    photo_count = int(sequence.get("photo_count", 1))
    sequence["photo_index"] = photo_index + 1
    sequence["adjustment_retry_used"] = {}
    sequence["vertical_backed_to_horizontal"] = False
    sequence["last_photo_successful"] = False
    sequence["hybrid_balance_active"] = False
    sequence["hybrid_balance_for_photo"] = False
    sequence["hybrid_balance_failure_count"] = 0
    sequence["hybrid_fallback_requested"] = False
    if failed_adjustment == "horizontal":
        sequence["next_photo_start_horizontal"] = True

    if sequence["photo_index"] < photo_count:
        next_start = "horizontal" if sequence.get("next_photo_start_horizontal", False) else "vertical"
        log_event(
            "state",
            (
                "Proceeding to next height after failed adjustment "
                f"({sequence['photo_index'] + 1}/{photo_count}); "
                f"next adjustment starts at {next_start}."
            ),
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_STEPPER_POSITION)
        return

    sequence["active"] = False
    log_event(
        "state",
        "Final auto sequence height failed. Returning to MODE_SELECT.",
        throttle_seconds=0.0,
    )
    fsm.switch_state(STATE_MODE_SELECT)


def _ensure_recovery_data_for_photo(sequence, index):
    if sequence.get("recovery_photo_index") == index:
        sequence.setdefault("adjustment_retry_used", {})
        sequence["adjustment_retry_used"].setdefault("height", False)
        sequence["adjustment_retry_used"].setdefault("horizontal", False)
        sequence["adjustment_retry_used"].setdefault("vertical", False)
        sequence.setdefault("vertical_backed_to_horizontal", False)
        sequence.setdefault("next_photo_start_horizontal", False)
        sequence.setdefault("reset_tilt_before_horizontal", False)
        sequence.setdefault("last_photo_successful", False)
        sequence.setdefault("hybrid_balance_active", False)
        sequence.setdefault("hybrid_balance_for_photo", False)
        sequence.setdefault("hybrid_balance_failure_count", 0)
        sequence.setdefault("hybrid_fallback_requested", False)
        return

    sequence["recovery_photo_index"] = index
    sequence["adjustment_retry_used"] = {
        "height": False,
        "horizontal": False,
        "vertical": False,
    }
    sequence["vertical_backed_to_horizontal"] = False
    sequence.setdefault("next_photo_start_horizontal", False)
    sequence.setdefault("reset_tilt_before_horizontal", False)
    sequence.setdefault("last_photo_successful", False)
    sequence.setdefault("hybrid_balance_active", False)
    sequence.setdefault("hybrid_balance_for_photo", False)
    sequence.setdefault("hybrid_balance_failure_count", 0)
    sequence.setdefault("hybrid_fallback_requested", False)


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
        fsm.state_data["success_until"] = time.monotonic() + LED_PHOTO_SUCCESS_SECONDS
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

    if time.monotonic() < fsm.state_data.get("success_until", 0.0):
        fsm.debug_problems = ["PHOTO_CAPTURE: green success light before next move"]
        return build_motor_command(
            fsm.current_angles["pan"],
            fsm.current_angles["tilt"],
            fsm.current_angles["height"],
            "PHOTO_CAPTURE: green success light before next move",
        )

    fsm.api.set_light("blue", pattern="solid")
    sequence = getattr(fsm, "auto_sequence", {})
    if sequence.get("active"):
        completed_index = int(sequence.get("photo_index", 0))
        sequence["last_photo_successful"] = True
        sequence["current_stepper_cm"] = completed_index * float(sequence.get("step_cm", 0.0))
        sequence["photo_index"] = completed_index + 1
        sequence["adjustment_retry_used"] = {}
        sequence["vertical_backed_to_horizontal"] = False
        sequence["next_photo_start_horizontal"] = False
        sequence["reset_tilt_before_horizontal"] = False
        sequence["hybrid_balance_active"] = False
        sequence["hybrid_balance_for_photo"] = False
        sequence["hybrid_balance_failure_count"] = 0
        sequence["hybrid_fallback_requested"] = False
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

    fsm.switch_state(STATE_MODE_SELECT)
    return fsm.last_command
