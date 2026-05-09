import time

from config import (
    LOG_AUTO_CONTROL_EVERY_FRAME,
    STEPPER_HYBRID_BALANCE_STEP_CM,
    STEPPER_PHOTO_COUNT,
    STEPPER_PHOTO_INTERVAL_CM,
)
from event_logger import log_event
from fsm_output import build_motor_command
from fsm_states import (
    STATE_FAILURE,
    STATE_AUTO_CONTROL,
    STATE_FRAME_BALANCE,
    STATE_HORIZONTAL_SWEEP,
    STATE_STEPPER_POSITION,
    STATE_VERTICAL_SWEEP,
)


def update_setting(fsm, _context):
    if time.monotonic() - fsm.state_started_at >= 0.1:
        fsm.switch_state(STATE_AUTO_CONTROL)
        return fsm.last_command

    return build_motor_command(
        fsm.default_angles["pan"],
        fsm.default_angles["tilt"],
        fsm.default_angles["height"],
        "SETTING: moving to hardware default",
    )


def update_manual_control(fsm, context):
    target = context.get("manual_target_angles", {})
    pan = target.get("pan", fsm.current_angles["pan"])
    tilt = target.get("tilt", fsm.current_angles["tilt"])
    height = target.get("height", fsm.current_angles["height"])
    fsm.debug_problems = ["MANUAL_CONTROL active"]
    return build_motor_command(
        pan,
        tilt,
        height,
        "MANUAL_CONTROL: hold operator target",
    )


def update_auto_control(fsm, context):
    person_detected = len(context["indices"]) > 0
    hand_raised = bool(context.get("hand_raised", False))
    api_triggered = fsm.api.consume_capture_request(context)
    pose_triggered = hand_raised or person_detected or api_triggered
    skeleton_count = len(context["indices"])
    face_count = len(context.get("face_boxes", []))

    if LOG_AUTO_CONTROL_EVERY_FRAME:
        log_event(
            "detect",
            (
                "AUTO_CONTROL trigger inputs: "
                f"frame={context.get('frame_counter')}, "
                f"skeletons={skeleton_count}, faces={face_count}, "
                f"person_detected={person_detected}, "
                f"hand_raised={hand_raised}, api_triggered={api_triggered}"
            ),
            throttle_key="auto_control_trigger_inputs",
            throttle_seconds=0.0,
        )

    if pose_triggered:
        photo_count = int(STEPPER_PHOTO_COUNT)
        fsm.auto_sequence = {
            "active": True,
            "photo_index": 0,
            "photo_count": photo_count,
            "step_cm": float(STEPPER_PHOTO_INTERVAL_CM),
            "hybrid_step_cm": float(STEPPER_HYBRID_BALANCE_STEP_CM),
            "current_stepper_cm": 0.0,
            "last_photo_successful": False,
            "hybrid_balance_active": False,
            "hybrid_balance_for_photo": False,
            "hybrid_balance_failure_count": 0,
            "hybrid_fallback_requested": False,
            "height_positioned_index": None,
            "recovery_photo_index": None,
            "adjustment_retry_used": {},
            "vertical_backed_to_horizontal": False,
            "next_photo_start_horizontal": False,
            "reset_tilt_before_horizontal": False,
        }
        log_event(
            "state",
            (
                "AUTO_CONTROL capture sequence started: "
                f"{photo_count} photos at {STEPPER_PHOTO_INTERVAL_CM:.1f}cm intervals "
                f"(person_detected={person_detected}, hand_raised={hand_raised}, "
                f"api_triggered={api_triggered})."
            ),
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_STEPPER_POSITION)
        return fsm.last_command

    fsm.debug_problems = ["AUTO_CONTROL: waiting for person, raised hand, or API trigger"]
    return build_motor_command(
        fsm.default_angles["pan"],
        fsm.default_angles["tilt"],
        fsm.current_angles["height"],
        "AUTO_CONTROL: centered pan/tilt, waiting for person, raised hand, or API trigger",
    )


def update_stepper_position(fsm, context):
    if fsm.state_data["position_started"]:
        next_state = _first_adjustment_state(fsm.auto_sequence)
        fsm.switch_state(next_state)
        return fsm.last_command

    index = int(fsm.auto_sequence["photo_index"])
    total = int(fsm.auto_sequence["photo_count"])
    step_cm = float(fsm.auto_sequence["step_cm"])
    target_cm = index * step_cm
    fsm.state_data["target_cm"] = target_cm
    _ensure_recovery_data_for_photo(fsm.auto_sequence, index)

    if fsm.auto_sequence.get("height_positioned_index") == index:
        fsm.auto_sequence["hybrid_fallback_requested"] = False
        log_event(
            "motor",
            (
                f"Stepper already positioned for photo {index + 1}/{total}; "
                "redoing downstream adjustment without moving z axis."
            ),
            throttle_seconds=0.0,
        )
        fsm.state_data["position_started"] = True
        next_state = _first_adjustment_state(fsm.auto_sequence)
        fsm.switch_state(next_state)
        return fsm.last_command

    try:
        if index == 0:
            home_bottom = context.get("stepper_home_bottom")
            if home_bottom is None:
                raise RuntimeError("stepper_home_bottom API is missing from context")
            result = home_bottom()
            log_event(
                "motor",
                (
                    "Stepper positioned for photo 1/"
                    f"{total}: homed bottom, home_steps={result.get('home_steps', 0)}, "
                    f"backoff_steps={result.get('backoff_steps', 0)}."
                ),
                throttle_seconds=0.0,
            )
            fsm.auto_sequence["current_stepper_cm"] = 0.0
            fsm.auto_sequence["hybrid_balance_active"] = False
            fsm.auto_sequence["hybrid_balance_for_photo"] = False
            fsm.auto_sequence["hybrid_balance_failure_count"] = 0
            fsm.auto_sequence["hybrid_fallback_requested"] = False
        else:
            if _should_use_hybrid_step(fsm.auto_sequence):
                result = _move_hybrid_step(fsm, context, target_cm)
                fsm.state_data["position_started"] = True
                fsm.switch_state(STATE_FRAME_BALANCE)
                return fsm.last_command

            current_cm = float(fsm.auto_sequence.get("current_stepper_cm", (index - 1) * step_cm))
            result = _move_stepper_to_target(fsm, context, target_cm, target_cm - current_cm)
            fsm.auto_sequence["hybrid_balance_active"] = False
            fsm.auto_sequence["hybrid_balance_for_photo"] = False
            fsm.auto_sequence["hybrid_balance_failure_count"] = 0
            fsm.auto_sequence["hybrid_fallback_requested"] = False
            log_event(
                "motor",
                (
                    f"Stepper positioned for photo {index + 1}/{total}: "
                    f"target={target_cm:.2f}cm, position={result.get('position_cm', target_cm):.2f}cm, "
                    f"steps={result.get('steps', 0)}."
                ),
                throttle_seconds=0.0,
            )
    except Exception as exc:
        log_event("error", f"Stepper positioning failed: {exc}", throttle_seconds=0.0)
        fsm.switch_state(STATE_FAILURE)
        return fsm.last_command

    fsm.auto_sequence["height_positioned_index"] = index
    fsm.state_data["position_started"] = True
    next_state = _first_adjustment_state(fsm.auto_sequence)
    fsm.switch_state(next_state)
    return fsm.last_command


def _should_use_hybrid_step(sequence):
    return (
        sequence.get("last_photo_successful", False) and
        not sequence.get("hybrid_fallback_requested", False)
    )


def _move_hybrid_step(fsm, context, target_cm):
    sequence = fsm.auto_sequence
    current_cm = float(sequence.get("current_stepper_cm", (int(sequence["photo_index"]) - 1) * float(sequence["step_cm"])))
    hybrid_step_cm = float(sequence.get("hybrid_step_cm", STEPPER_HYBRID_BALANCE_STEP_CM))
    next_cm = min(float(target_cm), current_cm + hybrid_step_cm)
    result = _move_stepper_to_target(fsm, context, next_cm, next_cm - current_cm)
    position_cm = float(result.get("position_cm", next_cm))

    reached_photo_point = abs(position_cm - float(target_cm)) <= 0.05
    sequence["current_stepper_cm"] = position_cm
    sequence["hybrid_balance_active"] = True
    sequence["hybrid_balance_for_photo"] = reached_photo_point
    sequence["hybrid_balance_failure_count"] = 0
    sequence["hybrid_fallback_requested"] = False
    if reached_photo_point:
        sequence["height_positioned_index"] = int(sequence["photo_index"])

    log_event(
        "motor",
        (
            "Hybrid stepper move "
            f"toward photo {int(sequence['photo_index']) + 1}/{int(sequence['photo_count'])}: "
            f"current={current_cm:.2f}cm, next={position_cm:.2f}cm, "
            f"target={float(target_cm):.2f}cm, "
            f"for_photo={reached_photo_point}, steps={result.get('steps', 0)}."
        ),
        throttle_seconds=0.0,
    )
    return result


def _move_stepper_to_target(fsm, context, target_cm, fallback_delta_cm):
    move_to_x_cm = context.get("stepper_move_to_x_cm")
    if move_to_x_cm is not None:
        result = move_to_x_cm(target_cm)
    else:
        adjust_x_cm = context.get("adjust_x_cm")
        if adjust_x_cm is None:
            raise RuntimeError("stepper movement API is missing from context")
        result = adjust_x_cm(fallback_delta_cm)

    fsm.auto_sequence["current_stepper_cm"] = float(result.get("position_cm", target_cm))
    return result


def _ensure_recovery_data_for_photo(sequence, index):
    if sequence.get("recovery_photo_index") == index:
        return

    sequence["recovery_photo_index"] = index
    sequence["adjustment_retry_used"] = {
        "height": False,
        "horizontal": False,
        "vertical": False,
    }
    sequence["vertical_backed_to_horizontal"] = False


def _first_adjustment_state(sequence):
    index = int(sequence.get("photo_index", 0))
    if index == 0:
        sequence["next_photo_start_horizontal"] = False
        return STATE_HORIZONTAL_SWEEP

    if sequence.get("next_photo_start_horizontal", False):
        sequence["next_photo_start_horizontal"] = False
        return STATE_HORIZONTAL_SWEEP

    return STATE_VERTICAL_SWEEP
