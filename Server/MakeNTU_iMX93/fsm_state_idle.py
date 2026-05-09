import time

from config import (
    MANUAL_GESTURE_COOLDOWN_SECONDS,
    MANUAL_HEIGHT_STEP_CM,
    MANUAL_PAN_STEP_DEGREES,
    MANUAL_START_HEIGHT_CM,
    MANUAL_TILT_STEP_DEGREES,
    MULTI_AUTO_HORIZONTAL_SCAN_DELTA,
    SINGLE_AUTO_HORIZONTAL_SCAN_DELTA,
    STEPPER_HYBRID_BALANCE_STEP_CM,
    STEPPER_PHOTO_COUNT,
    STEPPER_PHOTO_INTERVAL_CM,
)
from event_logger import log_event
from fsm_output import build_motor_command
from fsm_states import (
    STATE_FAILURE,
    STATE_FRAME_BALANCE,
    STATE_HORIZONTAL_SWEEP,
    STATE_MANUAL_CONTROL,
    STATE_MODE_SELECT,
    STATE_MULTI_MODE_AUTO,
    STATE_PHOTO_CAPTURE,
    STATE_SINGLE_MODE_AUTO,
    STATE_STEPPER_POSITION,
    STATE_VERTICAL_SWEEP,
)
from tracking_geometry import clamp_angle


def update_setting(fsm, context):
    if fsm.state_data.get("reset_done", False):
        fsm.switch_state(STATE_MODE_SELECT)
        return fsm.last_command

    try:
        log_event(
            "state",
            "Startup reset begin: centering pan/tilt servos before homing stepper.",
            throttle_seconds=0.0,
        )
        if hasattr(fsm.motor_rig, "center"):
            before_pan = fsm.current_angles.get("pan")
            before_tilt = fsm.current_angles.get("tilt")
            log_event(
                "state",
                (
                    "Startup servo centering requested: "
                    f"before_pan={float(before_pan):.1f}, "
                    f"before_tilt={float(before_tilt):.1f}"
                ),
                throttle_seconds=0.0,
            )
            fsm.motor_rig.center()
            fsm.current_angles.update(fsm.motor_rig.current)
            fsm.default_angles = dict(fsm.motor_rig.current)
            log_event(
                "state",
                (
                    "Startup servo centering complete: "
                    f"center_pan={fsm.current_angles['pan']:.1f}, "
                    f"center_tilt={fsm.current_angles['tilt']:.1f}"
                ),
                throttle_seconds=0.0,
            )
        else:
            log_event(
                "error",
                "Startup servo centering skipped: motor_rig.center is unavailable.",
                throttle_seconds=0.0,
            )

        home_bottom = context.get("stepper_home_bottom")
        if home_bottom is None:
            raise RuntimeError("stepper_home_bottom API is missing from context")
        result = home_bottom()
        fsm.state_data["reset_done"] = True
        fsm.stepper_position_cm = 0.0
        log_event(
            "state",
            (
                "Startup reset complete: servos centered and platform at 0cm "
                f"(pan={fsm.current_angles['pan']:.1f}, "
                f"tilt={fsm.current_angles['tilt']:.1f}, "
                f"home_steps={result.get('home_steps', 0)}, "
                f"backoff_steps={result.get('backoff_steps', 0)})."
            ),
            throttle_seconds=0.0,
        )
    except Exception as exc:
        log_event("error", f"Startup reset failed: {exc}", throttle_seconds=0.0)
        fsm.switch_state(STATE_FAILURE)
        return fsm.last_command

    return build_motor_command(
        fsm.default_angles["pan"],
        fsm.default_angles["tilt"],
        fsm.current_angles["height"],
        "SETTING: startup reset complete",
    )


def update_mode_select(fsm, context):
    hand_sign = context.get("hand_sign")
    if hand_sign is None:
        hand_sign = fsm.api.consume_mode_selection(context)

    if hand_sign in (1, "1"):
        fsm.control_mode = "single"
        fsm.switch_state(STATE_SINGLE_MODE_AUTO)
        return fsm.last_command

    if hand_sign in (2, "2"):
        fsm.control_mode = "multi"
        fsm.switch_state(STATE_MULTI_MODE_AUTO)
        return fsm.last_command

    if hand_sign in (3, "3"):
        fsm.control_mode = "manual"
        fsm.switch_state(STATE_MANUAL_CONTROL)
        return fsm.last_command

    fsm.debug_problems = ["MODE_SELECT: waiting for hand sign 1, 2, or 3"]
    return build_motor_command(
        fsm.default_angles["pan"],
        fsm.default_angles["tilt"],
        fsm.current_angles["height"],
        "MODE_SELECT: waiting for hand sign 1, 2, or 3",
    )


def update_single_mode_auto(fsm, _context):
    _start_auto_sequence(fsm, "single", SINGLE_AUTO_HORIZONTAL_SCAN_DELTA)
    return fsm.last_command


def update_multi_mode_auto(fsm, _context):
    _start_auto_sequence(fsm, "multi", MULTI_AUTO_HORIZONTAL_SCAN_DELTA)
    return fsm.last_command


def update_manual_control(fsm, context):
    if not fsm.state_data.get("positioned", False):
        try:
            move_to_x_cm = context.get("stepper_move_to_x_cm")
            if move_to_x_cm is None:
                raise RuntimeError("stepper_move_to_x_cm API is missing from context")
            result = move_to_x_cm(MANUAL_START_HEIGHT_CM)
            fsm.stepper_position_cm = float(result.get("position_cm", MANUAL_START_HEIGHT_CM))
            fsm.state_data["positioned"] = True
            log_event(
                "state",
                f"Manual control ready at height={fsm.stepper_position_cm:.2f}cm.",
                throttle_seconds=0.0,
            )
        except Exception as exc:
            log_event("error", f"Manual control setup failed: {exc}", throttle_seconds=0.0)
            fsm.switch_state(STATE_FAILURE)
            return fsm.last_command

        return build_motor_command(
            fsm.default_angles["pan"],
            fsm.default_angles["tilt"],
            fsm.current_angles["height"],
            "MANUAL_CONTROL: centered servos and moved to 10cm",
        )

    pan = fsm.state_data.get("target_pan", fsm.current_angles["pan"])
    tilt = fsm.state_data.get("target_tilt", fsm.current_angles["tilt"])
    height = fsm.current_angles["height"]
    gesture = context.get("manual_gesture")
    now = time.monotonic()

    if gesture == "finish":
        log_event("state", "Manual control finished by raised hand; taking photo.", throttle_seconds=0.0)
        fsm.switch_state(STATE_PHOTO_CAPTURE)
        return fsm.last_command

    if gesture and now >= fsm.state_data.get("gesture_ready_at", 0.0):
        pan, tilt = _apply_manual_servo_gesture(pan, tilt, gesture)
        if gesture in ("height_up", "height_down"):
            _apply_manual_height_gesture(fsm, context, gesture)
        fsm.state_data["target_pan"] = pan
        fsm.state_data["target_tilt"] = tilt
        fsm.state_data["gesture_ready_at"] = now + MANUAL_GESTURE_COOLDOWN_SECONDS
        log_event(
            "state",
            (
                f"Manual gesture={gesture}; "
                f"pan={pan:.1f}, tilt={tilt:.1f}, height={fsm.stepper_position_cm:.2f}cm."
            ),
            throttle_seconds=0.0,
        )

    fsm.debug_problems = [f"MANUAL_CONTROL gesture={gesture or 'none'}"]
    return build_motor_command(
        pan,
        tilt,
        height,
        "MANUAL_CONTROL: gesture control active",
    )


def _start_auto_sequence(fsm, mode, horizontal_scan_delta):
    photo_count = int(STEPPER_PHOTO_COUNT)
    fsm.control_mode = mode
    fsm.auto_sequence = {
        "active": True,
        "mode": mode,
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
        "horizontal_scan_delta": float(horizontal_scan_delta),
    }
    log_event(
        "state",
        (
            f"{mode.upper()} auto sequence started: "
            f"{photo_count} photos at {STEPPER_PHOTO_INTERVAL_CM:.1f}cm intervals, "
            f"horizontal_scan_delta={float(horizontal_scan_delta):.1f}deg."
        ),
        throttle_seconds=0.0,
    )
    fsm.switch_state(STATE_STEPPER_POSITION)


def _apply_manual_servo_gesture(pan, tilt, gesture):
    if gesture == "pan_left":
        pan = clamp_angle(float(pan) - float(MANUAL_PAN_STEP_DEGREES))
    elif gesture == "pan_right":
        pan = clamp_angle(float(pan) + float(MANUAL_PAN_STEP_DEGREES))
    elif gesture == "tilt_up":
        tilt = clamp_angle(float(tilt) + float(MANUAL_TILT_STEP_DEGREES))
    elif gesture == "tilt_down":
        tilt = clamp_angle(float(tilt) - float(MANUAL_TILT_STEP_DEGREES))
    return pan, tilt


def _apply_manual_height_gesture(fsm, context, gesture):
    step = float(MANUAL_HEIGHT_STEP_CM)
    delta_cm = step if gesture == "height_up" else -step
    target_cm = max(0.0, float(fsm.stepper_position_cm) + delta_cm)
    move_to_x_cm = context.get("stepper_move_to_x_cm")
    if move_to_x_cm is None:
        log_event("error", "Manual height gesture ignored: stepper_move_to_x_cm API missing.")
        return

    result = move_to_x_cm(target_cm)
    fsm.stepper_position_cm = float(result.get("position_cm", target_cm))


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
