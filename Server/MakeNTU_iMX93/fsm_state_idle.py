import time

from config import STEPPER_PHOTO_COUNT, STEPPER_ROD_LENGTH_CM
from event_logger import log_event
from fsm_output import build_motor_command
from fsm_states import (
    STATE_AUTO_CONTROL,
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
    pose_triggered = len(context["indices"]) > 0 or fsm.api.consume_capture_request(context)

    if pose_triggered:
        photo_count = int(STEPPER_PHOTO_COUNT)
        fsm.auto_sequence = {
            "active": True,
            "photo_index": 0,
            "photo_count": photo_count,
            "step_cm": float(STEPPER_ROD_LENGTH_CM) / max(1, photo_count),
        }
        log_event(
            "state",
            (
                "AUTO_CONTROL capture sequence started: "
                f"{photo_count} photos across {STEPPER_ROD_LENGTH_CM:.1f}cm."
            ),
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_STEPPER_POSITION)
        return fsm.last_command

    fsm.debug_problems = ["AUTO_CONTROL: waiting for skeleton or API trigger"]
    return build_motor_command(
        fsm.current_angles["pan"],
        fsm.current_angles["tilt"],
        fsm.current_angles["height"],
        "AUTO_CONTROL: full body mode standby",
    )


def update_stepper_position(fsm, context):
    if fsm.state_data["position_started"]:
        next_state = STATE_HORIZONTAL_SWEEP if fsm.auto_sequence["photo_index"] == 0 else STATE_VERTICAL_SWEEP
        fsm.switch_state(next_state)
        return fsm.last_command

    index = int(fsm.auto_sequence["photo_index"])
    total = int(fsm.auto_sequence["photo_count"])
    step_cm = float(fsm.auto_sequence["step_cm"])
    target_cm = index * step_cm
    fsm.state_data["target_cm"] = target_cm

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
                    f"{total}: homed bottom, steps={result.get('steps', 0)}."
                ),
                throttle_seconds=0.0,
            )
        else:
            move_to_x_cm = context.get("stepper_move_to_x_cm")
            if move_to_x_cm is None:
                adjust_x_cm = context.get("adjust_x_cm")
                if adjust_x_cm is None:
                    raise RuntimeError("stepper movement API is missing from context")
                result = adjust_x_cm(step_cm)
            else:
                result = move_to_x_cm(target_cm)
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
        raise

    fsm.state_data["position_started"] = True
    next_state = STATE_HORIZONTAL_SWEEP if index == 0 else STATE_VERTICAL_SWEEP
    fsm.switch_state(next_state)
    return fsm.last_command
