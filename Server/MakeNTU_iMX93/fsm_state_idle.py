import time

from fsm_output import build_motor_command
from fsm_states import STATE_AUTO_CONTROL, STATE_HORIZONTAL_SWEEP


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
        fsm.switch_state(STATE_HORIZONTAL_SWEEP)
        return fsm.last_command

    fsm.debug_problems = ["AUTO_CONTROL: waiting for skeleton or API trigger"]
    return build_motor_command(
        fsm.current_angles["pan"],
        fsm.current_angles["tilt"],
        fsm.current_angles["height"],
        "AUTO_CONTROL: full body mode standby",
    )
