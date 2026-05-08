import time

from config import HORIZONTAL_SCAN_SETTLE_SECONDS, LOG_NOW_ANGLE
from event_logger import log_event
from fsm_output import build_motor_command
from fsm_states import (
    SCAN_PAN_ANGLES,
    SCAN_TILT_ANGLES,
    STATE_FAILURE,
    STATE_HORIZONTAL_FIX,
    STATE_PHOTO_CAPTURE,
    STATE_VERTICAL_FIX,
    STATE_VERTICAL_SWEEP,
)
from tracking_geometry import (
    angles_reached,
    clamp_angle,
    extract_body_targets,
    extract_face_targets,
    get_leftmost_target,
    has_target_on_right_side,
    leftmost_target_has_left_frame,
    register_unique_angles,
    select_centered_body_target,
)


def update_horizontal_sweep(fsm, context):
    target_pan = fsm.state_data["target_pan"]

    if not angles_reached(fsm.current_angles["pan"], target_pan):
        return build_motor_command(
            target_pan,
            fsm.current_angles["tilt"],
            fsm.current_angles["height"],
            f"HORIZONTAL_SWEEP: moving to pan={target_pan:.1f}",
        )

    if not fsm.state_data["sweep_started"]:
        fsm.state_data["sweep_started"] = True
        fsm.state_data["settle_until"] = time.monotonic() + HORIZONTAL_SCAN_SETTLE_SECONDS
        log_event("state", "Horizontal sweep reached 0 degrees. Starting person detection sweep.", throttle_seconds=0.0)
        return build_motor_command(
            target_pan,
            fsm.current_angles["tilt"],
            fsm.current_angles["height"],
            f"HORIZONTAL_SWEEP: settle at pan={target_pan:.1f}",
        )

    if time.monotonic() < fsm.state_data["settle_until"]:
        return build_motor_command(
            target_pan,
            fsm.current_angles["tilt"],
            fsm.current_angles["height"],
            f"HORIZONTAL_SWEEP: waiting at pan={target_pan:.1f}",
        )

    if LOG_NOW_ANGLE:
        log_event(
            "angle",
            f"Now angle pan={fsm.current_angles['pan']:.1f}",
            throttle_key=f"horizontal_now_angle_{fsm.state_data['scan_index']}",
            throttle_seconds=0.0,
        )

    body_targets = extract_body_targets(
        context["indices"],
        context["boxes"],
        context["all_keypoints"],
        context["IMG_SIZE"],
        fsm.current_angles["pan"],
    )

    if body_targets:
        log_event(
            "detect",
            "Horizontal sweep detected a valid person.",
            throttle_key="horizontal_sweep_detect",
        )

    centered_body_target = select_centered_body_target(body_targets, context["IMG_SIZE"])
    candidate_angles = []

    if centered_body_target is not None:
        candidate_angles = [centered_body_target["centered_angle"]]
        log_event(
            "detect",
            (
                "Person center aligned with frame center "
                f"at pan={fsm.current_angles['pan']:.1f}"
            ),
            throttle_key="horizontal_center_hit",
        )

    new_angles = register_unique_angles(
        fsm.state_data["recorded_angles"],
        candidate_angles,
    )

    for angle in new_angles:
        if fsm.state_data["leftmost_angle"] is None:
            fsm.state_data["leftmost_angle"] = angle
            log_event(
                "angle",
                f"Recorded initial leftmost angle={angle:.1f}",
                throttle_seconds=0.0,
            )
        log_event(
            "angle",
            f"Appended horizontal target angle={angle:.1f}",
            throttle_key=f"append_horizontal_{angle:.1f}",
            throttle_seconds=0.0,
        )

    if (
        fsm.state_data["left_exit_seen"] and
        has_target_on_right_side(body_targets, context["IMG_SIZE"])
    ):
        leave_angle = fsm.state_data["leave_angle"]
        leave_angle_text = f"{leave_angle:.1f}" if leave_angle is not None else "None"
        log_event(
            "error",
            (
                "Leftmost person already left frame and a person is still visible on the right side. "
                f"leftmost_angle={fsm.state_data['leftmost_angle']:.1f}, "
                f"leave_angle={leave_angle_text}, "
                f"now_angle={fsm.current_angles['pan']:.1f}"
            ),
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_FAILURE)
        return fsm.last_command

    leftmost_target = get_leftmost_target(body_targets)
    if (
        leftmost_target is not None and
        not fsm.state_data["left_exit_seen"] and
        fsm.state_data["leftmost_angle"] is not None and
        fsm.current_angles["pan"] > fsm.state_data["leftmost_angle"] and
        leftmost_target_has_left_frame(body_targets)
    ):
        fsm.state_data["left_exit_seen"] = True
        fsm.state_data["leave_angle"] = fsm.current_angles["pan"]
        log_event(
            "angle",
            (
                "Leftmost target reached frame edge; "
                f"leftmost_angle={fsm.state_data['leftmost_angle']:.1f}, "
                f"leave_angle={fsm.state_data['leave_angle']:.1f}, "
                f"now_angle={fsm.current_angles['pan']:.1f}"
            ),
            throttle_seconds=0.0,
        )

    if fsm.state_data["scan_index"] >= len(SCAN_PAN_ANGLES) - 1:
        if fsm.state_data["recorded_angles"]:
            fsm.switch_state(STATE_HORIZONTAL_FIX)
        else:
            log_event("error", "Horizontal sweep completed without any targets. Entering FAILURE.")
            fsm.switch_state(STATE_FAILURE)
        return fsm.last_command

    fsm.state_data["scan_index"] += 1
    fsm.state_data["target_pan"] = float(SCAN_PAN_ANGLES[fsm.state_data["scan_index"]])
    fsm.state_data["settle_until"] = time.monotonic() + HORIZONTAL_SCAN_SETTLE_SECONDS
    fsm.debug_problems = [
        f"HORIZONTAL_SWEEP: recorded={len(fsm.state_data['recorded_angles'])}"
    ]
    return build_motor_command(
        fsm.state_data["target_pan"],
        fsm.current_angles["tilt"],
        fsm.current_angles["height"],
        f"HORIZONTAL_SWEEP: scanning pan={fsm.current_angles['pan']:.1f}",
    )


def update_horizontal_fix(fsm, _context):
    target_pan = clamp_angle(fsm.state_data["target_pan"])

    if angles_reached(fsm.current_angles["pan"], target_pan):
        fsm.switch_state(STATE_VERTICAL_SWEEP)
        return fsm.last_command

    return build_motor_command(
        target_pan,
        fsm.current_angles["tilt"],
        fsm.current_angles["height"],
        f"HORIZONTAL_FIX: target pan={target_pan:.1f}",
    )


def update_vertical_sweep(fsm, context):
    target_tilt = fsm.state_data["target_tilt"]

    if not angles_reached(fsm.current_angles["tilt"], target_tilt):
        return build_motor_command(
            fsm.current_angles["pan"],
            target_tilt,
            fsm.current_angles["height"],
            f"VERTICAL_SWEEP: moving to tilt={target_tilt:.1f}",
        )

    face_targets = extract_face_targets(
        context["face_boxes"],
        context["IMG_SIZE"],
        fsm.current_angles["tilt"],
    )

    if face_targets:
        log_event(
            "detect",
            f"Vertical sweep sees {len(face_targets)} face target(s) at tilt={fsm.current_angles['tilt']:.1f}",
            throttle_key="vertical_sweep_detect",
        )

    new_angles = register_unique_angles(
        fsm.state_data["recorded_angles"],
        [target["centered_angle"] for target in face_targets],
    )

    for angle in new_angles:
        log_event(
            "detect",
            f"Appended vertical target angle={angle:.1f}",
            throttle_key=f"append_vertical_{angle:.1f}",
            throttle_seconds=0.0,
        )

    if face_targets:
        fsm.state_data["snapshot_targets"] = face_targets

    if fsm.state_data["scan_index"] >= len(SCAN_TILT_ANGLES) - 1:
        if fsm.state_data["recorded_angles"]:
            fsm.switch_state(STATE_VERTICAL_FIX)
        else:
            log_event("error", "Vertical sweep completed without any face targets. Entering FAILURE.")
            fsm.switch_state(STATE_FAILURE)
        return fsm.last_command

    fsm.state_data["scan_index"] += 1
    fsm.state_data["target_tilt"] = float(SCAN_TILT_ANGLES[fsm.state_data["scan_index"]])
    fsm.debug_problems = [
        f"VERTICAL_SWEEP: recorded={len(fsm.state_data['recorded_angles'])} new={len(new_angles)}"
    ]
    return build_motor_command(
        fsm.current_angles["pan"],
        fsm.state_data["target_tilt"],
        fsm.current_angles["height"],
        f"VERTICAL_SWEEP: scanning tilt={fsm.current_angles['tilt']:.1f}",
    )


def update_vertical_fix(fsm, _context):
    target_tilt = clamp_angle(fsm.state_data["target_tilt"])

    if angles_reached(fsm.current_angles["tilt"], target_tilt):
        fsm.switch_state(STATE_PHOTO_CAPTURE)
        return fsm.last_command

    return build_motor_command(
        fsm.current_angles["pan"],
        target_tilt,
        fsm.current_angles["height"],
        f"VERTICAL_FIX: target tilt={target_tilt:.1f}",
    )
