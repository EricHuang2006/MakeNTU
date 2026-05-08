import time

from config import LOG_NOW_ANGLE, SCAN_SETTLE_SECONDS
from event_logger import log_event
from fsm_output import build_motor_command
from fsm_states import (
    STATE_FAILURE,
    STATE_HORIZONTAL_FIX,
    STATE_PHOTO_CAPTURE,
    STATE_VERTICAL_FIX,
    STATE_VERTICAL_SWEEP,
)
from tracking_geometry import (
    angles_reached,
    clamp_angle,
    compute_top_edge_face_target_tilt,
    extract_body_targets,
    extract_face_targets,
    get_rightmost_target,
    has_target_on_left_side,
    register_unique_angles,
    rightmost_target_has_right_frame,
    select_centered_body_target,
    select_centered_face_target,
    select_top_edge_face_target,
)


def update_horizontal_sweep(fsm, context):
    scan_angles = fsm.state_data["scan_angles"]
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
        fsm.state_data["settle_until"] = time.monotonic() + SCAN_SETTLE_SECONDS
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
        if fsm.state_data["rightmost_angle"] is None:
            fsm.state_data["rightmost_angle"] = angle
            log_event(
                "angle",
                f"Recorded initial rightmost angle={angle:.1f}",
                throttle_seconds=0.0,
            )
        log_event(
            "angle",
            f"Appended horizontal target angle={angle:.1f}",
            throttle_key=f"append_horizontal_{angle:.1f}",
            throttle_seconds=0.0,
        )

    if (
        fsm.state_data["right_exit_seen"] and
        has_target_on_left_side(body_targets, context["IMG_SIZE"])
    ):
        leave_angle = fsm.state_data["leave_angle"]
        leave_angle_text = f"{leave_angle:.1f}" if leave_angle is not None else "None"
        log_event(
            "error",
            (
                "Rightmost person already left frame and a person is still visible on the left side. "
                f"rightmost_angle={fsm.state_data['rightmost_angle']:.1f}, "
                f"leave_angle={leave_angle_text}, "
                f"now_angle={fsm.current_angles['pan']:.1f}"
            ),
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_FAILURE)
        return fsm.last_command

    rightmost_target = get_rightmost_target(body_targets)
    if (
        rightmost_target is not None and
        not fsm.state_data["right_exit_seen"] and
        fsm.state_data["rightmost_angle"] is not None and
        fsm.current_angles["pan"] > fsm.state_data["rightmost_angle"] and
        rightmost_target_has_right_frame(body_targets, context["IMG_SIZE"])
    ):
        fsm.state_data["right_exit_seen"] = True
        fsm.state_data["leave_angle"] = fsm.current_angles["pan"]
        log_event(
            "angle",
            (
                "Rightmost target reached frame edge; "
                f"rightmost_angle={fsm.state_data['rightmost_angle']:.1f}, "
                f"leave_angle={fsm.state_data['leave_angle']:.1f}, "
                f"now_angle={fsm.current_angles['pan']:.1f}"
            ),
            throttle_seconds=0.0,
        )

    if fsm.state_data["scan_index"] >= len(scan_angles) - 1:
        if fsm.state_data["recorded_angles"]:
            fsm.switch_state(STATE_HORIZONTAL_FIX)
        else:
            log_event("error", "Horizontal sweep completed without any targets. Entering FAILURE.")
            fsm.switch_state(STATE_FAILURE)
        return fsm.last_command

    fsm.state_data["scan_index"] += 1
    fsm.state_data["target_pan"] = float(scan_angles[fsm.state_data["scan_index"]])
    fsm.state_data["settle_until"] = 0.0
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
    scan_angles = fsm.state_data["scan_angles"]
    target_tilt = fsm.state_data["target_tilt"]

    if not angles_reached(fsm.current_angles["tilt"], target_tilt):
        return build_motor_command(
            fsm.current_angles["pan"],
            target_tilt,
            fsm.current_angles["height"],
            f"VERTICAL_SWEEP: moving to tilt={target_tilt:.1f}",
        )

    if not fsm.state_data["sweep_started"]:
        fsm.state_data["sweep_started"] = True
        fsm.state_data["settle_until"] = time.monotonic() + SCAN_SETTLE_SECONDS
        log_event("state", "Vertical sweep reached start angle. Starting face detection sweep.", throttle_seconds=0.0)
        return build_motor_command(
            fsm.current_angles["pan"],
            target_tilt,
            fsm.current_angles["height"],
            f"VERTICAL_SWEEP: settle at tilt={target_tilt:.1f}",
        )

    if time.monotonic() < fsm.state_data["settle_until"]:
        return build_motor_command(
            fsm.current_angles["pan"],
            target_tilt,
            fsm.current_angles["height"],
            f"VERTICAL_SWEEP: waiting at tilt={target_tilt:.1f}",
        )

    if LOG_NOW_ANGLE:
        log_event(
            "angle",
            f"Now angle tilt={fsm.current_angles['tilt']:.1f}",
            throttle_key=f"vertical_now_angle_{fsm.state_data['scan_index']}",
            throttle_seconds=0.0,
        )

    face_targets = extract_face_targets(
        context["face_boxes"],
        context["IMG_SIZE"],
        fsm.current_angles["tilt"],
    )

    if face_targets:
        log_event(
            "detect",
            "Vertical sweep detected a valid face.",
            throttle_key="vertical_sweep_detect",
        )

    candidate_angles = []
    top_edge_face_target = select_top_edge_face_target(face_targets, context["IMG_SIZE"])

    if top_edge_face_target is not None and not fsm.state_data["top_edge_target_recorded"]:
        computed_tilt = compute_top_edge_face_target_tilt(
            fsm.current_angles["tilt"],
            top_edge_face_target,
            context["IMG_SIZE"],
        )
        candidate_angles = [computed_tilt]
        fsm.state_data["top_edge_target_recorded"] = True
        log_event(
            "detect",
            (
                "Face reached frame top edge "
                f"at tilt={fsm.current_angles['tilt']:.1f}, "
                f"computed target tilt={computed_tilt:.1f}"
            ),
            throttle_key="vertical_top_edge_hit",
        )
    else:
        centered_face_target = select_centered_face_target(face_targets, context["IMG_SIZE"])
        if centered_face_target is not None:
            log_event(
                "detect",
                (
                    "Face center aligned with frame center "
                    f"at tilt={fsm.current_angles['tilt']:.1f}"
                ),
                throttle_key="vertical_center_hit",
            )

    new_angles = register_unique_angles(fsm.state_data["recorded_angles"], candidate_angles)

    for angle in new_angles:
        log_event(
            "angle",
            f"Appended vertical target angle={angle:.1f}",
            throttle_key=f"append_vertical_{angle:.1f}",
            throttle_seconds=0.0,
        )

    if top_edge_face_target is not None:
        fsm.state_data["snapshot_targets"] = [top_edge_face_target]

    if fsm.state_data["scan_index"] >= len(scan_angles) - 1:
        if fsm.state_data["recorded_angles"]:
            fsm.switch_state(STATE_VERTICAL_FIX)
        else:
            log_event("error", "Vertical sweep completed without any face targets. Entering FAILURE.")
            fsm.switch_state(STATE_FAILURE)
        return fsm.last_command

    fsm.state_data["scan_index"] += 1
    fsm.state_data["target_tilt"] = float(scan_angles[fsm.state_data["scan_index"]])
    fsm.state_data["settle_until"] = 0.0
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
