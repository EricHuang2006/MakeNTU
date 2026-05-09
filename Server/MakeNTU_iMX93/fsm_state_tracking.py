import time

from config import LOG_NOW_ANGLE, SCAN_SETTLE_SECONDS
from event_logger import log_event
from fsm_output import build_motor_command
from fsm_states import (
    STATE_FAILURE,
    STATE_HORIZONTAL_BALANCE,
    STATE_HORIZONTAL_FIX,
    STATE_PHOTO_CAPTURE,
    STATE_VERTICAL_FIX,
    STATE_VERTICAL_SWEEP,
)


HORIZONTAL_BALANCE_TOLERANCE_PX = 8.0
HORIZONTAL_BALANCE_MAX_ADJUSTMENTS = 8
HORIZONTAL_BALANCE_MAX_STEP_DEGREES = 3.0
HORIZONTAL_BALANCE_MIN_STEP_DEGREES = 0.5
from tracking_geometry import (
    angles_reached,
    clamp_angle,
    compute_top_edge_face_target_tilt,
    extract_body_targets,
    extract_face_targets,
    has_target_in_left_entry_zone,
    get_rightmost_target,
    register_unique_angles,
    rightmost_target_has_right_frame,
    select_centered_body_target,
    select_centered_face_target,
    select_nearest_body_target,
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
            f"Horizontal sweep sees {len(body_targets)} valid body target(s).",
            throttle_key=f"horizontal_sweep_detect_{len(body_targets)}",
        )

    centered_body_target = select_centered_body_target(body_targets, context["IMG_SIZE"])
    nearest_body_target = select_nearest_body_target(body_targets, context["IMG_SIZE"])
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
        fsm.state_data["pending_body_candidate_angle"] = None
        fsm.state_data["pending_body_candidate_offset"] = None
        fsm.state_data["pending_body_candidate_signed_offset"] = None
    elif nearest_body_target is not None:
        previous_angle = fsm.state_data["pending_body_candidate_angle"]
        previous_offset = fsm.state_data["pending_body_candidate_offset"]
        previous_signed_offset = fsm.state_data["pending_body_candidate_signed_offset"]
        current_offset = float(nearest_body_target["offset"])
        current_signed_offset = float(nearest_body_target["signed_offset"])

        if (
            previous_angle is not None and
            previous_offset is not None and
            previous_signed_offset is not None
        ):
            crossed_center = (previous_signed_offset <= 0.0 < current_signed_offset) or (
                previous_signed_offset >= 0.0 > current_signed_offset
            )
            moving_away = current_offset > previous_offset
            close_enough = previous_offset <= max(12.0, context["IMG_SIZE"] * 0.06)

            if close_enough and (crossed_center or moving_away):
                candidate_angles = [float(previous_angle)]
                log_event(
                    "detect",
                    (
                        "Body center crossing inferred near frame center "
                        f"at pan={float(previous_angle):.1f}, "
                        f"best_offset={float(previous_offset):.1f}px"
                    ),
                    throttle_key=f"horizontal_crossing_{float(previous_angle):.1f}",
                    throttle_seconds=0.0,
                )

        fsm.state_data["pending_body_candidate_angle"] = float(fsm.current_angles["pan"])
        fsm.state_data["pending_body_candidate_offset"] = current_offset
        fsm.state_data["pending_body_candidate_signed_offset"] = current_signed_offset
    else:
        fsm.state_data["pending_body_candidate_angle"] = None
        fsm.state_data["pending_body_candidate_offset"] = None
        fsm.state_data["pending_body_candidate_signed_offset"] = None

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

    if fsm.state_data["right_exit_seen"] and not fsm.state_data["left_entry_clear_seen_after_exit"]:
        if not has_target_in_left_entry_zone(body_targets):
            fsm.state_data["left_entry_clear_seen_after_exit"] = True
            log_event(
                "angle",
                (
                    "Left entry zone cleared after rightmost leave; "
                    "future left-edge arrivals will be treated as new entrants."
                ),
                throttle_seconds=0.0,
            )

    if (
        fsm.state_data["right_exit_seen"] and
        fsm.state_data["left_entry_clear_seen_after_exit"] and
        has_target_in_left_entry_zone(body_targets)
    ):
        leave_angle = fsm.state_data["leave_angle"]
        leave_angle_text = f"{leave_angle:.1f}" if leave_angle is not None else "None"
        log_event(
            "error",
            (
                "A new person entered from the left after a previous person already left the frame. "
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
        log_event(
            "state",
            f"Horizontal fix reached target pan angle {target_pan:.1f}. Transitioning to vertical sweep.",
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_VERTICAL_SWEEP)
        return fsm.last_command

    return build_motor_command(
        target_pan,
        fsm.current_angles["tilt"],
        fsm.current_angles["height"],
        f"HORIZONTAL_FIX: target pan={target_pan:.1f}",
    )


def update_horizontal_balance(fsm, context):
    target_pan = clamp_angle(fsm.state_data["target_pan"])

    if not angles_reached(fsm.current_angles["pan"], target_pan):
        log_event(
            "angle",
            (
                "Horizontal balance moving "
                f"current_pan={fsm.current_angles['pan']:.1f}, "
                f"target_pan={target_pan:.1f}"
            ),
            throttle_key=f"horizontal_balance_move_{target_pan:.1f}",
            throttle_seconds=0.0,
        )
        return build_motor_command(
            target_pan,
            fsm.current_angles["tilt"],
            fsm.current_angles["height"],
            f"HORIZONTAL_BALANCE: moving to pan={target_pan:.1f}",
        )

    if time.monotonic() < fsm.state_data["settle_until"]:
        log_event(
            "angle",
            f"Horizontal balance settling at pan={target_pan:.1f}",
            throttle_key=f"horizontal_balance_settle_{fsm.state_data['adjust_count']}",
            throttle_seconds=0.0,
        )
        return build_motor_command(
            target_pan,
            fsm.current_angles["tilt"],
            fsm.current_angles["height"],
            f"HORIZONTAL_BALANCE: waiting at pan={target_pan:.1f}",
        )

    body_targets = extract_body_targets(
        context["indices"],
        context["boxes"],
        context["all_keypoints"],
        context["IMG_SIZE"],
        fsm.current_angles["pan"],
    )

    if not body_targets:
        log_event(
            "error",
            "Horizontal balance could not see any body targets. Entering FAILURE.",
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_FAILURE)
        return fsm.last_command

    left_x = min(target["left_x"] for target in body_targets)
    right_x = max(target["right_x"] for target in body_targets)
    left_margin = float(left_x)
    right_margin = float(context["IMG_SIZE"] - right_x)
    margin_error = left_margin - right_margin
    fsm.state_data["last_margin_error"] = margin_error

    log_event(
        "angle",
        (
            "Horizontal balance margins "
            f"left={left_margin:.1f}px, right={right_margin:.1f}px, "
            f"error={margin_error:.1f}px"
        ),
        throttle_key=f"horizontal_balance_{fsm.state_data['adjust_count']}",
        throttle_seconds=0.0,
    )

    if abs(margin_error) <= HORIZONTAL_BALANCE_TOLERANCE_PX:
        log_event(
            "angle",
            (
                "Horizontal balance complete "
                f"left_margin={left_margin:.1f}px, "
                f"right_margin={right_margin:.1f}px, "
                f"error={margin_error:.1f}px, "
                f"pan={fsm.current_angles['pan']:.1f}"
            ),
            throttle_seconds=0.0,
        )
        fsm.debug_problems = [
            f"HORIZONTAL_BALANCE: balanced error={margin_error:.1f}px"
        ]
        fsm.switch_state(STATE_PHOTO_CAPTURE)
        return fsm.last_command

    if fsm.state_data["adjust_count"] >= HORIZONTAL_BALANCE_MAX_ADJUSTMENTS:
        log_event(
            "error",
            (
                "Horizontal balance reached adjustment limit; continuing to photo capture "
                f"with margin_error={margin_error:.1f}px."
            ),
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_PHOTO_CAPTURE)
        return fsm.last_command

    error_ratio = min(1.0, abs(margin_error) / max(1.0, float(context["IMG_SIZE"])))
    step = max(
        HORIZONTAL_BALANCE_MIN_STEP_DEGREES,
        min(HORIZONTAL_BALANCE_MAX_STEP_DEGREES, error_ratio * 20.0),
    )
    direction = -1.0 if margin_error > 0 else 1.0
    fsm.state_data["target_pan"] = clamp_angle(fsm.current_angles["pan"] + (direction * step))
    fsm.state_data["adjust_count"] += 1
    fsm.state_data["settle_until"] = time.monotonic() + SCAN_SETTLE_SECONDS
    log_event(
        "angle",
        (
            "Horizontal balance adjustment "
            f"#{fsm.state_data['adjust_count']}: "
            f"left_margin={left_margin:.1f}px, "
            f"right_margin={right_margin:.1f}px, "
            f"error={margin_error:.1f}px, "
            f"step={direction * step:.1f}deg, "
            f"target_pan={fsm.state_data['target_pan']:.1f}"
        ),
        throttle_seconds=0.0,
    )
    fsm.debug_problems = [
        (
            "HORIZONTAL_BALANCE: "
            f"left={left_margin:.1f}px right={right_margin:.1f}px "
            f"adjust={direction * step:.1f}deg"
        )
    ]

    return build_motor_command(
        fsm.state_data["target_pan"],
        fsm.current_angles["tilt"],
        fsm.current_angles["height"],
        f"HORIZONTAL_BALANCE: target pan={fsm.state_data['target_pan']:.1f}",
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
        fsm.switch_state(STATE_HORIZONTAL_BALANCE)
        return fsm.last_command

    return build_motor_command(
        fsm.current_angles["pan"],
        target_tilt,
        fsm.current_angles["height"],
        f"VERTICAL_FIX: target tilt={target_tilt:.1f}",
    )
