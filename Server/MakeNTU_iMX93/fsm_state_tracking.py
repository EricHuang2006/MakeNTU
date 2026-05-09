import time

from config import (
    LOG_NOW_ANGLE,
    SCAN_SETTLE_SECONDS,
    STEPPER_HYBRID_BALANCE_MAX_CONSECUTIVE_FAILURES,
)
from event_logger import log_event
from fsm_output import build_motor_command
from fsm_states import (
    STATE_FAILURE,
    STATE_FRAME_BALANCE,
    STATE_HORIZONTAL_BALANCE,
    STATE_HORIZONTAL_FIX,
    STATE_PHOTO_CAPTURE,
    STATE_STEPPER_POSITION,
    STATE_VERTICAL_BALANCE,
    STATE_VERTICAL_FIX,
    STATE_VERTICAL_SWEEP,
)


HORIZONTAL_BALANCE_TOLERANCE_PX = 8.0
HORIZONTAL_BALANCE_MAX_ADJUSTMENTS = 8
HORIZONTAL_BALANCE_MAX_STEP_DEGREES = 3.0
HORIZONTAL_BALANCE_MIN_STEP_DEGREES = 0.5
VERTICAL_BALANCE_TARGET_RATIO = 1.0 / 3.0
VERTICAL_BALANCE_TOLERANCE_PX = 8.0
VERTICAL_BALANCE_MAX_ADJUSTMENTS = 8
VERTICAL_BALANCE_MAX_STEP_DEGREES = 3.0
VERTICAL_BALANCE_MIN_STEP_DEGREES = 0.5
FRAME_BALANCE_MAX_ADJUSTMENTS = 16
from tracking_geometry import (
    angles_reached,
    clamp_angle,
    compute_top_edge_face_target_tilt,
    extract_body_targets,
    extract_face_targets,
    filter_right_exit_zone_targets,
    has_centered_target,
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
    target_tilt = fsm.state_data.get("target_tilt", fsm.current_angles["tilt"])

    if (
        not angles_reached(fsm.current_angles["pan"], target_pan) or
        not angles_reached(fsm.current_angles["tilt"], target_tilt)
    ):
        return build_motor_command(
            target_pan,
            target_tilt,
            fsm.current_angles["height"],
            f"HORIZONTAL_SWEEP: moving to pan={target_pan:.1f}, tilt={target_tilt:.1f}",
        )

    if not fsm.state_data["sweep_started"]:
        fsm.state_data["sweep_started"] = True
        fsm.state_data["settle_until"] = time.monotonic() + SCAN_SETTLE_SECONDS
        log_event("state", "Horizontal sweep reached 0 degrees. Starting person detection sweep.", throttle_seconds=0.0)
        return build_motor_command(
            target_pan,
            target_tilt,
            fsm.current_angles["height"],
            f"HORIZONTAL_SWEEP: settle at pan={target_pan:.1f}, tilt={target_tilt:.1f}",
        )

    if time.monotonic() < fsm.state_data["settle_until"]:
        return build_motor_command(
            target_pan,
            target_tilt,
            fsm.current_angles["height"],
            f"HORIZONTAL_SWEEP: waiting at pan={target_pan:.1f}, tilt={target_tilt:.1f}",
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

    if not fsm.state_data["initial_exit_side_ignored"]:
        original_count = len(body_targets)
        body_targets = filter_right_exit_zone_targets(body_targets, context["IMG_SIZE"])
        ignored_count = original_count - len(body_targets)
        fsm.state_data["initial_exit_side_ignored"] = True
        if ignored_count > 0:
            log_event(
                "detect",
                (
                    "Horizontal sweep ignored "
                    f"{ignored_count} target(s) already on the right exit side "
                    "in the first detection frame."
                ),
                throttle_seconds=0.0,
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
        has_centered_target(body_targets, context["IMG_SIZE"])
    ):
        leave_angle = fsm.state_data["leave_angle"]
        leave_angle_text = f"{leave_angle:.1f}" if leave_angle is not None else "None"
        log_event(
            "error",
            (
                "A new person entered from the left and reached center after a previous person left the frame. "
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
        target_tilt,
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
        if _fallback_from_failed_hybrid_balance(fsm, "no body targets"):
            return fsm.last_command
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
        fsm.switch_state(STATE_FRAME_BALANCE)
        return fsm.last_command

    return build_motor_command(
        fsm.current_angles["pan"],
        target_tilt,
        fsm.current_angles["height"],
        f"VERTICAL_FIX: target tilt={target_tilt:.1f}",
    )


def update_vertical_balance(fsm, context):
    target_tilt = clamp_angle(fsm.state_data["target_tilt"])

    if not angles_reached(fsm.current_angles["tilt"], target_tilt):
        log_event(
            "angle",
            (
                "Vertical balance moving "
                f"current_tilt={fsm.current_angles['tilt']:.1f}, "
                f"target_tilt={target_tilt:.1f}"
            ),
            throttle_key=f"vertical_balance_move_{target_tilt:.1f}",
            throttle_seconds=0.0,
        )
        return build_motor_command(
            fsm.current_angles["pan"],
            target_tilt,
            fsm.current_angles["height"],
            f"VERTICAL_BALANCE: moving to tilt={target_tilt:.1f}",
        )

    if time.monotonic() < fsm.state_data["settle_until"]:
        log_event(
            "angle",
            f"Vertical balance settling at tilt={target_tilt:.1f}",
            throttle_key=f"vertical_balance_settle_{fsm.state_data['adjust_count']}",
            throttle_seconds=0.0,
        )
        return build_motor_command(
            fsm.current_angles["pan"],
            target_tilt,
            fsm.current_angles["height"],
            f"VERTICAL_BALANCE: waiting at tilt={target_tilt:.1f}",
        )

    face_targets = extract_face_targets(
        context["face_boxes"],
        context["IMG_SIZE"],
        fsm.current_angles["tilt"],
    )

    if not face_targets:
        log_event(
            "error",
            "Vertical balance could not see any face targets. Entering FAILURE.",
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_FAILURE)
        return fsm.last_command

    avg_center_y = sum(target["center_y"] for target in face_targets) / len(face_targets)
    target_y = float(context["IMG_SIZE"]) * VERTICAL_BALANCE_TARGET_RATIO
    face_error = avg_center_y - target_y
    fsm.state_data["last_face_error"] = face_error

    log_event(
        "angle",
        (
            "Vertical balance face centers "
            f"count={len(face_targets)}, avg_y={avg_center_y:.1f}px, "
            f"target_y={target_y:.1f}px, error={face_error:.1f}px"
        ),
        throttle_key=f"vertical_balance_{fsm.state_data['adjust_count']}",
        throttle_seconds=0.0,
    )

    if abs(face_error) <= VERTICAL_BALANCE_TOLERANCE_PX:
        log_event(
            "angle",
            (
                "Vertical balance complete "
                f"avg_y={avg_center_y:.1f}px, target_y={target_y:.1f}px, "
                f"error={face_error:.1f}px, tilt={fsm.current_angles['tilt']:.1f}"
            ),
            throttle_seconds=0.0,
        )
        fsm.debug_problems = [
            f"VERTICAL_BALANCE: balanced error={face_error:.1f}px"
        ]
        fsm.switch_state(STATE_HORIZONTAL_BALANCE)
        return fsm.last_command

    if fsm.state_data["adjust_count"] >= VERTICAL_BALANCE_MAX_ADJUSTMENTS:
        log_event(
            "error",
            (
                "Vertical balance reached adjustment limit; continuing to horizontal balance "
                f"with face_error={face_error:.1f}px."
            ),
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_HORIZONTAL_BALANCE)
        return fsm.last_command

    error_ratio = min(1.0, abs(face_error) / max(1.0, float(context["IMG_SIZE"])))
    step = max(
        VERTICAL_BALANCE_MIN_STEP_DEGREES,
        min(VERTICAL_BALANCE_MAX_STEP_DEGREES, error_ratio * 20.0),
    )
    direction = 1.0 if face_error > 0 else -1.0
    fsm.state_data["target_tilt"] = clamp_angle(fsm.current_angles["tilt"] + (direction * step))
    fsm.state_data["adjust_count"] += 1
    fsm.state_data["settle_until"] = time.monotonic() + SCAN_SETTLE_SECONDS
    log_event(
        "angle",
        (
            "Vertical balance adjustment "
            f"#{fsm.state_data['adjust_count']}: "
            f"avg_y={avg_center_y:.1f}px, target_y={target_y:.1f}px, "
            f"error={face_error:.1f}px, step={direction * step:.1f}deg, "
            f"target_tilt={fsm.state_data['target_tilt']:.1f}"
        ),
        throttle_seconds=0.0,
    )
    fsm.debug_problems = [
        (
            "VERTICAL_BALANCE: "
            f"avg_y={avg_center_y:.1f}px target_y={target_y:.1f}px "
            f"adjust={direction * step:.1f}deg"
        )
    ]

    return build_motor_command(
        fsm.current_angles["pan"],
        fsm.state_data["target_tilt"],
        fsm.current_angles["height"],
        f"VERTICAL_BALANCE: target tilt={fsm.state_data['target_tilt']:.1f}",
    )


def update_frame_balance(fsm, context):
    target_pan = clamp_angle(fsm.state_data["target_pan"])
    target_tilt = clamp_angle(fsm.state_data["target_tilt"])

    if (
        not angles_reached(fsm.current_angles["pan"], target_pan) or
        not angles_reached(fsm.current_angles["tilt"], target_tilt)
    ):
        return build_motor_command(
            target_pan,
            target_tilt,
            fsm.current_angles["height"],
            f"FRAME_BALANCE: moving to pan={target_pan:.1f}, tilt={target_tilt:.1f}",
        )

    if time.monotonic() < fsm.state_data["settle_until"]:
        return build_motor_command(
            target_pan,
            target_tilt,
            fsm.current_angles["height"],
            f"FRAME_BALANCE: settling pan={target_pan:.1f}, tilt={target_tilt:.1f}",
        )

    body_targets = extract_body_targets(
        context["indices"],
        context["boxes"],
        context["all_keypoints"],
        context["IMG_SIZE"],
        fsm.current_angles["pan"],
    )
    face_targets = extract_face_targets(
        context["face_boxes"],
        context["IMG_SIZE"],
        fsm.current_angles["tilt"],
    )

    if not body_targets:
        if _fallback_from_failed_hybrid_balance(fsm, "no body targets"):
            return fsm.last_command
        log_event(
            "error",
            "Frame balance could not see any body targets. Entering FAILURE.",
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_FAILURE)
        return fsm.last_command

    if not face_targets:
        if _fallback_from_failed_hybrid_balance(fsm, "no face targets"):
            return fsm.last_command
        log_event(
            "error",
            "Frame balance could not see any face targets. Entering FAILURE.",
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_FAILURE)
        return fsm.last_command

    left_x = min(target["left_x"] for target in body_targets)
    right_x = max(target["right_x"] for target in body_targets)
    left_margin = float(left_x)
    right_margin = float(context["IMG_SIZE"] - right_x)
    margin_error = left_margin - right_margin

    avg_center_y = sum(target["center_y"] for target in face_targets) / len(face_targets)
    target_y = float(context["IMG_SIZE"]) * VERTICAL_BALANCE_TARGET_RATIO
    face_error = avg_center_y - target_y

    fsm.state_data["last_margin_error"] = margin_error
    fsm.state_data["last_face_error"] = face_error

    horizontal_ok = abs(margin_error) <= HORIZONTAL_BALANCE_TOLERANCE_PX
    vertical_ok = abs(face_error) <= VERTICAL_BALANCE_TOLERANCE_PX

    log_event(
        "angle",
        (
            "Frame balance check "
            f"phase={fsm.state_data['phase']}, "
            f"left_margin={left_margin:.1f}px, right_margin={right_margin:.1f}px, "
            f"x_error={margin_error:.1f}px, "
            f"avg_face_y={avg_center_y:.1f}px, target_y={target_y:.1f}px, "
            f"y_error={face_error:.1f}px, "
            f"horizontal_ok={horizontal_ok}, vertical_ok={vertical_ok}"
        ),
        throttle_key=f"frame_balance_{fsm.state_data['adjust_count']}_{fsm.state_data['phase']}",
        throttle_seconds=0.0,
    )

    if horizontal_ok and vertical_ok:
        log_event(
            "angle",
            (
                "Frame balance complete "
                f"x_error={margin_error:.1f}px, y_error={face_error:.1f}px, "
                f"pan={fsm.current_angles['pan']:.1f}, tilt={fsm.current_angles['tilt']:.1f}"
            ),
            throttle_seconds=0.0,
        )
        fsm.debug_problems = [
            f"FRAME_BALANCE: balanced x={margin_error:.1f}px y={face_error:.1f}px"
        ]
        if _complete_successful_hybrid_balance(fsm):
            return fsm.last_command

        fsm.switch_state(STATE_PHOTO_CAPTURE)
        return fsm.last_command

    if fsm.state_data["adjust_count"] >= FRAME_BALANCE_MAX_ADJUSTMENTS:
        if _fallback_from_failed_hybrid_balance(fsm, "adjustment limit"):
            return fsm.last_command
        log_event(
            "error",
            (
                "Frame balance reached adjustment limit before both axes were good. "
                f"x_error={margin_error:.1f}px, y_error={face_error:.1f}px."
            ),
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_FAILURE)
        return fsm.last_command

    phase = fsm.state_data["phase"]
    if phase == "horizontal":
        if horizontal_ok:
            fsm.state_data["phase"] = "vertical"
            fsm.debug_problems = [
                f"FRAME_BALANCE: horizontal ok, checking vertical y={face_error:.1f}px"
            ]
            return build_motor_command(
                fsm.current_angles["pan"],
                fsm.current_angles["tilt"],
                fsm.current_angles["height"],
                "FRAME_BALANCE: horizontal ok, switching to vertical",
            )

        error_ratio = min(1.0, abs(margin_error) / max(1.0, float(context["IMG_SIZE"])))
        step = max(
            HORIZONTAL_BALANCE_MIN_STEP_DEGREES,
            min(HORIZONTAL_BALANCE_MAX_STEP_DEGREES, error_ratio * 20.0),
        )
        direction = -1.0 if margin_error > 0 else 1.0
        fsm.state_data["target_pan"] = clamp_angle(fsm.current_angles["pan"] + (direction * step))
        fsm.state_data["phase"] = "vertical"
        fsm.state_data["adjust_count"] += 1
        fsm.state_data["settle_until"] = time.monotonic() + SCAN_SETTLE_SECONDS
        log_event(
            "angle",
            (
                "Frame balance horizontal adjustment "
                f"#{fsm.state_data['adjust_count']}: "
                f"x_error={margin_error:.1f}px, step={direction * step:.1f}deg, "
                f"target_pan={fsm.state_data['target_pan']:.1f}"
            ),
            throttle_seconds=0.0,
        )
        fsm.debug_problems = [
            f"FRAME_BALANCE: horizontal adjust={direction * step:.1f}deg"
        ]
        return build_motor_command(
            fsm.state_data["target_pan"],
            fsm.current_angles["tilt"],
            fsm.current_angles["height"],
            f"FRAME_BALANCE: target pan={fsm.state_data['target_pan']:.1f}",
        )

    if vertical_ok:
        fsm.state_data["phase"] = "horizontal"
        fsm.debug_problems = [
            f"FRAME_BALANCE: vertical ok, checking horizontal x={margin_error:.1f}px"
        ]
        return build_motor_command(
            fsm.current_angles["pan"],
            fsm.current_angles["tilt"],
            fsm.current_angles["height"],
            "FRAME_BALANCE: vertical ok, switching to horizontal",
        )

    error_ratio = min(1.0, abs(face_error) / max(1.0, float(context["IMG_SIZE"])))
    step = max(
        VERTICAL_BALANCE_MIN_STEP_DEGREES,
        min(VERTICAL_BALANCE_MAX_STEP_DEGREES, error_ratio * 20.0),
    )
    direction = 1.0 if face_error > 0 else -1.0
    fsm.state_data["target_tilt"] = clamp_angle(fsm.current_angles["tilt"] + (direction * step))
    fsm.state_data["phase"] = "horizontal"
    fsm.state_data["adjust_count"] += 1
    fsm.state_data["settle_until"] = time.monotonic() + SCAN_SETTLE_SECONDS
    log_event(
        "angle",
        (
            "Frame balance vertical adjustment "
            f"#{fsm.state_data['adjust_count']}: "
            f"y_error={face_error:.1f}px, step={direction * step:.1f}deg, "
            f"target_tilt={fsm.state_data['target_tilt']:.1f}"
        ),
        throttle_seconds=0.0,
    )
    fsm.debug_problems = [
        f"FRAME_BALANCE: vertical adjust={direction * step:.1f}deg"
    ]
    return build_motor_command(
        fsm.current_angles["pan"],
        fsm.state_data["target_tilt"],
        fsm.current_angles["height"],
        f"FRAME_BALANCE: target tilt={fsm.state_data['target_tilt']:.1f}",
    )


def _complete_successful_hybrid_balance(fsm):
    sequence = getattr(fsm, "auto_sequence", {})
    if not sequence.get("active") or not sequence.get("hybrid_balance_active"):
        return False

    if sequence.get("hybrid_balance_for_photo"):
        log_event(
            "state",
            "Hybrid frame balance succeeded at photo point; capturing photo.",
            throttle_seconds=0.0,
        )
        sequence["hybrid_balance_active"] = False
        sequence["hybrid_balance_for_photo"] = False
        sequence["hybrid_balance_failure_count"] = 0
        sequence["hybrid_fallback_requested"] = False
        fsm.switch_state(STATE_PHOTO_CAPTURE)
        return True

    log_event(
        "state",
        "Hybrid frame balance succeeded at intermediate point; continuing hybrid stepping.",
        throttle_seconds=0.0,
    )
    sequence["hybrid_balance_active"] = False
    sequence["hybrid_balance_for_photo"] = False
    sequence["hybrid_balance_failure_count"] = 0
    fsm.switch_state(STATE_STEPPER_POSITION)
    return True


def _fallback_from_failed_hybrid_balance(fsm, reason):
    sequence = getattr(fsm, "auto_sequence", {})
    if not sequence.get("active") or not sequence.get("hybrid_balance_active"):
        return False

    failure_count = int(sequence.get("hybrid_balance_failure_count", 0)) + 1
    sequence["hybrid_balance_failure_count"] = failure_count
    max_failures = max(1, int(STEPPER_HYBRID_BALANCE_MAX_CONSECUTIVE_FAILURES))

    if failure_count < max_failures:
        log_event(
            "state",
            (
                f"Hybrid frame balance failed {failure_count}/{max_failures} ({reason}); "
                "retrying hybrid frame balance at the same height."
            ),
            throttle_seconds=0.0,
        )
        fsm.switch_state(STATE_FRAME_BALANCE)
        return True

    log_event(
        "state",
        (
            f"Hybrid frame balance failed {failure_count}/{max_failures} consecutively ({reason}); "
            "falling back to normal adjustment."
        ),
        throttle_seconds=0.0,
    )
    sequence["hybrid_balance_active"] = False
    sequence["hybrid_balance_for_photo"] = False
    sequence["hybrid_balance_failure_count"] = 0
    sequence["hybrid_fallback_requested"] = True
    fsm.switch_state(STATE_STEPPER_POSITION)
    return True
