import time

from config import (
    HORIZONTAL_FIX_RIGHT_OFFSET_DEGREES,
    LED_PHOTO_COUNTDOWN_BLINKS,
    LED_PHOTO_COUNTDOWN_SECONDS,
    MULTI_AUTO_HORIZONTAL_SCAN_DELTA,
    SCAN_SETTLE_SECONDS,
    STEPPER_PHOTO_COUNT,
)
from event_logger import log_event
from fsm_output import build_motor_command
from fsm_states import (
    FAILURE_TIMEOUT_SECONDS,
    build_horizontal_scan_angles,
    build_vertical_scan_angles,
    STATE_FAILURE,
    STATE_FRAME_BALANCE,
    STATE_HORIZONTAL_FIX,
    STATE_HORIZONTAL_SWEEP,
    STATE_MANUAL_CONTROL,
    STATE_MODE_SELECT,
    STATE_MULTI_MODE_AUTO,
    STATE_PHOTO_CAPTURE,
    STATE_SETTING,
    STATE_SINGLE_MODE_AUTO,
    STATE_STEPPER_POSITION,
    STATE_VERTICAL_FIX,
    STATE_VERTICAL_SWEEP,
)
from tracking_geometry import clamp_angle


def build_state_data(previous_state_data, current_angles, state, fsm=None):
    if state == STATE_SETTING:
        return {
            "reset_done": False,
            "mode_select_ready_at": 0.0,
            "ready_blink_started": False,
        }
    if state == STATE_MODE_SELECT:
        return {}
    if state in (STATE_SINGLE_MODE_AUTO, STATE_MULTI_MODE_AUTO):
        return {}
    if state == STATE_MANUAL_CONTROL:
        return {
            "positioned": False,
            "target_pan": current_angles["pan"],
            "target_tilt": current_angles["tilt"],
            "gesture_ready_at": 0.0,
            "manual_photo_requested": False,
        }
    if state == STATE_HORIZONTAL_SWEEP:
        center_pan = current_angles["pan"]
        scan_delta = MULTI_AUTO_HORIZONTAL_SCAN_DELTA
        if fsm is not None:
            center_pan = fsm.default_angles.get("pan", center_pan)
            scan_delta = fsm.auto_sequence.get("horizontal_scan_delta", scan_delta)
        scan_angles = build_horizontal_scan_angles(center_pan, scan_delta)
        return {
            "scan_index": 0,
            "scan_angles": scan_angles,
            "recorded_angles": [],
            "right_exit_seen": False,
            "left_entry_clear_seen_after_exit": False,
            "initial_exit_side_ignored": False,
            "rightmost_angle": None,
            "leave_angle": None,
            "pending_body_candidate_angle": None,
            "pending_body_candidate_offset": None,
            "pending_body_candidate_signed_offset": None,
            "target_pan": float(scan_angles[0]),
            "sweep_started": False,
            "settle_until": 0.0,
        }
    if state == STATE_STEPPER_POSITION:
        return {
            "position_started": False,
            "target_cm": None,
        }
    if state == STATE_HORIZONTAL_FIX:
        return {
            "recorded_angles": list(previous_state_data.get("recorded_angles", [])),
            "target_pan": current_angles["pan"],
        }
    if state == STATE_VERTICAL_SWEEP:
        scan_angles = build_vertical_scan_angles(current_angles["tilt"])
        return {
            "scan_index": 0,
            "scan_angles": scan_angles,
            "recorded_angles": [],
            "snapshot_targets": [],
            "top_edge_target_recorded": False,
            "target_tilt": float(scan_angles[0]),
            "sweep_started": False,
            "settle_until": 0.0,
        }
    if state == STATE_VERTICAL_FIX:
        return {
            "recorded_angles": list(previous_state_data.get("recorded_angles", [])),
            "snapshot_targets": list(previous_state_data.get("snapshot_targets", [])),
            "target_tilt": current_angles["tilt"],
        }
    if state == STATE_FRAME_BALANCE:
        return {
            "target_pan": current_angles["pan"],
            "target_tilt": current_angles["tilt"],
            "settle_until": time.monotonic() + SCAN_SETTLE_SECONDS,
            "phase": "horizontal",
            "adjust_count": 0,
            "last_margin_error": None,
            "last_face_error": None,
            "missing_target_count": 0,
        }
    if state == STATE_PHOTO_CAPTURE:
        return {
            "flash_until": 0.0,
            "success_until": 0.0,
            "captured": False,
        }
    return {}


def enter_state(fsm, new_state):
    if new_state == STATE_SETTING:
        fsm.api.set_light("yellow", pattern="solid")
        fsm.last_command = build_motor_command(
            fsm.default_angles["pan"],
            fsm.default_angles["tilt"],
            fsm.default_angles["height"],
            "SETTING: hardware default position",
        )
        return

    if new_state == STATE_MODE_SELECT:
        fsm.auto_sequence["active"] = False
        fsm.api.set_light("blue", pattern="solid")
        fsm.debug_problems = ["MODE_SELECT waiting for hand sign 1, 2, or 3"]
        log_event("state", "Waiting for hand sign mode selection: 1=single, 2=multi, 3=manual.", throttle_seconds=0.0)
        return

    if new_state == STATE_SINGLE_MODE_AUTO:
        fsm.api.set_light("blue", pattern="solid")
        log_event("state", "Selected single-user auto mode.", throttle_seconds=0.0)
        return

    if new_state == STATE_MULTI_MODE_AUTO:
        fsm.api.set_light("blue", pattern="solid")
        log_event("state", "Selected multi-user auto mode.", throttle_seconds=0.0)
        return

    if new_state == STATE_MANUAL_CONTROL:
        fsm.auto_sequence["active"] = False
        fsm.auto_sequence["photo_index"] = 0
        fsm.auto_sequence["photo_count"] = 1
        fsm.api.set_light("blue", pattern="solid")
        if hasattr(fsm.motor_rig, "center"):
            fsm.motor_rig.center()
            fsm.current_angles.update(fsm.motor_rig.current)
            fsm.default_angles = dict(fsm.motor_rig.current)
            fsm.state_data["target_pan"] = fsm.default_angles["pan"]
            fsm.state_data["target_tilt"] = fsm.default_angles["tilt"]
        log_event("state", "Selected manual-control mode; centering servos and moving to 10cm.", throttle_seconds=0.0)
        return

    if new_state == STATE_STEPPER_POSITION:
        index = int(fsm.auto_sequence.get("photo_index", 0))
        total = int(fsm.auto_sequence.get("photo_count", STEPPER_PHOTO_COUNT))
        step_cm = float(fsm.auto_sequence.get("step_cm", 0.0))
        log_event(
            "state",
            (
                "Preparing stepper position "
                f"{index + 1}/{total} at x={index * step_cm:.2f}cm."
            ),
            throttle_seconds=0.0,
        )
        return

    if new_state == STATE_HORIZONTAL_SWEEP:
        fsm.api.notify_capture_trigger("pose_detected")
        fsm.state_data["sweep_started"] = False
        fsm.state_data["settle_until"] = 0.0
        if fsm.auto_sequence.get("reset_tilt_before_horizontal", False):
            fsm.state_data["target_tilt"] = fsm.default_angles["tilt"]
            fsm.auto_sequence["reset_tilt_before_horizontal"] = False
            log_event(
                "angle",
                (
                    "Resetting tilt to default before horizontal sweep: "
                    f"tilt={fsm.state_data['target_tilt']:.1f}"
                ),
                throttle_seconds=0.0,
            )
        log_event(
            "state",
            (
                "Starting horizontal sweep: "
                f"{fsm.state_data['scan_angles'][0]:.1f} to "
                f"{fsm.state_data['scan_angles'][-1]:.1f} degrees."
            ),
            throttle_seconds=0.0,
        )
        return

    if new_state == STATE_HORIZONTAL_FIX:
        left_angle = min(fsm.state_data["recorded_angles"])
        right_angle = max(fsm.state_data["recorded_angles"])
        midpoint = (left_angle + right_angle) / 2.0
        fsm.state_data["target_pan"] = clamp_angle(midpoint + HORIZONTAL_FIX_RIGHT_OFFSET_DEGREES)
        log_event(
            "angle",
            (
                f"Horizontal fix target computed from l={left_angle:.1f}, r={right_angle:.1f}, "
                f"mid={midpoint:.1f}, right_offset={HORIZONTAL_FIX_RIGHT_OFFSET_DEGREES:.1f} "
                f"-> pan={fsm.state_data['target_pan']:.1f}"
            ),
            throttle_seconds=0.0,
        )
        return

    if new_state == STATE_VERTICAL_FIX:
        fsm.state_data["target_tilt"] = _compute_vertical_fix_target(fsm.state_data)
        log_event(
            "angle",
            f"Vertical fix target computed -> tilt={fsm.state_data['target_tilt']:.1f}",
            throttle_seconds=0.0,
        )
        return

    if new_state == STATE_FRAME_BALANCE:
        log_event(
            "state",
            "Starting frame balance: alternating horizontal and vertical fine tuning.",
            throttle_seconds=0.0,
        )
        return

    if new_state == STATE_FAILURE:
        fsm.api.set_light("red", pattern="blink", duration_s=FAILURE_TIMEOUT_SECONDS)
        fsm.state_data["timeout_at"] = time.monotonic() + FAILURE_TIMEOUT_SECONDS
        fsm.debug_problems = ["FAILURE: target lost or people exceed single frame"]
        return

    if new_state == STATE_PHOTO_CAPTURE:
        countdown_seconds = float(LED_PHOTO_COUNTDOWN_SECONDS)
        blink_count = max(1, int(LED_PHOTO_COUNTDOWN_BLINKS))
        blink_interval_s = countdown_seconds / (blink_count * 2.0)
        fsm.state_data["flash_until"] = time.monotonic() + countdown_seconds
        fsm.api.set_light(
            "white",
            pattern="blink",
            duration_s=countdown_seconds,
            blink_interval_s=blink_interval_s,
        )
        log_event(
            "api",
            (
                "Photo capture countdown started: "
                f"{blink_count} white blinks over {countdown_seconds:.1f} seconds."
            ),
            throttle_seconds=0.0,
        )


def _compute_vertical_fix_target(state_data):
    lowest_angle = min(state_data["recorded_angles"])
    highest_angle = max(state_data["recorded_angles"])
    return clamp_angle((lowest_angle + highest_angle) / 2.0)
