import time

from event_logger import log_event
from fsm_output import build_motor_command
from fsm_states import (
    FAILURE_TIMEOUT_SECONDS,
    SCAN_PAN_ANGLES,
    build_vertical_scan_angles,
    STATE_AUTO_CONTROL,
    STATE_FAILURE,
    STATE_HORIZONTAL_FIX,
    STATE_HORIZONTAL_SWEEP,
    STATE_PHOTO_CAPTURE,
    STATE_SETTING,
    STATE_VERTICAL_FIX,
    STATE_VERTICAL_SWEEP,
)
from tracking_geometry import clamp_angle


def build_state_data(previous_state_data, current_angles, state):
    if state == STATE_HORIZONTAL_SWEEP:
        return {
            "scan_index": 0,
            "recorded_angles": [],
            "left_exit_seen": False,
            "leftmost_angle": None,
            "leave_angle": None,
            "target_pan": float(SCAN_PAN_ANGLES[0]),
            "sweep_started": False,
            "settle_until": 0.0,
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

    if new_state == STATE_AUTO_CONTROL:
        fsm.api.set_light("blue", pattern="solid")
        if hasattr(fsm.motor_rig, "center"):
            fsm.motor_rig.center()
            fsm.current_angles.update(fsm.motor_rig.current)
            fsm.default_angles = dict(fsm.motor_rig.current)
            fsm.last_command = build_motor_command(
                fsm.default_angles["pan"],
                fsm.default_angles["tilt"],
                fsm.default_angles["height"],
                "AUTO_CONTROL: reset to hardware default",
            )
        fsm.debug_problems = ["AUTO_CONTROL waiting for pose/api trigger"]
        return

    if new_state == STATE_HORIZONTAL_SWEEP:
        fsm.api.notify_capture_trigger("pose_detected")
        fsm.state_data["sweep_started"] = False
        fsm.state_data["settle_until"] = 0.0
        log_event("state", "Starting horizontal sweep: rotate from 90 to 0 without detection.", throttle_seconds=0.0)
        return

    if new_state == STATE_HORIZONTAL_FIX:
        left_angle = min(fsm.state_data["recorded_angles"])
        right_angle = max(fsm.state_data["recorded_angles"])
        fsm.state_data["target_pan"] = (left_angle + right_angle) / 2.0
        log_event(
            "angle",
            f"Horizontal fix target computed from l={left_angle:.1f}, r={right_angle:.1f} -> pan={fsm.state_data['target_pan']:.1f}",
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

    if new_state == STATE_FAILURE:
        fsm.api.set_light("red", pattern="blink", duration_s=FAILURE_TIMEOUT_SECONDS)
        fsm.state_data["timeout_at"] = time.monotonic() + FAILURE_TIMEOUT_SECONDS
        fsm.debug_problems = ["FAILURE: target lost or people exceed single frame"]
        return

    if new_state == STATE_PHOTO_CAPTURE:
        fsm.api.set_light("green", pattern="solid")


def _compute_vertical_fix_target(state_data):
    lowest_angle = min(state_data["recorded_angles"])
    highest_angle = max(state_data["recorded_angles"])
    return clamp_angle((lowest_angle + highest_angle) / 2.0)
