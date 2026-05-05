from config import *

# ==========================================
# Aux 1-2. Camera Adjustment Computation
# ==========================================
def compute_camera_adjustment(framing, img_size):
    """
    Decide camera movement direction and amount.

    Available motions:
        1. pan left/right
        2. tilt up/down

    No zoom.
    No forward/backward movement.
    """

    if framing["people_count"] == 0 or framing["group_box"] is None:
        return {
            "pan_dir": "none",
            "pan_amount_deg": 0.0,
            "tilt_dir": "none",
            "tilt_amount_deg": 0.0,
            "size_status": "unknown",
            "summary": "No person detected"
        }

    center_error_x = framing["center_error_x"]
    vertical_error = framing["vertical_error"]
    width_ratio = framing["width_ratio"]
    height_ratio = framing["height_ratio"]

    horizontal_fov = C270_FOV
    vertical_fov = C270_FOV * 0.75

    pan_deadzone_px = img_size * 0.08
    tilt_deadzone_px = img_size * 0.08

    # -----------------------------
    # 1. Pan left/right
    # -----------------------------
    if abs(center_error_x) <= pan_deadzone_px:
        pan_dir = "none"
        pan_amount_deg = 0.0
    else:
        pan_dir = "right" if center_error_x > 0 else "left"
        pan_amount_deg = abs(center_error_x / img_size) * horizontal_fov

    # -----------------------------
    # 2. Tilt up/down
    # -----------------------------
    if abs(vertical_error) <= tilt_deadzone_px:
        tilt_dir = "none"
        tilt_amount_deg = 0.0
    else:
        tilt_dir = "down" if vertical_error > 0 else "up"
        tilt_amount_deg = abs(vertical_error / img_size) * vertical_fov

    # -----------------------------
    # 3. Size status only, no action
    # -----------------------------
    if framing["people_count"] == 1:
        target_width_ratio = 0.55
        target_height_ratio = 0.70
    else:
        target_width_ratio = 0.75
        target_height_ratio = 0.80

    size_ratio = max(
        width_ratio / target_width_ratio,
        height_ratio / target_height_ratio
    )

    if size_ratio < 0.85:
        size_status = "too small"
    elif size_ratio > 1.15:
        size_status = "too large"
    else:
        size_status = "good"

    summary = (
        f"PAN {pan_dir} {pan_amount_deg:.1f} deg | "
        f"TILT {tilt_dir} {tilt_amount_deg:.1f} deg | "
        f"SIZE {size_status}"
    )

    return {
        "pan_dir": pan_dir,
        "pan_amount_deg": pan_amount_deg,
        "tilt_dir": tilt_dir,
        "tilt_amount_deg": tilt_amount_deg,
        "size_status": size_status,
        "summary": summary
    }
