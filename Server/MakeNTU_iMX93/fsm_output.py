from tracking_geometry import ANGLE_EPSILON, clamp_angle


def build_motor_command(pan, tilt, height, summary, error=None):
    return {
        "pan_angle": clamp_angle(pan),
        "tilt_angle": clamp_angle(tilt),
        "height_angle": clamp_angle(height),
        "summary": summary,
        "error": error,
    }


def summarize_motion(previous_value, next_value, positive_label, negative_label):
    delta = float(next_value) - float(previous_value)
    if abs(delta) <= ANGLE_EPSILON:
        return "hold", 0.0
    if delta > 0:
        return positive_label, abs(delta)
    return negative_label, abs(delta)


def build_adjustment_status(previous_angles, motor_command):
    pan_dir, pan_amount = summarize_motion(
        previous_angles["pan"],
        motor_command["pan_angle"],
        positive_label="right",
        negative_label="left",
    )
    tilt_dir, tilt_amount = summarize_motion(
        previous_angles["tilt"],
        motor_command["tilt_angle"],
        positive_label="down",
        negative_label="up",
    )

    return {
        "pan_dir": pan_dir,
        "pan_amount_deg": pan_amount,
        "tilt_dir": tilt_dir,
        "tilt_amount_deg": tilt_amount,
        "size_status": "fsm",
        "summary": motor_command["summary"],
    }
