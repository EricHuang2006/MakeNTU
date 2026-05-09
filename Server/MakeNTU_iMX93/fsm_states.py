from config import SCAN_STEP_DEGREES, TILT_MAX_DELTA, TILT_MIN_DELTA

STATE_SETTING = "SETTING"
STATE_MODE_SELECT = "MODE_SELECT"
STATE_SINGLE_MODE_AUTO = "SINGLE_MODE_AUTO"
STATE_MULTI_MODE_AUTO = "MULTI_MODE_AUTO"
STATE_MANUAL_CONTROL = "MANUAL_CONTROL"
STATE_STEPPER_POSITION = "STEPPER_POSITION"
STATE_HORIZONTAL_SWEEP = "HORIZONTAL_SWEEP"
STATE_HORIZONTAL_FIX = "HORIZONTAL_FIX"
STATE_VERTICAL_SWEEP = "VERTICAL_SWEEP"
STATE_VERTICAL_FIX = "VERTICAL_FIX"
STATE_FRAME_BALANCE = "FRAME_BALANCE"
STATE_FAILURE = "FAILURE"
STATE_PHOTO_CAPTURE = "PHOTO_CAPTURE"

FAILURE_TIMEOUT_SECONDS = 5.0


def build_horizontal_scan_angles(center_pan=90.0, scan_delta=90.0):
    step = max(1, int(round(SCAN_STEP_DEGREES)))
    start = max(0, int(round(float(center_pan) - float(scan_delta))))
    stop = min(180, int(round(float(center_pan) + float(scan_delta))))
    angles = [float(angle) for angle in range(start, stop + 1, step)]

    if not angles or angles[-1] != float(stop):
        angles.append(float(stop))

    return angles


def build_vertical_scan_angles(center_tilt):
    upper = min(180.0, float(center_tilt) + float(TILT_MAX_DELTA))
    lower = max(0.0, float(center_tilt) - float(TILT_MIN_DELTA))

    start = int(round(upper))
    stop = int(round(lower))
    step = max(1, int(round(SCAN_STEP_DEGREES)))

    angles = []
    current = start
    while current >= stop:
        angles.append(float(current))
        current -= step

    if not angles or angles[-1] != float(stop):
        angles.append(float(stop))

    return angles
