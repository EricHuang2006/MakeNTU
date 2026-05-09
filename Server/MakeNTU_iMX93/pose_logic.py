import numpy as np


LOCK_MIN_ELBOW_DEGREES = 70.0
LOCK_MAX_ELBOW_DEGREES = 110.0
STRAIGHT_MIN_ELBOW_DEGREES = 145.0
UPPER_ARM_MIN_VERTICAL_ANGLE_DEGREES = 30.0

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12

GESTURE_PAN_LEFT = "pan_left"
GESTURE_PAN_RIGHT = "pan_right"
GESTURE_TILT_UP = "tilt_up"
GESTURE_TILT_DOWN = "tilt_down"
GESTURE_HEIGHT_UP = "height_up"
GESTURE_HEIGHT_DOWN = "height_down"
GESTURE_FINISH = "finish"


def estimate_face_box(kpts, conf, img_size, keypoint_conf=0.3):
    """
    Estimate a face/head box using COCO face keypoints:
        0 nose
        1 left eye
        2 right eye
        3 left ear
        4 right ear

    Returns:
        (fx1, fy1, fx2, fy2, conf) or None
    """

    face_kpts = [kpts[j] for j in range(5) if kpts[j][2] > keypoint_conf]

    if len(face_kpts) < 2:
        return None

    xs = [p[0] for p in face_kpts]
    ys = [p[1] for p in face_kpts]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    pad_x = max(15, int((x_max - x_min) * 0.5))
    pad_y_top = max(20, int((y_max - y_min) * 1.0))
    pad_y_bot = max(10, int((y_max - y_min) * 0.4))

    fx1 = max(0, x_min - pad_x)
    fy1 = max(0, y_min - pad_y_top)
    fx2 = min(img_size, x_max + pad_x)
    fy2 = min(img_size, y_max + pad_y_bot)

    return fx1, fy1, fx2, fy2, conf


def analyze_people(indices, scores, all_keypoints, img_size, keypoint_conf=0.3):
    """
    Analyze final NMS people and return face boxes for display/FSM use.
    """

    face_boxes = []

    if len(indices) == 0:
        return {
            "face_boxes": face_boxes,
        }

    for i in indices:
        idx = i[0] if isinstance(i, (list, np.ndarray)) else i

        conf = scores[idx]
        kpts = all_keypoints[idx]

        # Face box
        face_box = estimate_face_box(
            kpts=kpts,
            conf=conf,
            img_size=img_size,
            keypoint_conf=keypoint_conf,
        )

        if face_box is not None:
            face_boxes.append(face_box)

    return {
        "face_boxes": face_boxes,
    }


def classify_manual_gesture(indices, all_keypoints, keypoint_conf=0.3):
    """
    Classify mode-3 manual-control gestures from COCO pose keypoints.

    The returned names are the command vocabulary consumed by
    fsm_state_idle.update_manual_control().
    """

    if len(indices) == 0:
        return None

    best_kpts = _select_manual_control_person(indices, all_keypoints, keypoint_conf)

    if best_kpts is None:
        return None

    left_arm = _arm_points(best_kpts, "left")
    right_arm = _arm_points(best_kpts, "right")
    shoulder_width = _distance(left_arm["shoulder"], right_arm["shoulder"], keypoint_conf)
    if shoulder_width is None:
        shoulder_width = 60.0

    thresholds = _gesture_thresholds(shoulder_width)

    if _is_hand_straight_up(left_arm, thresholds, keypoint_conf):
        return GESTURE_FINISH
    if _is_hand_straight_up(right_arm, thresholds, keypoint_conf):
        return GESTURE_FINISH

    if _upper_arm_is_horizontal_enough(left_arm, thresholds, keypoint_conf):
        if _is_l_lock(left_arm, "up", thresholds, keypoint_conf):
            return GESTURE_TILT_UP
        elif _is_l_lock(left_arm, "down", thresholds, keypoint_conf):
            return GESTURE_TILT_DOWN
        else:
            return GESTURE_PAN_LEFT
    elif _upper_arm_is_horizontal_enough(right_arm, thresholds, keypoint_conf):
        if _is_l_lock(right_arm, "up", thresholds, keypoint_conf):
            return GESTURE_HEIGHT_UP
        elif _is_l_lock(right_arm, "down", thresholds, keypoint_conf):
            return GESTURE_HEIGHT_DOWN
        else:
            return GESTURE_PAN_RIGHT
    return None


def classify_mode_selection_gesture(indices, all_keypoints, keypoint_conf=0.3):
    """
    Classify mode-selection gestures from COCO pose keypoints.

    Mode mapping:
        left arm horizontal -> 1
        either hand straight up -> 2
        right arm horizontal -> 3
    """

    if len(indices) == 0:
        return None

    best_kpts = _select_manual_control_person(indices, all_keypoints, keypoint_conf)
    if best_kpts is None:
        return None

    left_arm = _arm_points(best_kpts, "left")
    right_arm = _arm_points(best_kpts, "right")
    shoulder_width = _distance(left_arm["shoulder"], right_arm["shoulder"], keypoint_conf)
    if shoulder_width is None:
        shoulder_width = 60.0

    thresholds = _gesture_thresholds(shoulder_width)

    if (
        _is_hand_straight_up(left_arm, thresholds, keypoint_conf) or
        _is_hand_straight_up(right_arm, thresholds, keypoint_conf)
    ):
        return 2

    if _is_straight_horizontal(left_arm, thresholds, keypoint_conf, 50) and abs(left_arm["shoulder"][0] - left_arm["elbow"][0]) > thresholds["raised_x_tol"] * 1.5:
        return 1

    if _is_straight_horizontal(right_arm, thresholds, keypoint_conf, 50) and abs(left_arm["shoulder"][0] - left_arm["elbow"][0]) > thresholds["raised_x_tol"] * 1.5:
        return 3

    return None


def _arm_points(kpts, side):
    if side == "left":
        # COCO "left" is the person's anatomical left. In a normal front-facing
        # camera image, that arm extends toward larger x values.
        return {
            "side": side,
            "outward_sign": 1.0,
            "shoulder": kpts[LEFT_SHOULDER],
            "elbow": kpts[LEFT_ELBOW],
            "wrist": kpts[LEFT_WRIST],
        }

    return {
        "side": side,
        "outward_sign": -1.0,
        "shoulder": kpts[RIGHT_SHOULDER],
        "elbow": kpts[RIGHT_ELBOW],
        "wrist": kpts[RIGHT_WRIST],
    }


def _gesture_thresholds(shoulder_width):
    return {
        "horizontal_y_tol": max(18.0, shoulder_width * 0.30),
        "vertical_x_tol": max(16.0, shoulder_width * 0.25),
        "upper_arm_min": max(24.0, shoulder_width * 0.40),
        "forearm_min": max(24.0, shoulder_width * 0.40),
        "raised_x_tol": max(24.0, shoulder_width * 0.45),
    }


def _select_manual_control_person(indices, all_keypoints, keypoint_conf):
    best_kpts = None
    best_score = -1.0

    for i in indices:
        idx = i[0] if isinstance(i, (list, np.ndarray)) else i
        kpts = all_keypoints[idx]
        visible = sum(1 for _x, _y, conf in kpts if conf > keypoint_conf)
        arm_visible = sum(
            1
            for point_index in (
                LEFT_SHOULDER,
                RIGHT_SHOULDER,
                LEFT_ELBOW,
                RIGHT_ELBOW,
                LEFT_WRIST,
                RIGHT_WRIST,
            )
            if kpts[point_index][2] > keypoint_conf
        )
        score = visible + (arm_visible * 2)
        if score > best_score:
            best_score = score
            best_kpts = kpts

    return best_kpts


def _visible(point, keypoint_conf):
    return point[2] > keypoint_conf


def _distance(point_a, point_b, keypoint_conf):
    if not _visible(point_a, keypoint_conf) or not _visible(point_b, keypoint_conf):
        return None
    return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


def _arm_visible(arm, keypoint_conf):
    return (
        _visible(arm["shoulder"], keypoint_conf) and
        _visible(arm["elbow"], keypoint_conf) and
        _visible(arm["wrist"], keypoint_conf)
    )


def _is_l_lock(arm, direction, thresholds, keypoint_conf):
    if not _arm_visible(arm, keypoint_conf):
        return False

    elbow = arm["elbow"]
    wrist = arm["wrist"]

    upper_arm_horizontal = _upper_arm_is_horizontal_enough(arm, thresholds, keypoint_conf)
    forearm_vertical = abs(wrist[0] - elbow[0]) <= thresholds["vertical_x_tol"]
    if direction == "up":
        forearm_direction = wrist[1] <= elbow[1]
    else:
        forearm_direction = wrist[1] >= elbow[1]

    return (
        upper_arm_horizontal and
        forearm_vertical and
        forearm_direction 
    )


def _is_straight_horizontal(arm, thresholds, keypoint_conf, boundary = UPPER_ARM_MIN_VERTICAL_ANGLE_DEGREES):
    if not _arm_visible(arm, keypoint_conf):
        return False

    wrist = arm["wrist"]
    shoulder = arm["shoulder"]

    return (
        _upper_arm_is_horizontal_enough(arm, thresholds, keypoint_conf, boundary) and
        abs(wrist[1] - shoulder[1]) <= thresholds["horizontal_y_tol"] 
    )


def _upper_arm_is_horizontal_enough(arm, thresholds, keypoint_conf, boundary = UPPER_ARM_MIN_VERTICAL_ANGLE_DEGREES):
    shoulder = arm["shoulder"]
    elbow = arm["elbow"]
    upper_arm_length = _distance(shoulder, elbow, keypoint_conf)
    if upper_arm_length is None:
        return False

    dx = elbow[0] - shoulder[0]
    dy = elbow[1] - shoulder[1]
    angle_from_vertical = _angle_from_vertical_degrees(dx, dy)
    return (
        angle_from_vertical is not None and
        angle_from_vertical > boundary
    )


def _is_hand_straight_up(arm, thresholds, keypoint_conf):
    if not _arm_visible(arm, keypoint_conf):
        return False

    shoulder = arm["shoulder"]
    elbow = arm["elbow"]
    wrist = arm["wrist"]

    return (
        elbow[1] <= shoulder[1] - (thresholds["upper_arm_min"] * 0.35) and
        wrist[1] <= elbow[1] - (thresholds["forearm_min"] * 0.7) and
        wrist[1] <= shoulder[1] - thresholds["forearm_min"] and
        abs(elbow[0] - shoulder[0]) <= thresholds["raised_x_tol"] and
        abs(elbow[0] - wrist[0]) <= thresholds["raised_x_tol"] and
        abs(wrist[0] - shoulder[0]) <= thresholds["raised_x_tol"] 
    )

def _elbow_angle_degrees(shoulder, elbow, wrist, keypoint_conf):
    if not (
        _visible(shoulder, keypoint_conf) and
        _visible(elbow, keypoint_conf) and
        _visible(wrist, keypoint_conf)
    ):
        return None

    upper_arm = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]], dtype=float)
    forearm = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]], dtype=float)
    upper_norm = float(np.linalg.norm(upper_arm))
    forearm_norm = float(np.linalg.norm(forearm))
    if upper_norm == 0.0 or forearm_norm == 0.0:
        return None

    cos_angle = float(np.dot(upper_arm, forearm) / (upper_norm * forearm_norm))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def _angle_from_vertical_degrees(dx, dy):
    length = float(np.hypot(dx, dy))
    if length == 0.0:
        return None

    cos_angle = abs(float(dy)) / length
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def _elbow_angle_in_range(shoulder, elbow, wrist, min_degrees, max_degrees, keypoint_conf):
    angle = _elbow_angle_degrees(shoulder, elbow, wrist, keypoint_conf)
    return (
        angle is not None and
        min_degrees <= angle <= max_degrees
    )
