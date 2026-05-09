import numpy as np


LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12


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
    if len(indices) == 0:
        return None

    best_kpts = None
    best_visible = -1
    for i in indices:
        idx = i[0] if isinstance(i, (list, np.ndarray)) else i
        kpts = all_keypoints[idx]
        visible = sum(1 for _x, _y, conf in kpts if conf > keypoint_conf)
        if visible > best_visible:
            best_visible = visible
            best_kpts = kpts

    if best_kpts is None:
        return None

    left_shoulder = best_kpts[LEFT_SHOULDER]
    right_shoulder = best_kpts[RIGHT_SHOULDER]
    left_elbow = best_kpts[LEFT_ELBOW]
    right_elbow = best_kpts[RIGHT_ELBOW]
    left_wrist = best_kpts[LEFT_WRIST]
    right_wrist = best_kpts[RIGHT_WRIST]
    left_hip = best_kpts[LEFT_HIP]
    right_hip = best_kpts[RIGHT_HIP]

    shoulder_width = _distance(left_shoulder, right_shoulder, keypoint_conf)
    if shoulder_width is None:
        shoulder_width = 60.0

    horizontal_tol = max(18.0, shoulder_width * 0.35)
    vertical_arm_min = max(25.0, shoulder_width * 0.45)

    if (
        _hand_straight_up(left_shoulder, left_elbow, left_wrist, vertical_arm_min, keypoint_conf) or
        _hand_straight_up(right_shoulder, right_elbow, right_wrist, vertical_arm_min, keypoint_conf)
    ):
        return "finish"

    if _elbow_horizontal(left_shoulder, left_elbow, horizontal_tol, keypoint_conf):
        if _forearm_points_up(left_elbow, left_wrist, vertical_arm_min, keypoint_conf):
            return "tilt_up"

    if _elbow_horizontal(right_shoulder, right_elbow, horizontal_tol, keypoint_conf):
        if _forearm_points_down(right_elbow, right_wrist, vertical_arm_min, keypoint_conf):
            return "tilt_down"

    if _left_arm_horizontal(left_shoulder, left_elbow, left_wrist, horizontal_tol, keypoint_conf):
        return "pan_left"

    if _right_arm_horizontal(right_shoulder, right_elbow, right_wrist, horizontal_tol, keypoint_conf):
        return "pan_right"

    if _right_up_lock(right_shoulder, right_elbow, right_wrist, horizontal_tol, vertical_arm_min, keypoint_conf):
        return "height_up"

    hip_y = _average_y(left_hip, right_hip, keypoint_conf)
    if hip_y is not None and _right_down_lock(right_elbow, right_wrist, hip_y, vertical_arm_min, keypoint_conf):
        return "height_down"

    return None


def _visible(point, keypoint_conf):
    return point[2] > keypoint_conf


def _distance(point_a, point_b, keypoint_conf):
    if not _visible(point_a, keypoint_conf) or not _visible(point_b, keypoint_conf):
        return None
    return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


def _average_y(point_a, point_b, keypoint_conf):
    ys = [point[1] for point in (point_a, point_b) if _visible(point, keypoint_conf)]
    if not ys:
        return None
    return sum(ys) / len(ys)


def _left_arm_horizontal(shoulder, elbow, wrist, horizontal_tol, keypoint_conf):
    return (
        _visible(shoulder, keypoint_conf) and
        _visible(elbow, keypoint_conf) and
        _visible(wrist, keypoint_conf) and
        abs(elbow[1] - shoulder[1]) <= horizontal_tol and
        abs(wrist[1] - shoulder[1]) <= horizontal_tol and
        wrist[0] < shoulder[0]
    )


def _elbow_horizontal(shoulder, elbow, horizontal_tol, keypoint_conf):
    return (
        _visible(shoulder, keypoint_conf) and
        _visible(elbow, keypoint_conf) and
        abs(elbow[1] - shoulder[1]) <= horizontal_tol
    )


def _right_arm_horizontal(shoulder, elbow, wrist, horizontal_tol, keypoint_conf):
    return (
        _visible(shoulder, keypoint_conf) and
        _visible(elbow, keypoint_conf) and
        _visible(wrist, keypoint_conf) and
        abs(elbow[1] - shoulder[1]) <= horizontal_tol and
        abs(wrist[1] - shoulder[1]) <= horizontal_tol and
        wrist[0] > shoulder[0]
    )


def _forearm_points_up(elbow, wrist, vertical_arm_min, keypoint_conf):
    return (
        _visible(elbow, keypoint_conf) and
        _visible(wrist, keypoint_conf) and
        wrist[1] < elbow[1] - vertical_arm_min
    )


def _forearm_points_down(elbow, wrist, vertical_arm_min, keypoint_conf):
    return (
        _visible(elbow, keypoint_conf) and
        _visible(wrist, keypoint_conf) and
        wrist[1] > elbow[1] + vertical_arm_min
    )


def _hand_straight_up(shoulder, elbow, wrist, vertical_arm_min, keypoint_conf):
    return (
        _visible(shoulder, keypoint_conf) and
        _visible(elbow, keypoint_conf) and
        _visible(wrist, keypoint_conf) and
        elbow[1] < shoulder[1] - (vertical_arm_min * 0.4) and
        wrist[1] < elbow[1] - vertical_arm_min
    )


def _right_up_lock(shoulder, elbow, wrist, horizontal_tol, vertical_arm_min, keypoint_conf):
    return (
        _visible(shoulder, keypoint_conf) and
        _visible(elbow, keypoint_conf) and
        _visible(wrist, keypoint_conf) and
        abs(elbow[1] - shoulder[1]) <= horizontal_tol and
        wrist[1] < elbow[1] - vertical_arm_min and
        wrist[0] > shoulder[0]
    )


def _right_down_lock(elbow, wrist, hip_y, vertical_arm_min, keypoint_conf):
    return (
        _visible(elbow, keypoint_conf) and
        _visible(wrist, keypoint_conf) and
        wrist[1] > elbow[1] + vertical_arm_min and
        wrist[1] > hip_y
    )
