import math

from config import CAMERA_CY, CAMERA_FY, SCAN_STEP_DEGREES


ANGLE_EPSILON = 0.75
PERSON_MATCH_ANGLE_DEG = max(0.75, float(SCAN_STEP_DEGREES) * 0.75)
EDGE_MARGIN_PX = 8
EDGE_EXIT_CENTER_MARGIN_PX = 32
BODY_CENTER_TOLERANCE_PX = 6
FACE_CENTER_TOLERANCE_PX = 6
BODY_VISIBLE_MIN_KEYPOINTS = 5
BODY_KEYPOINT_CONFIDENCE = 0.3


def clamp_angle(angle):
    return max(0.0, min(180.0, float(angle)))


def _scale_y(pixel_y, img_size):
    raw_image_height = CAMERA_CY * 2.0
    return (pixel_y / float(img_size)) * raw_image_height


def pixel_y_to_angle(pixel_y, img_size):
    dy = _scale_y(pixel_y, img_size) - CAMERA_CY
    return math.degrees(math.atan(dy / CAMERA_FY))


def compute_centered_tilt_angle(current_tilt, center_y, img_size):
    return clamp_angle(current_tilt + pixel_y_to_angle(center_y, img_size))


def compute_top_third_tilt_angle(centered_tilt_angle, desired_ratio, img_size):
    frame_center_y = img_size * 0.5
    target_y = img_size * desired_ratio
    shift = pixel_y_to_angle(frame_center_y, img_size) - pixel_y_to_angle(target_y, img_size)
    return clamp_angle(centered_tilt_angle + shift)


def compute_top_edge_face_target_tilt(current_tilt, face_target, img_size, desired_ratio=1.0 / 3.0):
    centered_tilt = compute_centered_tilt_angle(current_tilt, face_target["center_y"], img_size)
    return compute_top_third_tilt_angle(centered_tilt, desired_ratio, img_size)


def normalize_indices(indices):
    normalized = []

    for entry in indices:
        if hasattr(entry, "__len__") and len(entry) > 0:
            normalized.append(int(entry[0]))
        else:
            normalized.append(int(entry))

    return normalized


def count_visible_keypoints(keypoints, keypoint_conf=BODY_KEYPOINT_CONFIDENCE):
    return sum(1 for _x, _y, conf in keypoints if conf > keypoint_conf)


def _keypoint_mid_x(keypoints, left_idx, right_idx, keypoint_conf=BODY_KEYPOINT_CONFIDENCE):
    left = keypoints[left_idx]
    right = keypoints[right_idx]
    if left[2] > keypoint_conf and right[2] > keypoint_conf:
        return (left[0] + right[0]) / 2.0
    return None


def compute_body_center_x(keypoints, box, keypoint_conf=BODY_KEYPOINT_CONFIDENCE):
    for left_idx, right_idx in ((5, 6), (11, 12), (1, 2), (3, 4)):
        mid_x = _keypoint_mid_x(keypoints, left_idx, right_idx, keypoint_conf)
        if mid_x is not None:
            return mid_x

    x, _y, w, _h = box
    return x + (w / 2.0)


def extract_body_targets(indices, boxes, all_keypoints, img_size, current_pan):
    targets = []

    for idx in normalize_indices(indices):
        keypoints = all_keypoints[idx]
        visible_keypoints = count_visible_keypoints(keypoints)
        if visible_keypoints < BODY_VISIBLE_MIN_KEYPOINTS:
            continue

        x, y, w, h = boxes[idx]
        center_x = compute_body_center_x(keypoints, boxes[idx])
        targets.append(
            {
                "center_x": center_x,
                "left_x": x,
                "right_x": x + w,
                "centered_angle": clamp_angle(current_pan),
                "visible_keypoints": visible_keypoints,
            }
        )

    return targets


def select_centered_body_target(targets, img_size, tolerance_px=BODY_CENTER_TOLERANCE_PX):
    if not targets:
        return None

    frame_center_x = img_size / 2.0
    centered_targets = []

    for target in targets:
        center_offset = abs(target["center_x"] - frame_center_x)
        if center_offset <= tolerance_px:
            centered_targets.append((center_offset, target))

    if not centered_targets:
        return None

    centered_targets.sort(key=lambda item: item[0])
    return centered_targets[0][1]


def extract_face_targets(face_boxes, img_size, current_tilt):
    targets = []

    for face_box in face_boxes:
        fx1, fy1, fx2, fy2, _conf = face_box
        center_y = (fy1 + fy2) / 2.0
        targets.append(
            {
                "top_y": fy1,
                "bottom_y": fy2,
                "center_y": center_y,
                "centered_angle": clamp_angle(current_tilt),
            }
        )

    return targets


def select_centered_face_target(targets, img_size, tolerance_px=FACE_CENTER_TOLERANCE_PX):
    if not targets:
        return None

    frame_center_y = img_size / 2.0
    centered_targets = []

    for target in targets:
        center_offset = abs(target["center_y"] - frame_center_y)
        if center_offset <= tolerance_px:
            centered_targets.append((center_offset, target))

    if not centered_targets:
        return None

    centered_targets.sort(key=lambda item: item[0])
    return centered_targets[0][1]


def select_top_edge_face_target(targets, img_size, top_ratio=0.10):
    if not targets:
        return None

    top_limit = img_size * top_ratio
    top_edge_targets = [target for target in targets if target["top_y"] <= top_limit]
    if not top_edge_targets:
        return None

    top_edge_targets.sort(key=lambda target: target["top_y"])
    return top_edge_targets[0]


def register_unique_angles(existing_angles, candidate_angles, threshold_deg=PERSON_MATCH_ANGLE_DEG):
    new_angles = []

    for candidate in candidate_angles:
        if all(abs(candidate - known) > threshold_deg for known in existing_angles):
            existing_angles.append(candidate)
            new_angles.append(candidate)

    return new_angles


def rightmost_target_has_right_frame(targets, img_size):
    if not targets:
        return False

    rightmost = max(targets, key=lambda target: target["center_x"])
    right_edge_reached = rightmost["right_x"] >= (img_size - EDGE_MARGIN_PX)
    center_in_exit_zone = rightmost["center_x"] >= (img_size - EDGE_EXIT_CENTER_MARGIN_PX)
    return right_edge_reached and center_in_exit_zone


def has_target_on_left_side(targets, img_size):
    if not targets:
        return False

    frame_center_x = img_size / 2.0
    return any(target["center_x"] <= frame_center_x for target in targets)


def has_target_in_left_entry_zone(targets):
    if not targets:
        return False

    return any(
        target["center_x"] <= EDGE_EXIT_CENTER_MARGIN_PX or target["left_x"] <= EDGE_MARGIN_PX
        for target in targets
    )


def get_rightmost_target(targets):
    if not targets:
        return None

    return max(targets, key=lambda target: target["center_x"])


def highest_face_near_top(targets, img_size):
    if not targets:
        return False

    highest = min(target["top_y"] for target in targets)
    return highest <= (img_size * 0.10)


def angles_reached(current_value, target_value, epsilon=ANGLE_EPSILON):
    return abs(float(current_value) - float(target_value)) <= epsilon
