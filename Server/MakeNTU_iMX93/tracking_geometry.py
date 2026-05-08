import math

from config import CAMERA_CX, CAMERA_CY, CAMERA_FX, CAMERA_FY


ANGLE_EPSILON = 0.75
PERSON_MATCH_ANGLE_DEG = 4.0
EDGE_MARGIN_PX = 8
BODY_CENTER_TOLERANCE_PX = 6
BODY_VISIBLE_MIN_KEYPOINTS = 5
BODY_KEYPOINT_CONFIDENCE = 0.3


def clamp_angle(angle):
    return max(0.0, min(180.0, float(angle)))


def _scale_x(pixel_x, img_size):
    raw_image_width = CAMERA_CX * 2.0
    return (pixel_x / float(img_size)) * raw_image_width


def _scale_y(pixel_y, img_size):
    raw_image_height = CAMERA_CY * 2.0
    return (pixel_y / float(img_size)) * raw_image_height


def pixel_x_to_angle(pixel_x, img_size):
    dx = _scale_x(pixel_x, img_size) - CAMERA_CX
    return math.degrees(math.atan(dx / CAMERA_FX))


def pixel_y_to_angle(pixel_y, img_size):
    dy = _scale_y(pixel_y, img_size) - CAMERA_CY
    return math.degrees(math.atan(dy / CAMERA_FY))


def compute_centered_pan_angle(current_pan, center_x, img_size):
    return clamp_angle(current_pan + pixel_x_to_angle(center_x, img_size))


def compute_centered_tilt_angle(current_tilt, center_y, img_size):
    return clamp_angle(current_tilt + pixel_y_to_angle(center_y, img_size))


def compute_top_third_tilt_angle(centered_tilt_angle, desired_ratio, img_size):
    frame_center_y = img_size * 0.5
    target_y = img_size * desired_ratio
    shift = pixel_y_to_angle(frame_center_y, img_size) - pixel_y_to_angle(target_y, img_size)
    return clamp_angle(centered_tilt_angle + shift)


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


def extract_body_targets(indices, boxes, all_keypoints, img_size, current_pan):
    targets = []

    for idx in normalize_indices(indices):
        visible_keypoints = count_visible_keypoints(all_keypoints[idx])
        if visible_keypoints < BODY_VISIBLE_MIN_KEYPOINTS:
            continue

        x, y, w, h = boxes[idx]
        center_x = x + (w / 2.0)
        centered_angle = compute_centered_pan_angle(current_pan, center_x, img_size)
        targets.append(
            {
                "center_x": center_x,
                "left_x": x,
                "right_x": x + w,
                "centered_angle": centered_angle,
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
        centered_angle = compute_centered_tilt_angle(current_tilt, center_y, img_size)
        targets.append(
            {
                "top_y": fy1,
                "bottom_y": fy2,
                "center_y": center_y,
                "centered_angle": centered_angle,
            }
        )

    return targets


def register_unique_angles(existing_angles, candidate_angles, threshold_deg=PERSON_MATCH_ANGLE_DEG):
    new_angles = []

    for candidate in candidate_angles:
        if all(abs(candidate - known) > threshold_deg for known in existing_angles):
            existing_angles.append(candidate)
            new_angles.append(candidate)

    return new_angles


def leftmost_target_has_left_frame(targets):
    if not targets:
        return False

    leftmost = min(targets, key=lambda target: target["center_x"])
    return leftmost["left_x"] <= EDGE_MARGIN_PX


def has_target_on_right_side(targets, img_size):
    if not targets:
        return False

    frame_center_x = img_size / 2.0
    return any(target["center_x"] >= frame_center_x for target in targets)


def get_leftmost_target(targets):
    if not targets:
        return None

    return min(targets, key=lambda target: target["center_x"])


def highest_face_near_top(targets, img_size):
    if not targets:
        return False

    highest = min(target["top_y"] for target in targets)
    return highest <= (img_size * 0.10)


def angles_reached(current_value, target_value, epsilon=ANGLE_EPSILON):
    return abs(float(current_value) - float(target_value)) <= epsilon
