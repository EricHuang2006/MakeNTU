import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# ==========================================
# Aux 1-1. Image quality evaluation
# ==========================================

def evaluate_photo_quality(indices, boxes, all_keypoints, img_size):
    """
    Evaluate whether the current frame is a good photo.

    Returns:
        photo_good: bool
        quality_score: int, 0~100
        quality_problems: list[str]
        framing: dict
    """

    people_boxes = []
    people_keypoints = []

    if len(indices) > 0:
        for i in indices:
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            people_boxes.append(boxes[idx])
            people_keypoints.append(all_keypoints[idx])

    if len(people_boxes) == 0:
        return False, 0, ["No person detected"], {
            "people_count": 0,
            "center_error_x": 0,
            "vertical_error": 0,
            "group_box": None,
            "group_center": None,
            "target_y": 0,
            "width_ratio": 0,
            "height_ratio": 0,
            "face_visibility_ratio": 0,
        }

    quality_score = 100
    quality_problems = []

    # ----------------------------------------------------
    # 1. Group bounding box
    # ----------------------------------------------------
    group_x1 = min(box[0] for box in people_boxes)
    group_y1 = min(box[1] for box in people_boxes)
    group_x2 = max(box[0] + box[2] for box in people_boxes)
    group_y2 = max(box[1] + box[3] for box in people_boxes)

    group_w = group_x2 - group_x1
    group_h = group_y2 - group_y1

    group_center_x = (group_x1 + group_x2) / 2
    group_center_y = (group_y1 + group_y2) / 2

    image_center_x = img_size / 2
    center_error_x = group_center_x - image_center_x

    # ----------------------------------------------------
    # 2. Horizontal centering
    # ----------------------------------------------------
    center_x_allowance = img_size * 0.10

    if abs(center_error_x) > center_x_allowance:
        quality_score -= 25

        if center_error_x < 0:
            quality_problems.append("Group too far left")
        else:
            quality_problems.append("Group too far right")

    # ----------------------------------------------------
    # 3. Vertical framing
    # ----------------------------------------------------
    if len(people_boxes) == 1:
        target_y = img_size * 0.40
    else:
        target_y = img_size * 0.45

    vertical_error = group_center_y - target_y
    center_y_allowance = img_size * 0.12

    if abs(vertical_error) > center_y_allowance:
        quality_score -= 20

        if vertical_error < 0:
            quality_problems.append("Group too high")
        else:
            quality_problems.append("Group too low")

    # ----------------------------------------------------
    # 4. Safe margin / cutoff check
    # ----------------------------------------------------
    safe_margin = 15

    if group_x1 < safe_margin:
        quality_score -= 15
        quality_problems.append("Too close to left edge")

    if group_x2 > img_size - safe_margin:
        quality_score -= 15
        quality_problems.append("Too close to right edge")

    if group_y1 < safe_margin:
        quality_score -= 15
        quality_problems.append("Too close to top edge")

    if group_y2 > img_size - safe_margin:
        quality_score -= 15
        quality_problems.append("Too close to bottom edge")

    # ----------------------------------------------------
    # 5. Subject size
    # ----------------------------------------------------
    width_ratio = group_w / img_size
    height_ratio = group_h / img_size

    if width_ratio < 0.25 and height_ratio < 0.35:
        quality_score -= 15
        quality_problems.append("People too small")

    if width_ratio > 0.90 or height_ratio > 0.95:
        quality_score -= 20
        quality_problems.append("People too large")

    # ----------------------------------------------------
    # 6. Face visibility from COCO keypoints
    # ----------------------------------------------------
    visible_face_count = 0

    for kpts in people_keypoints:
        face_point_count = 0

        for face_idx in [0, 1, 2, 3, 4]:
            _, _, conf = kpts[face_idx]
            if conf > 0.3:
                face_point_count += 1

        if face_point_count >= 2:
            visible_face_count += 1

    face_visibility_ratio = visible_face_count / len(people_keypoints)

    if face_visibility_ratio < 0.7:
        quality_score -= 20
        quality_problems.append("Faces not visible enough")

    # ----------------------------------------------------
    # 7. Final decision
    # ----------------------------------------------------
    quality_score = max(0, min(100, quality_score))
    photo_good = quality_score >= 60 and len(quality_problems) <= 1

    framing = {
        "people_count": len(people_boxes),
        "center_error_x": center_error_x,
        "vertical_error": vertical_error,
        "group_box": (group_x1, group_y1, group_x2, group_y2),
        "group_center": (group_center_x, group_center_y),
        "target_y": target_y,
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "face_visibility_ratio": face_visibility_ratio,
    }

    return photo_good, quality_score, quality_problems, framing