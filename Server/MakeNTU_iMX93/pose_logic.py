import numpy as np


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


def is_hand_raised(kpts, keypoint_conf=0.3):
    """
    Raised-hand rule:
        If either wrist is above the nose, count as hand raised.

    COCO keypoints:
        0 = nose
        9 = left wrist
        10 = right wrist
    """

    nose = kpts[0]
    left_wrist = kpts[9]
    right_wrist = kpts[10]

    if nose[2] <= keypoint_conf:
        return False

    left_hand_up = (
        left_wrist[2] > keypoint_conf and
        left_wrist[1] < nose[1]
    )

    right_hand_up = (
        right_wrist[2] > keypoint_conf and
        right_wrist[1] < nose[1]
    )

    return left_hand_up or right_hand_up


def analyze_people(indices, scores, all_keypoints, img_size, keypoint_conf=0.3):
    """
    Analyze final NMS people.

    Computes:
        face_boxes
        any_hand_raised
        target_nose_x for temporary motor tracking

    Returns:
        result dict
    """

    face_boxes = []
    any_hand_raised = False

    best_conf = -1.0
    target_nose_x = -1

    if len(indices) == 0:
        return {
            "face_boxes": face_boxes,
            "any_hand_raised": False,
            "target_nose_x": -1,
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

        # Hand raise
        if is_hand_raised(kpts, keypoint_conf):
            any_hand_raised = True

        # Temporary motor target: clearest visible nose
        nose_x, nose_y, nose_conf = kpts[0]

        if conf > best_conf and nose_conf > keypoint_conf:
            best_conf = conf
            target_nose_x = nose_x

    return {
        "face_boxes": face_boxes,
        "any_hand_raised": any_hand_raised,
        "target_nose_x": target_nose_x,
    }


def compute_temporary_pan_angle(target_nose_x, img_size, camera_fov):
    """
    Temporary old motor logic:
        Use nose X position to calculate one pan angle.

    Later, replace this with camera_adjustment-based control.
    """

    if target_nose_x == -1:
        return None

    offset_pixel = target_nose_x - (img_size / 2)

    degree_offset = (
        offset_pixel / (img_size / 2)
    ) * (camera_fov / 2)

    target_angle = int(np.clip(90 + degree_offset, 0, 180))

    return target_angle