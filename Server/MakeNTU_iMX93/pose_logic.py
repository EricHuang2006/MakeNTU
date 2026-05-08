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
