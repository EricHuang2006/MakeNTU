import cv2
from config import DISPLAY_SIZE


SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
    (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6)
]


def sx(x, scale):
    return int(x * scale)


def sy(y, scale):
    return int(y * scale)


def draw_gesture_overlay(display_img, gesture_mode, gesture_label, gesture_boxes, gesture_scores, scale):
    """
    Draw gesture detection results when gesture mode is active.
    """

    if not gesture_mode:
        return

    label_text = "Gesture Mode"
    if gesture_label:
        label_text = f"Gesture: {gesture_label}"

    cv2.putText(
        display_img,
        label_text,
        (12, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    for idx, box in enumerate(gesture_boxes):
        score = gesture_scores[idx] if idx < len(gesture_scores) else 0.0
        x, y, w, h = box
        cv2.rectangle(
            display_img,
            (sx(x), sy(y)),
            (sx(x + w), sy(y + h)),
            (0, 255, 255),
            2
        )

        if gesture_label:
            cv2.putText(
                display_img,
                f"{gesture_label} {score:.2f}",
                (sx(x), max(14, sy(y) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )


def draw_status_panel(img, photo_good, quality_score, quality_problems, adjustment):
    """
    Draw compact debug/status panel on the display image.
    """

    panel_x1, panel_y1 = 8, 8
    panel_x2, panel_y2 = 390, 148

    overlay = img.copy()
    alpha = 0.55

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.rectangle(
        img,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (255, 255, 255),
        1
    )

    status_text = "GOOD" if photo_good else "ADJUST"
    status_color = (0, 255, 0) if photo_good else (0, 0, 255)

    problem_text = quality_problems[0] if quality_problems else "None"

    if len(problem_text) > 24:
        problem_text = problem_text[:21] + "..."

    lines = [
        (f"Q:{quality_score} {status_text}", status_color, 0.60),
        (f"Pan:{adjustment['pan_dir']} {adjustment['pan_amount_deg']:.1f}", (255, 255, 255), 0.60),
        (f"Tilt:{adjustment['tilt_dir']} {adjustment['tilt_amount_deg']:.1f}", (255, 255, 255), 0.60),
        (f"Size:{adjustment['size_status']}", (255, 255, 255), 0.60),
        (f"Issue:{problem_text}", (255, 255, 255), 0.54),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 34

    for text, color, scale in lines:
        cv2.putText(
            img,
            text,
            (16, y),
            font,
            scale,
            color,
            1,
            cv2.LINE_AA
        )
        y += 26


def draw_skeletons(display_img, indices, all_keypoints, scale):
    """
    Draw COCO pose skeletons.
    """

    if len(indices) == 0 or len(all_keypoints) == 0:
        return

    for i in indices:
        idx = i[0] if hasattr(i, "__len__") else i
        kpts = all_keypoints[idx]

        for p1, p2 in SKELETON:
            x1, y1, c1 = kpts[p1]
            x2, y2, c2 = kpts[p2]

            if c1 > 0.3 and c2 > 0.3:
                cv2.line(
                    display_img,
                    (sx(x1, scale), sy(y1, scale)),
                    (sx(x2, scale), sy(y2, scale)),
                    (255, 0, 0),
                    2
                )

                cv2.circle(
                    display_img,
                    (sx(x1, scale), sy(y1, scale)),
                    3,
                    (0, 0, 255),
                    -1
                )

                cv2.circle(
                    display_img,
                    (sx(x2, scale), sy(y2, scale)),
                    3,
                    (0, 0, 255),
                    -1
                )


def draw_face_boxes(display_img, face_boxes, scale):
    """
    Draw estimated face boxes.
    """

    for fx1, fy1, fx2, fy2, conf in face_boxes:
        cv2.rectangle(
            display_img,
            (sx(fx1, scale), sy(fy1, scale)),
            (sx(fx2, scale), sy(fy2, scale)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_img,
            f"{conf:.2f}",
            (sx(fx1, scale), max(14, sy(fy1, scale) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


def draw_group_box(display_img, framing, photo_good, scale):
    """
    Draw group framing box and group center.
    """

    if framing["group_box"] is None:
        return

    gx1, gy1, gx2, gy2 = framing["group_box"]
    gcx, gcy = framing["group_center"]

    group_color = (0, 255, 0) if photo_good else (0, 0, 255)

    cv2.rectangle(
        display_img,
        (sx(gx1, scale), sy(gy1, scale)),
        (sx(gx2, scale), sy(gy2, scale)),
        group_color,
        2
    )

    cv2.circle(
        display_img,
        (sx(gcx, scale), sy(gcy, scale)),
        5,
        group_color,
        -1
    )


def draw_gesture_overlay(display_img, gesture_mode, gesture_label, gesture_boxes, gesture_scores, gesture_debug_boxes, gesture_debug_scores, scale):
    """
    Draw gesture detection results when gesture mode is active.
    """

    if not gesture_mode:
        return

    label_text = "Gesture Mode"
    if gesture_label:
        label_text = f"Gesture: {gesture_label}"

    cv2.putText(
        display_img,
        label_text,
        (12, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    if len(gesture_debug_boxes) > 0 and len(gesture_debug_boxes) <= 50:
        for idx, box in enumerate(gesture_debug_boxes):
            score = gesture_debug_scores[idx] if idx < len(gesture_debug_scores) else 0.0
            x, y, w, h = box
            cv2.rectangle(
                display_img,
                (sx(x, scale), sy(y, scale)),
                (sx(x + w, scale), sy(y + h, scale)),
                (0, 128, 255),
                1
            )
            cv2.putText(
                display_img,
                f"DBG {score:.2f}",
                (sx(x, scale), max(14, sy(y, scale) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 128, 255),
                1,
                cv2.LINE_AA
            )

    for idx, box in enumerate(gesture_boxes):
        score = gesture_scores[idx] if idx < len(gesture_scores) else 0.0
        x, y, w, h = box
        cv2.rectangle(
            display_img,
            (sx(x, scale), sy(y, scale)),
            (sx(x + w, scale), sy(y + h, scale)),
            (0, 255, 255),
            2
        )

        if gesture_label:
            cv2.putText(
                display_img,
                f"{gesture_label} {score:.2f}",
                (sx(x, scale), max(14, sy(y, scale) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )


def draw_hold_timer(display_img, hold_elapsed):
    """
    Draw raised-hand hold timer.
    """

    if hold_elapsed is None:
        return

    cv2.putText(
        display_img,
        f"Hold: {hold_elapsed:.1f}s",
        (12, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        1,
        cv2.LINE_AA
    )


def draw_debug_view(
    img,
    indices,
    all_keypoints,
    face_boxes,
    framing,
    photo_good,
    quality_score,
    quality_problems,
    adjustment,
    gesture_mode=False,
    gesture_label=None,
    gesture_boxes=None,
    gesture_scores=None,
    gesture_debug_boxes=None,
    gesture_debug_scores=None,
    hold_elapsed=None,
):
    """
    Main drawing function.

    Input:
        img: 416x416 AI image

    Output:
        display_img: 640x640 debug image ready for streaming
    """

    display_img = cv2.resize(img, (DISPLAY_SIZE, DISPLAY_SIZE))
    scale = DISPLAY_SIZE / float(img.shape[0])

    draw_skeletons(display_img, indices, all_keypoints, scale)
    draw_face_boxes(display_img, face_boxes, scale)
    draw_group_box(display_img, framing, photo_good, scale)
    draw_gesture_overlay(
        display_img,
        gesture_mode,
        gesture_label,
        gesture_boxes or [],
        gesture_scores or [],
        gesture_debug_boxes or [],
        gesture_debug_scores or [],
        scale,
    )

    draw_status_panel(
        display_img,
        photo_good,
        quality_score,
        quality_problems,
        adjustment
    )

    draw_hold_timer(display_img, hold_elapsed)

    return display_img