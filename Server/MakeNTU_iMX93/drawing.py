import cv2
from config import DISPLAY_SIZE, DISPLAY_SCALE


SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
    (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6)
]


def sx(x):
    return int(x * DISPLAY_SCALE)


def sy(y):
    return int(y * DISPLAY_SCALE)


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


def draw_skeletons(display_img, indices, all_keypoints):
    """
    Draw COCO pose skeletons.
    """

    if len(indices) == 0:
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
                    (sx(x1), sy(y1)),
                    (sx(x2), sy(y2)),
                    (255, 0, 0),
                    2
                )

                cv2.circle(
                    display_img,
                    (sx(x1), sy(y1)),
                    3,
                    (0, 0, 255),
                    -1
                )

                cv2.circle(
                    display_img,
                    (sx(x2), sy(y2)),
                    3,
                    (0, 0, 255),
                    -1
                )


def draw_face_boxes(display_img, face_boxes):
    """
    Draw estimated face boxes.
    """

    for fx1, fy1, fx2, fy2, conf in face_boxes:
        cv2.rectangle(
            display_img,
            (sx(fx1), sy(fy1)),
            (sx(fx2), sy(fy2)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_img,
            f"{conf:.2f}",
            (sx(fx1), max(14, sy(fy1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


def draw_group_box(display_img, framing, photo_good):
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
        (sx(gx1), sy(gy1)),
        (sx(gx2), sy(gy2)),
        group_color,
        2
    )

    cv2.circle(
        display_img,
        (sx(gcx), sy(gcy)),
        5,
        group_color,
        -1
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
    hold_elapsed=None,
):
    """
    Main drawing function.

    Input:
        img: 320x320 AI image

    Output:
        display_img: 640x640 debug image ready for streaming
    """

    display_img = cv2.resize(img, (DISPLAY_SIZE, DISPLAY_SIZE))

    draw_skeletons(display_img, indices, all_keypoints)
    draw_face_boxes(display_img, face_boxes)
    draw_group_box(display_img, framing, photo_good)

    draw_status_panel(
        display_img,
        photo_good,
        quality_score,
        quality_problems,
        adjustment
    )

    draw_hold_timer(display_img, hold_elapsed)

    return display_img