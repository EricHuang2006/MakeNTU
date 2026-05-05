import cv2
import numpy as np
import serial
import socket
import struct
import time
import os
import requests
from tflite_runtime.interpreter import Interpreter, load_delegate

# ==========================================
# 1. Configuration
# ==========================================
MODEL_PATH = 'yolov8n-pose_int8_vela.tflite'
UART_PORT = '/dev/ttyLP1'
BAUD_RATE = 115200
# Lowered threshold slightly to accommodate scaled confidence
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
IMG_SIZE = 320
DISPLAY_SIZE = 640
DISPLAY_SCALE = DISPLAY_SIZE / IMG_SIZE
C270_FOV = 60
HOST_IP = '0.0.0.0'
PORT = 9999
DISCORD_WEBHOOK_URL=os.getenv("DISCORD_WEBHOOK_URL")

# ==========================================
# Helper: coordinate scaling
# ==========================================
def sx(x):
    return int(x * DISPLAY_SCALE)

def sy(y):
    return int(y * DISPLAY_SCALE)

# ==========================================
# 2. Hardware & Network Initialization
# ==========================================
try:
    ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
    print(f"UART initialized on {UART_PORT}")
except Exception as e:
    print(f"UART Error: {e}. Running without UART.")
    ser = None

try:
    npu_delegate = [load_delegate('libethosu_delegate.so')]
    interpreter = Interpreter(model_path=MODEL_PATH, experimental_delegates=npu_delegate)
    print("NPU Acceleration enabled.")
except Exception as e:
    print(f"NPU Delegate failed, using CPU: {e}")
    interpreter = Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_scale, in_zp = input_details[0]['quantization']
in_dtype = input_details[0]['dtype']
in_shape = input_details[0]['shape']

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST_IP, PORT))
server_socket.listen(1)
print(f"Waiting for PC connection on port {PORT}...")

client_socket, addr = server_socket.accept()
print(f"PC Connected from: {addr}")


# ==========================================
# Aux. Discord Sender
# ==========================================
def send_image_to_discord(img, webhook_url):
    return
    files = {
        "file": ("capture.jpg", img)
    }

    response = requests.post(webhook_url, files=files)

    if response.status_code in [200, 204]:
        print("Discord 傳送成功！")
    else:
        print(f"傳送失敗，錯誤碼：{response.status_code}")

    time.sleep(3)


gesture_start_time = 0


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


# ==========================================
# Aux 1-2. Camera Adjustment Computation
# ==========================================
def compute_camera_adjustment(framing, img_size):
    """
    Decide camera movement direction and amount.

    Available motions:
        1. pan left/right
        2. tilt up/down

    No zoom.
    No forward/backward movement.
    """

    if framing["people_count"] == 0 or framing["group_box"] is None:
        return {
            "pan_dir": "none",
            "pan_amount_deg": 0.0,
            "tilt_dir": "none",
            "tilt_amount_deg": 0.0,
            "size_status": "unknown",
            "summary": "No person detected"
        }

    center_error_x = framing["center_error_x"]
    vertical_error = framing["vertical_error"]
    width_ratio = framing["width_ratio"]
    height_ratio = framing["height_ratio"]

    horizontal_fov = C270_FOV
    vertical_fov = C270_FOV * 0.75

    pan_deadzone_px = img_size * 0.08
    tilt_deadzone_px = img_size * 0.08

    # -----------------------------
    # 1. Pan left/right
    # -----------------------------
    if abs(center_error_x) <= pan_deadzone_px:
        pan_dir = "none"
        pan_amount_deg = 0.0
    else:
        pan_dir = "right" if center_error_x > 0 else "left"
        pan_amount_deg = abs(center_error_x / img_size) * horizontal_fov

    # -----------------------------
    # 2. Tilt up/down
    # -----------------------------
    if abs(vertical_error) <= tilt_deadzone_px:
        tilt_dir = "none"
        tilt_amount_deg = 0.0
    else:
        tilt_dir = "down" if vertical_error > 0 else "up"
        tilt_amount_deg = abs(vertical_error / img_size) * vertical_fov

    # -----------------------------
    # 3. Size status only, no action
    # -----------------------------
    if framing["people_count"] == 1:
        target_width_ratio = 0.55
        target_height_ratio = 0.70
    else:
        target_width_ratio = 0.75
        target_height_ratio = 0.80

    size_ratio = max(
        width_ratio / target_width_ratio,
        height_ratio / target_height_ratio
    )

    if size_ratio < 0.85:
        size_status = "too small"
    elif size_ratio > 1.15:
        size_status = "too large"
    else:
        size_status = "good"

    summary = (
        f"PAN {pan_dir} {pan_amount_deg:.1f} deg | "
        f"TILT {tilt_dir} {tilt_amount_deg:.1f} deg | "
        f"SIZE {size_status}"
    )

    return {
        "pan_dir": pan_dir,
        "pan_amount_deg": pan_amount_deg,
        "tilt_dir": tilt_dir,
        "tilt_amount_deg": tilt_amount_deg,
        "size_status": size_status,
        "summary": summary
    }


# ==========================================
# Aux 1-3. Status Panel Drawing
# ==========================================
def draw_status_panel(img, photo_good, quality_score, quality_problems, adjustment):
    """
    Draw a compact debug/status panel.
    This is drawn on the 640x640 display image.
    """

    panel_x1, panel_y1 = 8, 8
    panel_x2, panel_y2 = 390, 142

    overlay = img.copy()
    alpha = 0.55  # 0.0 = fully transparent, 1.0 = fully solid

    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.rectangle(img, (panel_x1, panel_y1), (panel_x2, panel_y2), (255, 255, 255), 1)

    status_text = "GOOD" if photo_good else "ADJUST"
    status_color = (0, 255, 0) if photo_good else (0, 0, 255)

    problem_text = quality_problems[0] if quality_problems else "None"

    if len(problem_text) > 24:
        problem_text = problem_text[:21] + "..."

    line1 = f"Q:{quality_score} {status_text}"
    line2 = f"Pan:{adjustment['pan_dir']} {adjustment['pan_amount_deg']:.1f}"
    line3 = f"Tilt:{adjustment['tilt_dir']} {adjustment['tilt_amount_deg']:.1f}"
    line4 = f"Size:{adjustment['size_status']}"
    line5 = f"Issue:{problem_text}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.60
    thickness = 1

    cv2.putText(img, line1, (16, 34), font, scale, status_color, thickness, cv2.LINE_AA)
    cv2.putText(img, line2, (16, 60), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(img, line3, (16, 86), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(img, line4, (16, 112), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(img, line5, (16, 136), font, 0.54, (255, 255, 255), thickness, cv2.LINE_AA)

# ==========================================
# 3. Main Loop
# ==========================================
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Resize and convert BGR to RGB for inference
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Dynamic quantization based on model metadata
        if in_dtype == np.int8:
            if in_scale > 0:
                img_norm = img_rgb.astype(np.float32) / 255.0
                input_data = np.clip(np.round(img_norm / in_scale + in_zp), -128, 127).astype(np.int8)
            else:
                input_data = np.clip(img_rgb.astype(np.float32) - 128.0, -128, 127).astype(np.int8)

        elif in_dtype == np.uint8:
            if in_scale > 0:
                img_norm = img_rgb.astype(np.float32) / 255.0
                input_data = np.clip(
                    np.round(img_norm / in_scale + in_zp),
                    0,
                    255
                ).astype(np.uint8)
            else:
                input_data = img_rgb.astype(np.uint8)

        else:
            input_data = img_rgb.astype(np.float32) / 255.0

        input_data = np.expand_dims(input_data, axis=0)

        # 3. Handle NCHW vs NHWC layout requirements
        if len(in_shape) == 4 and in_shape[1] == 3:
            input_data = np.transpose(input_data, (0, 3, 1, 2))

        # NPU Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Handle transposed outputs from TFLite
        if output_data.shape[0] > output_data.shape[1]:
            output_data = np.transpose(output_data)

        # ==========================================
        # Post-process: Pose / Skeleton / Face Logic
        # ==========================================
        scale, zero_point = output_details[0]['quantization']
        if scale == 0.0:
            scale = 1.0

        raw_confs = output_data[4, :]
        confs = (raw_confs.astype(np.float32) - zero_point) * scale

        valid_indices = np.where(confs > CONF_THRESHOLD)[0]

        boxes = []
        scores = []
        all_keypoints = []

        for idx in valid_indices:
            # 1. Person bounding box
            raw_cx, raw_cy, raw_w, raw_h = output_data[0:4, idx]

            cx = (raw_cx - zero_point) * scale * IMG_SIZE
            cy = (raw_cy - zero_point) * scale * IMG_SIZE
            w = (raw_w - zero_point) * scale * IMG_SIZE
            h = (raw_h - zero_point) * scale * IMG_SIZE

            x = int(cx - w / 2)
            y = int(cy - h / 2)

            boxes.append([x, y, int(w), int(h)])
            scores.append(float(confs[idx]))

            # 2. Extract all 17 keypoints
            person_kpts = []

            for k in range(17):
                kx_raw = output_data[5 + k * 3, idx]
                ky_raw = output_data[6 + k * 3, idx]
                kconf_raw = output_data[7 + k * 3, idx]

                kx = (kx_raw - zero_point) * scale * IMG_SIZE
                ky = (ky_raw - zero_point) * scale * IMG_SIZE
                kconf = (kconf_raw - zero_point) * scale

                person_kpts.append((int(kx), int(ky), float(kconf)))

            all_keypoints.append(person_kpts)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)

        # COCO 17 Keypoint connections (Skeleton map)
        SKELETON = [
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
            (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
            (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
            (3, 5), (4, 6)
        ]

        fnd = 0
        any_hand_raised = False
        best_conf = -1.0
        target_nose_x = -1
        face_boxes = []

        # Default no-detection state
        photo_good = False
        quality_score = 0
        quality_problems = ["No person detected"]
        framing = {
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

        if len(indices) > 0:
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                conf = scores[idx]
                kpts = all_keypoints[idx]

                # ----------------------------------------------------
                # A. Dynamic face bounding box calculation
                # ----------------------------------------------------
                face_kpts = [kpts[j] for j in range(5) if kpts[j][2] > 0.3]

                if len(face_kpts) >= 2:
                    fnd = 1

                    xs = [p[0] for p in face_kpts]
                    ys = [p[1] for p in face_kpts]

                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    pad_x = max(15, int((x_max - x_min) * 0.5))
                    pad_y_top = max(20, int((y_max - y_min) * 1.0))
                    pad_y_bot = max(10, int((y_max - y_min) * 0.4))

                    fx1 = max(0, x_min - pad_x)
                    fy1 = max(0, y_min - pad_y_top)
                    fx2 = min(IMG_SIZE, x_max + pad_x)
                    fy2 = min(IMG_SIZE, y_max + pad_y_bot)

                    face_boxes.append((fx1, fy1, fx2, fy2, conf))

                # ----------------------------------------------------
                # B. Target selection for motor
                # ----------------------------------------------------
                nose_x, nose_y, nose_conf = kpts[0]

                if conf > best_conf and nose_conf > 0.3:
                    best_conf = conf
                    target_nose_x = nose_x

                # ----------------------------------------------------
                # C. Raised-hand detection
                # ----------------------------------------------------
                nose = kpts[0]
                l_wrist = kpts[9]
                r_wrist = kpts[10]

                if nose[2] > 0.3:
                    if (l_wrist[2] > 0.3 and l_wrist[1] < nose[1]) or \
                       (r_wrist[2] > 0.3 and r_wrist[1] < nose[1]):
                        any_hand_raised = True

            # Motor tracking based on clearest nose X coordinate
            if target_nose_x != -1:
                offset_pixel = target_nose_x - (IMG_SIZE / 2)
                degree_offset = (offset_pixel / (IMG_SIZE / 2)) * (C270_FOV / 2)
                target_angle = int(np.clip(90 + degree_offset, 0, 180))

                if ser:
                    ser.write(f"{target_angle}\n".encode())

            # Image quality evaluation
            photo_good, quality_score, quality_problems, framing = evaluate_photo_quality(
                indices,
                boxes,
                all_keypoints,
                IMG_SIZE
            )

        adjustment = compute_camera_adjustment(framing, IMG_SIZE)

        print(
            "Quality:", quality_score,
            "| Good:", photo_good,
            "| Problems:", quality_problems,
            "| Adjustment:", adjustment["summary"]
        )

        # ==========================================
        # Raised-hand timer / photo trigger
        # ==========================================
        hold_elapsed_for_display = None

        if any_hand_raised:
            if gesture_start_time == 0:
                gesture_start_time = time.time()
                hold_elapsed_for_display = 0.0
                print("偵測到舉手！開始計時...")
            else:
                elapsed = time.time() - gesture_start_time
                hold_elapsed_for_display = elapsed
                print(f"elapsed time : {elapsed}")

                if elapsed >= 2.0:
                    print("觸發拍照！正在傳送至 Discord...")

                    _, buffer = cv2.imencode('.jpg', frame)

                    if DISCORD_WEBHOOK_URL:
                        try:
                            requests.post(
                                DISCORD_WEBHOOK_URL,
                                data={"content": "📷 系統偵測到舉手，自動拍照！"},
                                files={"file": ("capture.jpg", buffer.tobytes())}
                            )
                            print("照片傳送成功！")
                        except Exception as e:
                            print(f"Discord 傳送失敗: {e}")
                    else:
                        print("DISCORD_WEBHOOK_URL not set. Skipping Discord upload.")

                    gesture_start_time = 0
                    hold_elapsed_for_display = None
                    time.sleep(1.5)

        else:
            if gesture_start_time != 0:
                print("手已放下，計時中斷。")
            gesture_start_time = 0

        # ==========================================
        # Resize first, then draw labels/boxes
        # ==========================================
        display_img = cv2.resize(img, (DISPLAY_SIZE, DISPLAY_SIZE))

        # Draw skeletons
        if len(indices) > 0:
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
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

        # Draw face boxes
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

        # Draw group box
        if framing["group_box"] is not None:
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

        # Draw status panel
        draw_status_panel(
            display_img,
            photo_good,
            quality_score,
            quality_problems,
            adjustment
        )

        # Draw hand hold timer
        if hold_elapsed_for_display is not None:
            cv2.putText(
                display_img,
                f"Hold: {hold_elapsed_for_display:.1f}s",
                (12, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        # Encode image to JPEG
        ret, encoded_img = cv2.imencode(
            '.jpg',
            display_img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )

        data = encoded_img.tobytes()

        # Pack and send
        size = struct.pack("Q", len(data))
        client_socket.sendall(size + data)

except Exception as e:
    print(f"Stream stopped: {e}")

finally:
    cap.release()

    if ser:
        ser.close()

    client_socket.close()
    server_socket.close()

    print("Server shut down cleanly.")