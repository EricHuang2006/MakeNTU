import cv2
import serial
import socket
import struct
import time
import requests

from config import *
from vision import (
    load_pose_model,
    preprocess_frame,
    run_inference,
    decode_pose_output,
    apply_nms,
)
from photo_quality import evaluate_photo_quality
from camera_adjustment import compute_camera_adjustment
from drawing import draw_debug_view
from pose_logic import analyze_people, compute_temporary_pan_angle
from DC_sender import send_frame_to_discord


# ==========================================
# 1. Hardware & Network Initialization
# ==========================================
try:
    ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
    print(f"UART initialized on {UART_PORT}")

except Exception as e:
    print(f"UART Error: {e}. Running without UART.")
    ser = None


interpreter, model_info = load_pose_model(MODEL_PATH)


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST_IP, PORT))
server_socket.listen(1)

print(f"Waiting for PC connection on port {PORT}...")
client_socket, addr = server_socket.accept()
print(f"PC Connected from: {addr}")


# ==========================================
# 2. Runtime State
# ==========================================
gesture_start_time = 0


DEFAULT_FRAMING = {
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


# ==========================================
# 3. Main Loop
# ==========================================
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # ==========================================
        # A. Vision pipeline
        # ==========================================
        input_data, img = preprocess_frame(
            frame,
            model_info,
            IMG_SIZE
        )

        output_data = run_inference(
            interpreter,
            model_info,
            input_data
        )

        boxes, scores, all_keypoints = decode_pose_output(
            output_data,
            model_info,
            IMG_SIZE,
            CONF_THRESHOLD
        )

        indices = apply_nms(
            boxes,
            scores,
            CONF_THRESHOLD,
            NMS_THRESHOLD
        )

        # ==========================================
        # B. Pose logic
        # ==========================================
        pose_result = analyze_people(
            indices=indices,
            scores=scores,
            all_keypoints=all_keypoints,
            img_size=IMG_SIZE,
            keypoint_conf=0.3,
        )

        face_boxes = pose_result["face_boxes"]
        any_hand_raised = pose_result["any_hand_raised"]
        target_nose_x = pose_result["target_nose_x"]

        # Temporary old motor tracking.
        # Later this should be replaced by camera_adjustment output.
        target_angle = compute_temporary_pan_angle(
            target_nose_x=target_nose_x,
            img_size=IMG_SIZE,
            camera_fov=C270_FOV,
        )

        if target_angle is not None and ser:
            ser.write(f"{target_angle}\n".encode())

        # ==========================================
        # C. Photo quality
        # ==========================================
        photo_good = False
        quality_score = 0
        quality_problems = ["No person detected"]
        framing = DEFAULT_FRAMING.copy()

        if len(indices) > 0:
            photo_good, quality_score, quality_problems, framing = evaluate_photo_quality(
                indices,
                boxes,
                all_keypoints,
                IMG_SIZE
            )

        # ==========================================
        # D. Camera adjustment recommendation
        # ==========================================
        adjustment = compute_camera_adjustment(
            framing,
            IMG_SIZE
        )

        print(
            "Quality:", quality_score,
            "| Good:", photo_good,
            "| Problems:", quality_problems,
            "| Adjustment:", adjustment["summary"]
        )

        # ==========================================
        # E. Raised-hand timer / photo trigger
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
                print(f"elapsed time: {elapsed:.2f}")

                if elapsed >= 2.0:
                    print("觸發拍照！正在傳送至 Discord...")
                    send_frame_to_discord(frame, DISCORD_WEBHOOK_URL)
                    gesture_start_time = 0
                    hold_elapsed_for_display = None
                    time.sleep(1.5)

        else:
            if gesture_start_time != 0:
                print("手已放下，計時中斷。")

            gesture_start_time = 0

        # ==========================================
        # F. Draw debug view
        # ==========================================
        display_img = draw_debug_view(
            img=img,
            indices=indices,
            all_keypoints=all_keypoints,
            face_boxes=face_boxes,
            framing=framing,
            photo_good=photo_good,
            quality_score=quality_score,
            quality_problems=quality_problems,
            adjustment=adjustment,
            hold_elapsed=hold_elapsed_for_display,
        )

        # ==========================================
        # G. Encode and stream to laptop
        # ==========================================
        ret, encoded_img = cv2.imencode(
            ".jpg",
            display_img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )

        if not ret:
            print("JPEG encoding failed.")
            continue

        data = encoded_img.tobytes()

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