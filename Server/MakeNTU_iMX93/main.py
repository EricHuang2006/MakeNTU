import cv2
import serial
import socket
import struct
import time
import requests

from config import *
from vision import (
    load_pose_model,
    get_model_input_size,
    preprocess_frame,
    run_inference,
    decode_pose_output,
    decode_yolox_detection_output,
    apply_nms,
)
from photo_quality import evaluate_photo_quality
from camera_adjustment import compute_camera_adjustment
from drawing import draw_debug_view
from pose_logic import analyze_people, GestureModeSwitcher, MODE_FULL_BODY
from motor_control import (
    CameraServoRig,
    compute_camera_target_angles,
)
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
pose_input_size = get_model_input_size(model_info)
gesture_interpreter, gesture_model_info = load_pose_model(GESTURE_MODEL_PATH)
gesture_input_size = get_model_input_size(gesture_model_info)


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
SIMULATE_MOTOR_OUTPUT = False
motor_rig = CameraServoRig()
mode_switcher = GestureModeSwitcher()
pose_mode = MODE_FULL_BODY
gesture_mode = False
gesture_label = None
gesture_raw_label = None
gesture_boxes = []
gesture_scores = []
gesture_debug_boxes = []
gesture_debug_scores = []
gesture_exit_confirm = 0


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
            
        orig_h, orig_w = frame.shape[:2]

        # ==========================================
        # A. Vision pipeline
        # ==========================================
        current_interpreter = gesture_interpreter if gesture_mode else interpreter
        current_model_info = gesture_model_info if gesture_mode else model_info
        current_input_size = gesture_input_size if gesture_mode else pose_input_size

        input_data, img = preprocess_frame(
            frame,
            current_model_info,
            current_input_size
        )
        
        if gesture_mode:
            print(f"[DEBUG INPUT] Tensor Min: {input_data.min()} | Max: {input_data.max()}")

        output_data = run_inference(
            current_interpreter,
            current_model_info,
            input_data
        )

        if gesture_mode:
            gesture_debug_boxes = []
            gesture_debug_scores = []

            boxes, scores, class_ids = decode_yolox_detection_output(
                output_data,
                current_model_info,
                current_input_size,
                GESTURE_CONF_THRESHOLD,
            )

            filtered = [
                (b, s, c)
                for b, s, c in zip(boxes, scores, class_ids)
                if c < len(GESTURE_LABELS)
            ]
            if filtered:
                boxes, scores, class_ids = zip(*filtered)
                boxes, scores, class_ids = list(boxes), list(scores), list(class_ids)
            else:
                boxes, scores, class_ids = [], [], []

            indices = apply_nms(
                boxes,
                scores,
                GESTURE_CONF_THRESHOLD,
                NMS_THRESHOLD
            )

            gesture_label = None
            gesture_boxes = []
            gesture_scores = []
            all_keypoints = []

            if len(indices) > 0:
                best_index = indices[0][0] if hasattr(indices[0], "__len__") else indices[0]
                best_index = int(best_index)
                gesture_raw_label = GESTURE_LABELS[class_ids[best_index]]
                gesture_label = GESTURE_LABELS_DISPLAY[class_ids[best_index]]
                
                # [🌟 修正點 1]：放大燈座標轉換！將模型迷你世界的座標，轉換回真實攝影機的解析度
                model_w, model_h = current_input_size if isinstance(current_input_size, tuple) else (current_input_size, current_input_size)
                scale_x = orig_w / model_w
                scale_y = orig_h / model_h
                
                x, y, w, h = boxes[best_index]
                real_x = int(x * scale_x)
                real_y = int(y * scale_y)
                real_w = int(w * scale_x)
                real_h = int(h * scale_y)

                gesture_boxes = [[real_x, real_y, real_w, real_h]]
                gesture_scores = [scores[best_index]]
                
                print(f"Gesture-mode detection: class={class_ids[best_index]} label={gesture_raw_label} display={gesture_label} score={scores[best_index]:.2f}")
            else:
                gesture_raw_label = None
                gesture_exit_confirm = 0

            # (省略部分與原本相同的解除結印判斷...)
            if gesture_raw_label == GESTURE_EXIT_LABEL and (len(scores) > 0 and scores[best_index] >= 0.6):
                gesture_exit_confirm += 1
                print(f"Exit hand sign candidate frame {gesture_exit_confirm}/3")
            else:
                gesture_exit_confirm = 0

            if gesture_exit_confirm >= 3:
                print(f"偵測到亥手印 3 次，結束手勢模型，回到姿態模型。 ({gesture_label})")
                gesture_mode = False
                gesture_start_time = 0
                gesture_label = None
                gesture_raw_label = None
                gesture_boxes = []
                gesture_scores = []
                gesture_exit_confirm = 0

            face_boxes = []
            any_hand_raised = False
            photo_good = False
            quality_score = 0
            quality_problems = ["Gesture mode"]
            framing = DEFAULT_FRAMING.copy()

        else:
            boxes, scores, all_keypoints = decode_pose_output(
                output_data,
                model_info,
                current_input_size,
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
            pose_result = analyze_people(
                indices=indices,
                scores=scores,
                all_keypoints=all_keypoints,
                img_size=IMG_SIZE,
                keypoint_conf=0.3,
            )

            face_boxes = pose_result["face_boxes"]
            any_hand_raised = pose_result["any_hand_raised"]

            pose_mode, switched = mode_switcher.update(any_hand_raised)
            if switched:
                print(f"Gesture detected, switched camera mode to: {pose_mode}")

        # ==========================================
        # C. Photo quality
        # ==========================================
        if gesture_mode:
            photo_good = False
            quality_score = 0
            quality_problems = ["Gesture mode"]
            framing = DEFAULT_FRAMING.copy()
        else:
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

        motor_adjustment = compute_camera_target_angles(
            framing,
            pose_mode,
            IMG_SIZE,
        )

        if motor_rig.enabled and not SIMULATE_MOTOR_OUTPUT:
            # [🌟 修正點 2]：手勢模式下，暫停控制馬達 PID 追蹤
            if gesture_mode:
                print("[Motor OUTPUT] Motor tracking paused in Gesture Mode")
            else:
                motor_rig.set_angles(
                    pan=motor_adjustment["pan_angle"],
                    tilt=motor_adjustment["tilt_angle"],
                    height=motor_adjustment["height_angle"],
                )
        else:
            print(
                f"[Motor OUTPUT] {motor_adjustment['summary']}"
            )

        # ==========================================
        # D. Camera adjustment recommendation
        # ==========================================
        adjustment = compute_camera_adjustment(
            framing,
            IMG_SIZE
        )
        adjustment["summary"] = motor_adjustment["summary"]

        # ==========================================
        # E. Raised-hand timer / photo trigger
        # ==========================================
        hold_elapsed_for_display = None

        if not gesture_mode and any_hand_raised:
            if gesture_start_time == 0:
                gesture_start_time = time.time()
                hold_elapsed_for_display = 0.0
                print("偵測到舉手！開始計時...")
            else:
                elapsed = time.time() - gesture_start_time
                hold_elapsed_for_display = elapsed
                if elapsed >= GESTURE_HOLD_SECONDS:
                    print("舉手超過3秒，切換到手勢模型。")
                    gesture_mode = True
                    gesture_label = None
                    gesture_raw_label = None
                    gesture_boxes = []
                    gesture_scores = []
                    gesture_start_time = 0
                    hold_elapsed_for_display = None
                elif elapsed >= 5.0:
                    print("\n" + "="*60)
                    print("📸 PHOTO TRIGGERED!")
                    send_frame_to_discord(frame, DISCORD_WEBHOOK_URL)
                    gesture_start_time = 0
                    hold_elapsed_for_display = None
                    time.sleep(1.5)
        else:
            gesture_start_time = 0

        # ==========================================
        # F. Draw debug view
        # ==========================================
        
        # [🌟 修正點 3 (加碼加倍)]：因為我們已經在上面把手勢座標放大了，
        # 這裡直接把傳入的底圖從 img 改成 frame（原始高畫質畫面）！
        # 這樣綠色框框就會精準鎖定在手部，而且不再是模糊的正方形畫面了。
        
        display_img = draw_debug_view(
            img=frame if gesture_mode else img, # 為了不影響你原本寫好的 pose 畫圖邏輯，只在結印時使用高畫質
            indices=indices,
            all_keypoints=all_keypoints,
            face_boxes=face_boxes,
            framing=framing,
            photo_good=photo_good,
            quality_score=quality_score,
            quality_problems=quality_problems,
            adjustment=adjustment,
            gesture_mode=gesture_mode,
            gesture_label=gesture_label,
            gesture_boxes=gesture_boxes,
            gesture_scores=gesture_scores,
            gesture_debug_boxes=gesture_debug_boxes,
            gesture_debug_scores=gesture_debug_scores,
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