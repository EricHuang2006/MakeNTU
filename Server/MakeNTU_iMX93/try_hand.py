import cv2
import numpy as np
import serial
import socket
import struct
import time
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
C270_FOV = 60
HOST_IP = '0.0.0.0'  
PORT = 9999          

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

# Extract model's required input parameters
in_scale, in_zp = input_details[0]['quantization']
in_dtype = input_details[0]['dtype']
in_shape = input_details[0]['shape']

# Setup Socket Server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST_IP, PORT))
server_socket.listen(1)
print(f"Waiting for PC connection on port {PORT}...")

client_socket, addr = server_socket.accept()
print(f"PC Connected from: {addr}")

# ==========================================
# Aux. DC Sender, ...
# ==========================================

import requests

DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1499679166436479118/HEuB-H9TEwPLmNCVd1URnNWgPcC5YVPfOJoHUGIPVawjWBeVx_tB0uGhV8Zgz90HRDy6"

def send_image_to_discord(img, webhook_url):
    # return
    files = {
        "file": ("capture.jpg", img)
    }
    response = requests.post(webhook_url, files=files)
    
    if response.status_code in [200, 204]:
        print("Discord 傳送成功！")
    else:
        print(f"傳送失敗，錯誤碼：{response.status_code}")

    time.sleep(3)

gesture_start_time = 0  # 記錄開始舉手的時間

# ==========================================
# Aux 1-1. Image quality evaluation
# ==========================================

# ==========================================
# 3. Main Loop
# ==========================================
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Resize and convert BGR to RGB
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
                input_data = np.clip(np.round(img_norm / in_scale + in_zp), 0, 255).astype(np.uint8)
            else:
                input_data = img_rgb.astype(np.uint8)
        else:
            input_data = (img_rgb.astype(np.float32) / 255.0)

        input_data = np.expand_dims(input_data, axis=0)

        # 3. Handle NCHW layout requirements
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
        # Post-process: Face Box & Skeleton Action Tracking
        # ==========================================
        scale, zero_point = output_details[0]['quantization']
        if scale == 0.0: 
            scale = 1.0 

        # Row 4 contains person confidences
        raw_confs = output_data[4, :]
        confs = (raw_confs.astype(np.float32) - zero_point) * scale

        valid_indices = np.where(confs > CONF_THRESHOLD)[0]

        boxes = []
        scores = []
        all_keypoints = []

        for idx in valid_indices:
            # 1. Extract Person Bounding Box (Used ONLY for NMS filtering, not drawing)
            raw_cx, raw_cy, raw_w, raw_h = output_data[0:4, idx]
            cx = (raw_cx - zero_point) * scale * IMG_SIZE
            cy = (raw_cy - zero_point) * scale * IMG_SIZE
            w = (raw_w - zero_point) * scale * IMG_SIZE
            h = (raw_h - zero_point) * scale * IMG_SIZE
            x, y = int(cx - w/2), int(cy - h/2)
            
            boxes.append([x, y, int(w), int(h)])
            scores.append(float(confs[idx]))

            # 2. Extract all 17 Keypoints
            person_kpts = []
            for k in range(17):
                kx_raw = output_data[5 + k*3, idx]
                ky_raw = output_data[6 + k*3, idx]
                kconf_raw = output_data[7 + k*3, idx]
                
                kx = (kx_raw - zero_point) * scale * IMG_SIZE
                ky = (ky_raw - zero_point) * scale * IMG_SIZE
                kconf = (kconf_raw - zero_point) * scale
                
                person_kpts.append((int(kx), int(ky), float(kconf)))
            
            all_keypoints.append(person_kpts)

        # Apply NMS to remove duplicate detections
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)

        # COCO 17 Keypoint connections (Skeleton map)
        SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), 
                    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), 
                    (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
        fnd = 0
        any_hand_raised = False

        if len(indices) > 0:
            best_conf = -1.0
            target_nose_x = -1

            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                conf = scores[idx]
                kpts = all_keypoints[idx]
                
                # ----------------------------------------------------
                # A. Draw Skeleton (Blue lines, Red dots)
                # ----------------------------------------------------
                for p1, p2 in SKELETON:
                    x1, y1, c1 = kpts[p1]
                    x2, y2, c2 = kpts[p2]
                    # Only draw lines if both joints are detected with good confidence
                    if c1 > 0.3 and c2 > 0.3:
                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.circle(img, (x1, y1), 3, (0, 0, 255), -1)
                        cv2.circle(img, (x2, y2), 3, (0, 0, 255), -1)

                # ----------------------------------------------------
                # B. Dynamic Face Bounding Box
                # ----------------------------------------------------
                # Face keypoints are indices 0 to 4 (Nose, L-Eye, R-Eye, L-Ear, R-Ear)
                face_kpts = [kpts[j] for j in range(5) if kpts[j][2] > 0.3]
                
                if len(face_kpts) >= 2:
                    fnd = 1
                    xs = [p[0] for p in face_kpts]
                    ys = [p[1] for p in face_kpts]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    # Add dynamic padding to wrap the whole head based on feature distance
                    pad_x = max(15, int((x_max - x_min) * 0.5))
                    pad_y_top = max(20, int((y_max - y_min) * 1.0)) # More padding for forehead
                    pad_y_bot = max(10, int((y_max - y_min) * 0.4)) # Less padding for chin
                    
                    fx1 = max(0, x_min - pad_x)
                    fy1 = max(0, y_min - pad_y_top)
                    fx2 = min(IMG_SIZE, x_max + pad_x)
                    fy2 = min(IMG_SIZE, y_max + pad_y_bot)
                    
                    cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                    cv2.putText(img, f"Face: {conf:.2f}", (fx1, fy1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # ----------------------------------------------------
                # C. Target Selection for Motor
                # ----------------------------------------------------
                nose_x, nose_y, nose_conf = kpts[0]
                # Track the most confident person whose nose is visible
                if conf > best_conf and nose_conf > 0.3:
                    best_conf = conf
                    target_nose_x = nose_x

                # Rasied-hand detection
                nose = kpts[0]      # 鼻子
                l_wrist = kpts[9]   # 左手腕
                r_wrist = kpts[10]  # 右手腕
                
                # 影像中 Y 座標越小代表位置越高
                if nose[2] > 0.3: # 確保有抓到鼻子
                    if (l_wrist[2] > 0.3 and l_wrist[1] < nose[1]) or \
                       (r_wrist[2] > 0.3 and r_wrist[1] < nose[1]):
                        any_hand_raised = True

            # Motor tracking based on the clearest nose X coordinate
            if target_nose_x != -1:
                offset_pixel = target_nose_x - (IMG_SIZE / 2)
                degree_offset = (offset_pixel / (IMG_SIZE / 2)) * (C270_FOV / 2)
                target_angle = int(np.clip(90 + degree_offset, 0, 180))
                
                if ser:
                    ser.write(f"{target_angle}\n".encode())

        # Encode image to JPEG
        ret, encoded_img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = encoded_img.tobytes()
        # if fnd:
            # send_image_to_discord(data, "https://discord.com/api/webhooks/1499679166436479118/HEuB-H9TEwPLmNCVd1URnNWgPcC5YVPfOJoHUGIPV")
        # Pack and send
        size = struct.pack("Q", len(data))
        client_socket.sendall(size + data)


        if any_hand_raised:
            if gesture_start_time == 0:
                gesture_start_time = time.time() # 剛舉起手，記錄當下時間
                print("偵測到舉手！開始計時...")
            else:
                elapsed = time.time() - gesture_start_time
                # 在畫面上印出讀秒，方便你 Debug
                print(f"elapsed time : {elapsed}")
                cv2.putText(img, f"Hold: {elapsed:.1f}s", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if elapsed >= 2.0:
                    print("觸發拍照！正在傳送至 Discord...")
                    # 擷取原始的 frame (未壓縮的高畫質版本) 轉為二進位
                    _, buffer = cv2.imencode('.jpg', frame)
                    
                    try:
                        # 直接將記憶體中的照片傳送出去，不寫入硬碟
                        # raise Exception
                        requests.post(
                            DISCORD_WEBHOOK_URL, 
                            data={"content": "📷 系統偵測到舉手，自動拍照！"}, 
                            files={"file": ("capture.jpg", buffer.tobytes())}
                        )
                        print("照片傳送成功！")
                    except Exception as e:
                        print(f"Discord 傳送失敗: {e}")
                        
                    # 傳送完畢後重置計時器，並強制睡 1.5 秒，避免手還沒放下就連續狂拍
                    gesture_start_time = 0
                    time.sleep(1.5) 
        else:
            # 如果手放下了，就把計時器歸零
            if gesture_start_time != 0:
                print("手已放下，計時中斷。")
            gesture_start_time = 0
        

except Exception as e:
    print(f"Stream stopped: {e}")

finally:
    cap.release()
    if ser: ser.close()
    client_socket.close()
    server_socket.close()
