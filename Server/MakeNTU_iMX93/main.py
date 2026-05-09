import cv2
import serial
import socket
import struct
import time

from cli_model_input import CliModelInput
from config import (
    BAUD_RATE,
    CONF_THRESHOLD,
    DISCORD_WEBHOOK_URL,
    ENABLE_POSE_MANUAL_GESTURES,
    ENABLE_MOTOR_OUTPUT,
    HOST_IP,
    IMG_SIZE,
    MODEL_PATH,
    NMS_THRESHOLD,
    PORT,
    UART_PORT,
)
from drawing import draw_debug_view
from event_logger import log_event, log_once_per_change
from hand_sign_classifier import HandSignClassifier
from motor_control import CameraServoRig
from pose_logic import analyze_people, classify_manual_gesture
from stepper_axis_control import StepperAxisController
from status import CameraRigFSM
from vision import (
    apply_nms,
    decode_pose_output,
    load_pose_model,
    preprocess_frame,
    run_inference,
)
from fsm_states import (
    STATE_FAILURE,
    STATE_HORIZONTAL_SWEEP,
    STATE_MODE_SELECT,
    STATE_PHOTO_CAPTURE,
    STATE_VERTICAL_FIX,
    STATE_VERTICAL_SWEEP,
)


VERTICAL_FACE_ONLY_STATES = {
    STATE_VERTICAL_SWEEP,
    STATE_VERTICAL_FIX,
    STATE_PHOTO_CAPTURE,
}


def initialize_uart():
    try:
        device = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
        log_event("system", f"UART initialized on {UART_PORT}", throttle_seconds=0.0)
        return device
    except Exception as exc:
        log_event("error", f"UART unavailable: {exc}. Running without UART.", throttle_seconds=0.0)
        return None


def initialize_socket():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST_IP, PORT))
    server_socket.listen(1)
    server_socket.settimeout(0.01)
    log_event("system", f"PC stream server listening on port {PORT}", throttle_seconds=0.0)
    return server_socket


def accept_client_if_needed(server_socket, client_socket):
    if client_socket is not None:
        return client_socket

    try:
        client_socket, addr = server_socket.accept()
        client_socket.settimeout(1.0)
        log_event("system", f"PC connected from {addr}", throttle_seconds=0.0)
        return client_socket
    except socket.timeout:
        return None


def stream_frame(client_socket, display_img):
    if client_socket is None:
        return None

    ok, encoded_img = cv2.imencode(
        ".jpg",
        display_img,
        [int(cv2.IMWRITE_JPEG_QUALITY), 80],
    )
    if not ok:
        log_event("error", "JPEG encoding failed.", throttle_seconds=0.0)
        return client_socket

    try:
        payload = encoded_img.tobytes()
        size = struct.pack("Q", len(payload))
        client_socket.sendall(size + payload)
        return client_socket
    except OSError as exc:
        log_event("error", f"PC stream disconnected: {exc}", throttle_seconds=0.0)
        client_socket.close()
        return None


def apply_motor_output(motor_rig, motor_command):
    command_signature = (
        round(float(motor_command["pan_angle"]), 1),
        round(float(motor_command["tilt_angle"]), 1),
        round(float(motor_command["height_angle"]), 1),
    )

    if not motor_rig.enabled:
        log_once_per_change(
            "error",
            "motor_rig_disabled",
            "disabled",
            "Motor rig is disabled; commands are not being sent to hardware.",
        )
        return

    if not ENABLE_MOTOR_OUTPUT:
        log_once_per_change(
            "motor",
            "motor_output_simulated",
            command_signature,
            (
                "Motor output disabled by config; "
                f"simulated command pan={command_signature[0]:.1f}, "
                f"tilt={command_signature[1]:.1f}, "
                f"height={command_signature[2]:.1f}"
            ),
        )
        return

    motor_rig.set_angles(
        pan=motor_command["pan_angle"],
        tilt=motor_command["tilt_angle"],
        height=motor_command["height_angle"],
    )
    actual_signature = (
        round(float(motor_rig.current["pan"]), 1),
        round(float(motor_rig.current["tilt"]), 1),
        round(float(motor_rig.current["height"]), 1),
    )
    log_once_per_change(
        "motor",
        "motor_output_sent",
        actual_signature,
        (
            f"Motor command sent pan={actual_signature[0]:.1f}, "
            f"tilt={actual_signature[1]:.1f}, "
            f"height={actual_signature[2]:.1f}"
        ),
    )


def main():
    cap = None
    client_socket = None
    server_socket = None
    hand_sign_classifier = None
    cli_model_input = None
    uart_device = initialize_uart()

    motor_rig = CameraServoRig()
    stepper_axis = StepperAxisController()
    if motor_rig.enabled:
        log_event(
            "motor",
            (
                "Motor rig ready at "
                f"pan={motor_rig.current['pan']:.1f}, "
                f"tilt={motor_rig.current['tilt']:.1f}, "
                f"height={motor_rig.current['height']:.1f}"
            ),
            throttle_seconds=0.0,
        )
    fsm = CameraRigFSM(motor_rig)
    fsm.init()

    startup_context = {
        "adjust_x_cm": stepper_axis.adjust_x_cm,
        "stepper_home_bottom": stepper_axis.home_bottom,
        "stepper_move_to_x_cm": stepper_axis.move_to_x_cm,
    }
    motor_command = fsm.update(startup_context)
    apply_motor_output(motor_rig, motor_command)

    frame_counter = 0

    try:
        interpreter, model_info = load_pose_model(MODEL_PATH)
        log_event("system", "Pose model loaded.", throttle_seconds=0.0)
        hand_sign_classifier = HandSignClassifier()
        cli_model_input = CliModelInput()
        server_socket = initialize_socket()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera.")
        log_event("system", "Camera opened successfully.", throttle_seconds=0.0)

        while cap.isOpened():
            client_socket = accept_client_if_needed(server_socket, client_socket)

            ret, frame = cap.read()
            if not ret:
                log_event("error", "Camera read failed; stopping main loop.", throttle_seconds=0.0)
                break

            img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            boxes = []
            scores = []
            all_keypoints = []
            indices = []
            face_boxes = []
            hand_sign = None
            manual_gesture = None
            cli_inputs = cli_model_input.pop_frame_inputs() if cli_model_input is not None else {}

            if fsm.state == STATE_MODE_SELECT:
                hand_sign = hand_sign_classifier.classify(frame) if hand_sign_classifier is not None else None
            elif fsm.state != STATE_FAILURE:
                input_data, img = preprocess_frame(frame, model_info, IMG_SIZE)
                output_data = run_inference(interpreter, model_info, input_data)
                boxes, scores, all_keypoints = decode_pose_output(
                    output_data,
                    model_info,
                    IMG_SIZE,
                    CONF_THRESHOLD,
                )
                indices = apply_nms(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)

                pose_result = analyze_people(
                    indices=indices,
                    scores=scores,
                    all_keypoints=all_keypoints,
                    img_size=IMG_SIZE,
                    keypoint_conf=0.3,
                )
                face_boxes = pose_result["face_boxes"]
                if ENABLE_POSE_MANUAL_GESTURES:
                    manual_gesture = classify_manual_gesture(
                        indices=indices,
                        all_keypoints=all_keypoints,
                        keypoint_conf=0.3,
                    )
            else:
                log_once_per_change(
                    "state",
                    "failure_detection_paused",
                    "paused",
                    "FAILURE state active: detection and stream output paused.",
                )

            hand_sign = cli_inputs.get("hand_sign", hand_sign)
            manual_gesture = cli_inputs.get("manual_gesture", manual_gesture)
            if "hand_sign" in cli_inputs:
                fsm.request_mode_selection(cli_inputs["hand_sign"])

            frame_counter += 1
            if fsm.state != STATE_HORIZONTAL_SWEEP and fsm.state != STATE_FAILURE:
                if fsm.state not in VERTICAL_FACE_ONLY_STATES:
                    log_once_per_change(
                        "detect",
                        "people_detected_count",
                        len(indices),
                        f"Detected {len(indices)} skeleton target(s).",
                    )
                log_once_per_change(
                    "detect",
                    "faces_detected_count",
                    len(face_boxes),
                    f"Detected {len(face_boxes)} face target(s).",
                )

            context = {
                "frame": frame,
                "img": img,
                "boxes": boxes,
                "scores": scores,
                "all_keypoints": all_keypoints,
                "indices": indices,
                "face_boxes": face_boxes,
                "hand_sign": hand_sign,
                "manual_gesture": manual_gesture,
                "IMG_SIZE": IMG_SIZE,
                "DISCORD_WEBHOOK_URL": DISCORD_WEBHOOK_URL,
                "frame_counter": frame_counter,
                "adjust_x_cm": stepper_axis.adjust_x_cm,
                "stepper_home_bottom": stepper_axis.home_bottom,
                "stepper_move_to_x_cm": stepper_axis.move_to_x_cm,
            }

            motor_command = fsm.update(context)
            apply_motor_output(motor_rig, motor_command)

            debug_view = fsm.get_debug_view(indices)
            display_face_boxes = face_boxes if fsm.state != STATE_HORIZONTAL_SWEEP else []
            display_indices = [] if fsm.state in VERTICAL_FACE_ONLY_STATES else indices
            display_keypoints = [] if fsm.state in VERTICAL_FACE_ONLY_STATES else all_keypoints
            display_img = draw_debug_view(
                img=img,
                indices=display_indices,
                all_keypoints=display_keypoints,
                face_boxes=display_face_boxes,
                photo_good=debug_view["photo_good"],
                quality_score=debug_view["quality_score"],
                quality_problems=debug_view["quality_problems"],
                adjustment=debug_view["adjustment"],
            )
            if fsm.state != STATE_FAILURE:
                client_socket = stream_frame(client_socket, display_img)

    except Exception as exc:
        log_event("error", f"Main loop stopped: {exc}", throttle_seconds=0.0)

    finally:
        fsm.deinit()
        if cap is not None:
            cap.release()

        if uart_device:
            uart_device.close()
        if hand_sign_classifier is not None:
            hand_sign_classifier.close()
        if cli_model_input is not None:
            cli_model_input.close()

        motor_rig.shutdown()
        stepper_axis.shutdown()
        if client_socket is not None:
            client_socket.close()
        if server_socket is not None:
            server_socket.close()
        log_event("system", "Server shut down cleanly.", throttle_seconds=0.0)


if __name__ == "__main__":
    main()
