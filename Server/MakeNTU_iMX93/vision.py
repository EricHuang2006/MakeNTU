import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate


def load_pose_model(model_path):
    """
    Load TFLite pose model.
    Tries NPU delegate first, falls back to CPU.
    """

    try:
        npu_delegate = [load_delegate("libethosu_delegate.so")]
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=npu_delegate
        )
        print("NPU Acceleration enabled.")

    except Exception as e:
        print(f"NPU Delegate failed, using CPU: {e}")
        interpreter = Interpreter(model_path=model_path)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    model_info = {
        "input_details": input_details,
        "output_details": output_details,
        "in_scale": input_details[0]["quantization"][0],
        "in_zp": input_details[0]["quantization"][1],
        "in_dtype": input_details[0]["dtype"],
        "in_shape": input_details[0]["shape"],
    }

    return interpreter, model_info


def preprocess_frame(frame, model_info, img_size):
    """
    Convert camera frame into model input.

    Returns:
        input_data: tensor for TFLite model
        ai_img: resized 320x320 BGR image for drawing/debug
    """

    ai_img = cv2.resize(frame, (img_size, img_size))
    img_rgb = cv2.cvtColor(ai_img, cv2.COLOR_BGR2RGB)

    in_scale = model_info["in_scale"]
    in_zp = model_info["in_zp"]
    in_dtype = model_info["in_dtype"]
    in_shape = model_info["in_shape"]

    if in_dtype == np.int8:
        if in_scale > 0:
            img_norm = img_rgb.astype(np.float32) / 255.0
            input_data = np.clip(
                np.round(img_norm / in_scale + in_zp),
                -128,
                127
            ).astype(np.int8)
        else:
            input_data = np.clip(
                img_rgb.astype(np.float32) - 128.0,
                -128,
                127
            ).astype(np.int8)

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

    # Handle NCHW layout
    if len(in_shape) == 4 and in_shape[1] == 3:
        input_data = np.transpose(input_data, (0, 3, 1, 2))

    return input_data, ai_img


def run_inference(interpreter, model_info, input_data):
    """
    Run TFLite inference and return model output.
    """

    input_details = model_info["input_details"]
    output_details = model_info["output_details"]

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])[0]

    # YOLO output may be transposed depending on export/runtime
    if output_data.shape[0] > output_data.shape[1]:
        output_data = np.transpose(output_data)

    return output_data


def decode_yolox_detection_output(output_data, model_info, img_size, conf_threshold):
    """
    Decode generic YOLOX detection output into:
        boxes: [x, y, w, h]
        scores: confidence scores
        class_ids: predicted class indices
    """

    output_details = model_info["output_details"]
    scale, zero_point = output_details[0]["quantization"]

    if scale == 0.0:
        scale = 1.0

    if output_data.shape[0] > output_data.shape[1]:
        output_data = np.transpose(output_data)

    output_data = output_data.astype(np.float32)
    output_data = (output_data - zero_point) * scale

    if output_data.ndim == 3:
        output_data = output_data.reshape(-1, output_data.shape[-1])

    if output_data.shape[1] < 6:
        return [], [], []

    raw_boxes = output_data[:, :4]
    objectness = output_data[:, 4]
    class_logits = output_data[:, 5:]

    class_scores = np.max(class_logits, axis=1)
    class_ids = np.argmax(class_logits, axis=1)
    scores = objectness * class_scores

    boxes = []
    valid_scores = []
    valid_class_ids = []

    for idx, score in enumerate(scores):
        if score <= conf_threshold:
            continue

        cx, cy, w, h = raw_boxes[idx]
        x = int(cx * img_size - (w * img_size) / 2)
        y = int(cy * img_size - (h * img_size) / 2)
        boxes.append([
            max(0, x),
            max(0, y),
            int(max(0, w * img_size)),
            int(max(0, h * img_size)),
        ])
        valid_scores.append(float(score))
        valid_class_ids.append(int(class_ids[idx]))

    return boxes, valid_scores, valid_class_ids


def decode_pose_output(output_data, model_info, img_size, conf_threshold):
    """
    Decode YOLOv8 pose output into:
        boxes: [x, y, w, h]
        scores: confidence scores
        all_keypoints: 17 COCO keypoints for each detection
    """

    output_details = model_info["output_details"]

    scale, zero_point = output_details[0]["quantization"]

    if scale == 0.0:
        scale = 1.0

    raw_confs = output_data[4, :]
    confs = (raw_confs.astype(np.float32) - zero_point) * scale

    valid_indices = np.where(confs > conf_threshold)[0]

    boxes = []
    scores = []
    all_keypoints = []

    for idx in valid_indices:
        # Person bounding box
        raw_cx, raw_cy, raw_w, raw_h = output_data[0:4, idx]

        cx = (raw_cx - zero_point) * scale * img_size
        cy = (raw_cy - zero_point) * scale * img_size
        w = (raw_w - zero_point) * scale * img_size
        h = (raw_h - zero_point) * scale * img_size

        x = int(cx - w / 2)
        y = int(cy - h / 2)

        boxes.append([x, y, int(w), int(h)])
        scores.append(float(confs[idx]))

        # 17 COCO keypoints
        person_kpts = []

        for k in range(17):
            kx_raw = output_data[5 + k * 3, idx]
            ky_raw = output_data[6 + k * 3, idx]
            kconf_raw = output_data[7 + k * 3, idx]

            kx = (kx_raw - zero_point) * scale * img_size
            ky = (ky_raw - zero_point) * scale * img_size
            kconf = (kconf_raw - zero_point) * scale

            person_kpts.append((int(kx), int(ky), float(kconf)))

        all_keypoints.append(person_kpts)

    return boxes, scores, all_keypoints


def apply_nms(boxes, scores, conf_threshold, nms_threshold):
    """
    Apply non-maximum suppression.
    """

    if len(boxes) == 0:
        return []

    return cv2.dnn.NMSBoxes(
        boxes,
        scores,
        conf_threshold,
        nms_threshold
    )