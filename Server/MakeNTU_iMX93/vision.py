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


def get_model_input_size(model_info):
    """Return (height, width) for the model input tensor."""

    in_shape = tuple(int(x) for x in model_info["in_shape"])

    if len(in_shape) != 4:
        raise ValueError("Unexpected model input shape: %s" % (in_shape,))

    if in_shape[1] == 3:
        return in_shape[2], in_shape[3]

    if in_shape[3] == 3:
        return in_shape[1], in_shape[2]

    return in_shape[1], in_shape[2]


def preprocess_frame(frame, model_info, img_size):
    if isinstance(img_size, tuple):
        height, width = img_size
    else:
        height = width = img_size

    ai_img = cv2.resize(frame, (width, height))
    
    # 🌟 最終拼圖 1：恢復 RGB！從分數暴跌證實了模型對顏色極度敏感
    img_process = cv2.cvtColor(ai_img, cv2.COLOR_BGR2RGB)

    in_scale = model_info["in_scale"]
    in_zp = model_info["in_zp"]
    in_dtype = model_info["in_dtype"]
    in_shape = model_info["in_shape"]

    if in_dtype == np.int8:
        if in_scale > 0:
            # 🌟 最終拼圖 2：保留完美的量化讀心術公式
            if in_zp == 0:
                img_float = (img_process.astype(np.float32) / 127.5) - 1.0
            elif in_scale > 0.5:
                img_float = img_process.astype(np.float32)
            else:
                img_float = img_process.astype(np.float32) / 255.0
                
            input_data = np.clip(
                np.round(img_float / in_scale + in_zp),
                -128, 127
            ).astype(np.int8)
        else:
            input_data = np.clip(img_process.astype(np.float32) - 128.0, -128, 127).astype(np.int8)

    elif in_dtype == np.uint8:
        if in_scale > 0:
            img_float = img_process.astype(np.float32) / 255.0
            input_data = np.clip(np.round(img_float / in_scale + in_zp), 0, 255).astype(np.uint8)
        else:
            input_data = img_process.astype(np.uint8)

    else:
        input_data = img_process.astype(np.float32) / 255.0

    input_data = np.expand_dims(input_data, axis=0)

    # NCHW 轉換
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

    output_data = interpreter.get_tensor(output_details[0]["index"])

    if output_data.ndim == 3 and output_data.shape[0] == 1:
        output_data = output_data[0]

    return output_data


def decode_yolox_detection_output(output_data, model_info, img_size, conf_threshold, debug=False):
    """
    Decode generic YOLOX detection output into:
        boxes: [x, y, w, h]
        scores: confidence scores
        class_ids: predicted class indices

    If debug=True, return looser candidate boxes for visualization.
    """

    output_details = model_info["output_details"]
    scale, zero_point = output_details[0]["quantization"]

    if scale == 0.0:
        scale = 1.0

    if output_data.ndim == 3:
        if output_data.shape[0] == 1:
            output_data = output_data[0]
        elif output_data.shape[1] == 1:
            output_data = output_data[:, 0, :]

    output_data = output_data.astype(np.float32)
    output_data = (output_data - zero_point) * scale

    print(f"[DEBUG YOLOX] shape after dequant: {output_data.shape}")

    if output_data.ndim == 2:
        if output_data.shape[0] == 21 and output_data.shape[1] != 21:
            output_data = output_data.T
            print(f"[DEBUG YOLOX] Transposed to: {output_data.shape}")
        else:
            print(f"[DEBUG YOLOX] Output already [boxes, features]: {output_data.shape}")

    if output_data.ndim == 3:
        output_data = output_data.reshape(-1, output_data.shape[-1])

    print(f"[DEBUG YOLOX] After reshape: {output_data.shape}")

    if output_data.shape[1] < 6:
        print(f"[DEBUG YOLOX] Not enough features: {output_data.shape[1]}")
        return [], [], []

    if isinstance(img_size, tuple):
        height, width = img_size
    else:
        height = width = img_size

    num_classes = output_data.shape[1] - 5
    strides = [8, 16, 32]
    hsizes = [height // stride for stride in strides]
    wsizes = [width // stride for stride in strides]

    grids = []
    expanded_strides = []
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride, dtype=np.float32))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)

    outputs = output_data.reshape(1, -1, output_data.shape[-1])
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    predictions = outputs[0]
    raw_boxes = predictions[:, :4]
    objectness = predictions[:, 4]
    class_probs = predictions[:, 5:]

    class_ids = np.argmax(class_probs, axis=1)
    class_scores = class_probs[np.arange(class_ids.shape[0]), class_ids]
    scores = objectness * class_scores

    candidate_count = int(np.sum(scores > conf_threshold))
    print(f"[DEBUG YOLOX] candidate count above {conf_threshold}: {candidate_count}")
    print(f"[DEBUG YOLOX] score range: {scores.min():.4f}-{scores.max():.4f}")
    print(f"[DEBUG YOLOX] class_ids unique: {np.unique(class_ids)}")

    boxes = []
    valid_scores = []
    valid_class_ids = []

    for idx in np.where(scores > conf_threshold)[0]:
        score = float(scores[idx])
        cid = int(class_ids[idx])
        if cid < 0 or cid >= num_classes:
            continue

        cx, cy, w, h = raw_boxes[idx]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
            continue

        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(width, x2))
        y2 = int(min(height, y2))

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            continue
        if not debug and (box_w < 8 or box_h < 8):
            continue

        boxes.append([x1, y1, box_w, box_h])
        valid_scores.append(score)
        valid_class_ids.append(cid)

    print(f"[DEBUG YOLOX] filtered candidates: {len(boxes)}")
    if len(boxes) == 0:
        return [], [], []

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

    if isinstance(img_size, tuple):
        height, width = img_size
    else:
        height = width = img_size

    raw_confs = output_data[4, :]
    confs = (raw_confs.astype(np.float32) - zero_point) * scale

    valid_indices = np.where(confs > conf_threshold)[0]

    boxes = []
    scores = []
    all_keypoints = []

    for idx in valid_indices:
        # Person bounding box
        raw_cx, raw_cy, raw_w, raw_h = output_data[0:4, idx]

        cx = (raw_cx - zero_point) * scale * width
        cy = (raw_cy - zero_point) * scale * height
        w = (raw_w - zero_point) * scale * width
        h = (raw_h - zero_point) * scale * height

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

            kx = (kx_raw - zero_point) * scale * width
            ky = (ky_raw - zero_point) * scale * height
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