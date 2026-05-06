import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HARDWARE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'Hardware'))
if HARDWARE_DIR not in sys.path:
    sys.path.insert(0, HARDWARE_DIR)

try:
    from servo_driver import PCA9685, Servo  # type: ignore[import]
except ImportError:
    PCA9685 = None
    Servo = None

from config import (
    MOTOR_PAN_CHANNEL,
    MOTOR_TILT_CHANNEL,
    MOTOR_HEIGHT_CHANNEL,
    MOTOR_I2C_BUS,
    MOTOR_I2C_ADDRESS,
    PAN_CENTER_ANGLE,
    TILT_CENTER_ANGLE,
    HEIGHT_CENTER_ANGLE,
    PAN_ANGLE_RANGE,
    TILT_ANGLE_RANGE,
    HEIGHT_ANGLE_RANGE,
)


class CameraServoRig:
    def __init__(self):
        self.enabled = False
        self.board = None
        self.servos = {}
        self.current = {
            'pan': PAN_CENTER_ANGLE,
            'tilt': TILT_CENTER_ANGLE,
            'height': HEIGHT_CENTER_ANGLE,
        }

        if PCA9685 is None or Servo is None:
            print('Motor control unavailable: Servo driver import failed.')
            return

        try:
            self.board = PCA9685(bus=MOTOR_I2C_BUS, address=MOTOR_I2C_ADDRESS)
            self.servos['pan'] = Servo(self.board, MOTOR_PAN_CHANNEL)
            self.servos['tilt'] = Servo(self.board, MOTOR_TILT_CHANNEL)

            if MOTOR_HEIGHT_CHANNEL is not None:
                self.servos['height'] = Servo(self.board, MOTOR_HEIGHT_CHANNEL)

            self.enabled = True
            self.center()
            print('CameraServoRig initialized.')

        except Exception as exc:
            print(f'Failed to initialize motor rig: {exc}')
            self.enabled = False
            self.board = None
            self.servos = {}

    def center(self):
        for name, servo in self.servos.items():
            angle = self.current[name]
            servo.set_angle(angle)
        time.sleep(0.08)

    def set_angles(self, pan=None, tilt=None, height=None):
        if not self.enabled:
            print('Motor rig disabled; skipping set_angles.')
            return

        targets = {
            'pan': pan,
            'tilt': tilt,
            'height': height,
        }

        for name, angle in targets.items():
            if angle is None or name not in self.servos:
                continue
            safe_angle = max(0, min(180, angle))
            self.servos[name].set_angle(safe_angle)
            self.current[name] = safe_angle

    def shutdown(self):
        if not self.enabled:
            return
        for servo in self.servos.values():
            servo.off()
        if self.board is not None:
            self.board.close()
        self.enabled = False


def compute_camera_target_angles(framing, mode, img_size):
    if framing['group_box'] is None:
        return {
            'pan_angle': PAN_CENTER_ANGLE,
            'tilt_angle': TILT_CENTER_ANGLE,
            'height_angle': HEIGHT_CENTER_ANGLE,
            'summary': 'No person detected',
        }

    center_error_x = framing['center_error_x']
    vertical_error = framing['vertical_error']
    width_ratio = framing['width_ratio']
    height_ratio = framing['height_ratio']
    group_center_x, group_center_y = framing['group_center']

    target_y = img_size * (0.30 if mode == 'half_body' else 0.45)
    vertical_error = group_center_y - target_y

    pan_offset = (center_error_x / (img_size / 2.0)) * PAN_ANGLE_RANGE
    tilt_offset = (vertical_error / (img_size / 2.0)) * TILT_ANGLE_RANGE

    pan_angle = PAN_CENTER_ANGLE + pan_offset
    tilt_angle = TILT_CENTER_ANGLE + tilt_offset

    if mode == 'half_body':
        desired_height_ratio = 0.35
        height_bias = (desired_height_ratio - height_ratio) * HEIGHT_ANGLE_RANGE * 1.2
        height_angle = HEIGHT_CENTER_ANGLE + height_bias
    else:
        desired_height_ratio = 0.65
        height_bias = (desired_height_ratio - height_ratio) * HEIGHT_ANGLE_RANGE
        height_angle = HEIGHT_CENTER_ANGLE + height_bias

    height_correction = ((target_y - group_center_y) / (img_size / 2.0)) * (HEIGHT_ANGLE_RANGE * 0.3)
    height_angle += height_correction

    pan_angle = max(0, min(180, pan_angle))
    tilt_angle = max(0, min(180, tilt_angle))
    height_angle = max(0, min(180, height_angle))

    summary = (
        f"mode={mode} pan={pan_angle:.1f} tilt={tilt_angle:.1f} height={height_angle:.1f}"
    )

    return {
        'pan_angle': pan_angle,
        'tilt_angle': tilt_angle,
        'height_angle': height_angle,
        'summary': summary,
    }


def build_framing_from_detection(indices, boxes, all_keypoints, img_size):
    if len(indices) == 0:
        return {
            'people_count': 0,
            'center_error_x': 0,
            'vertical_error': 0,
            'group_box': None,
            'group_center': None,
            'width_ratio': 0,
            'height_ratio': 0,
        }

    people_boxes = []
    for i in indices:
        idx = i[0] if isinstance(i, (list, tuple)) else i
        people_boxes.append(boxes[idx])

    group_x1 = min(box[0] for box in people_boxes)
    group_y1 = min(box[1] for box in people_boxes)
    group_x2 = max(box[0] + box[2] for box in people_boxes)
    group_y2 = max(box[1] + box[3] for box in people_boxes)

    group_w = group_x2 - group_x1
    group_h = group_y2 - group_y1
    group_center_x = (group_x1 + group_x2) / 2.0
    group_center_y = (group_y1 + group_y2) / 2.0

    target_y = img_size * 0.45
    vertical_error = group_center_y - target_y

    return {
        'people_count': len(people_boxes),
        'center_error_x': group_center_x - (img_size / 2.0),
        'vertical_error': vertical_error,
        'group_box': (group_x1, group_y1, group_x2, group_y2),
        'group_center': (group_center_x, group_center_y),
        'width_ratio': group_w / img_size,
        'height_ratio': group_h / img_size,
    }
