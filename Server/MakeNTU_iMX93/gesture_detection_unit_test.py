import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pose_logic import (
    classify_cancel_gesture,
    classify_manual_gesture,
    classify_mode_selection_gesture,
)


CONF = 0.9


class GestureDetectionTest(unittest.TestCase):
    def test_straight_arm_pan_gestures(self):
        self.assertGesture("pan_left", {7: (230, 150), 9: (275, 150)})
        self.assertGesture("pan_right", {8: (90, 150), 10: (45, 150)})

    def test_left_arm_tilt_lock_gestures(self):
        self.assertGesture("tilt_up", {7: (230, 150), 9: (230, 95)})
        self.assertGesture("tilt_down", {7: (230, 150), 9: (230, 205)})

    def test_right_arm_height_lock_gestures(self):
        self.assertGesture("height_up", {8: (90, 150), 10: (90, 95)})
        self.assertGesture("height_down", {8: (90, 150), 10: (90, 205)})

    def test_left_and_right_locks_have_different_commands(self):
        self.assertGesture("tilt_up", {7: (230, 150), 9: (230, 95)})
        self.assertGesture("tilt_down", {7: (230, 150), 9: (230, 205)})
        self.assertGesture("height_up", {8: (90, 150), 10: (90, 95)})
        self.assertGesture("height_down", {8: (90, 150), 10: (90, 205)})

    def test_finish_gesture(self):
        self.assertGesture("finish", {8: (130, 110), 10: (130, 65)})
        self.assertNoGesture({7: (190, 110), 9: (190, 65)})

    def test_height_down_requires_right_arm_l_shape(self):
        self.assertNoGesture({8: (130, 210), 10: (130, 270)})
        self.assertGesture("pan_right", {8: (90, 150), 10: (60, 195)})

    def test_mode_selection_gestures(self):
        self.assertMode(1, {7: (230, 150), 9: (275, 150)})
        self.assertMode(2, {8: (130, 110), 10: (130, 65)})
        self.assertMode(3, {8: (90, 150), 10: (45, 150)})
        self.assertIsNone(classify_mode_selection_gesture([0], self.frame({7: (190, 110), 9: (190, 65)})))

    def test_cancel_gesture_is_left_hand_raised(self):
        self.assertTrue(classify_cancel_gesture([0], self.frame({7: (190, 110), 9: (190, 65)})))
        self.assertFalse(classify_cancel_gesture([0], self.frame({8: (130, 110), 10: (130, 65)})))

    def assertGesture(self, expected, keypoint_changes):
        self.assertEqual(expected, classify_manual_gesture([0], self.frame(keypoint_changes)))

    def assertNoGesture(self, keypoint_changes):
        self.assertIsNone(classify_manual_gesture([0], self.frame(keypoint_changes)))

    def assertMode(self, expected, keypoint_changes):
        self.assertEqual(expected, classify_mode_selection_gesture([0], self.frame(keypoint_changes)))

    def frame(self, keypoint_changes):
        keypoints = self.base_keypoints()
        for index, xy in keypoint_changes.items():
            keypoints[index] = (*xy, CONF)
        return np.array([keypoints])

    def base_keypoints(self):
        keypoints = np.zeros((17, 3), dtype=float)
        for index, xy in {
            5: (190, 150),
            6: (130, 150),
            7: (190, 190),
            8: (130, 190),
            9: (190, 230),
            10: (130, 230),
            11: (140, 240),
            12: (180, 240),
        }.items():
            keypoints[index] = (*xy, CONF)
        return keypoints


if __name__ == "__main__":
    unittest.main()
