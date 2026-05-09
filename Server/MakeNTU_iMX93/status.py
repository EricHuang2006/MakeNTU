import time

from event_logger import log_event
from fsm_output import build_adjustment_status, build_motor_command
from fsm_state_actions import update_failure, update_photo_capture
from fsm_state_idle import update_auto_control, update_manual_control, update_setting
from fsm_state_lifecycle import build_state_data, enter_state
from fsm_state_tracking import (
    update_horizontal_balance,
    update_horizontal_fix,
    update_horizontal_sweep,
    update_vertical_fix,
    update_vertical_sweep,
)
from fsm_states import (
    STATE_AUTO_CONTROL,
    STATE_FAILURE,
    STATE_HORIZONTAL_BALANCE,
    STATE_HORIZONTAL_FIX,
    STATE_HORIZONTAL_SWEEP,
    STATE_MANUAL_CONTROL,
    STATE_PHOTO_CAPTURE,
    STATE_SETTING,
    STATE_VERTICAL_FIX,
    STATE_VERTICAL_SWEEP,
)
from rig_api import DummyRigApi


STATE_HANDLERS = {
    STATE_SETTING: update_setting,
    STATE_MANUAL_CONTROL: update_manual_control,
    STATE_AUTO_CONTROL: update_auto_control,
    STATE_HORIZONTAL_SWEEP: update_horizontal_sweep,
    STATE_HORIZONTAL_FIX: update_horizontal_fix,
    STATE_HORIZONTAL_BALANCE: update_horizontal_balance,
    STATE_VERTICAL_SWEEP: update_vertical_sweep,
    STATE_VERTICAL_FIX: update_vertical_fix,
    STATE_FAILURE: update_failure,
    STATE_PHOTO_CAPTURE: update_photo_capture,
}


class CameraRigFSM:
    def __init__(self, motor_rig):
        self.motor_rig = motor_rig
        self.api = DummyRigApi()
        self.initialized = False
        self.state = None
        self.state_started_at = 0.0
        self.state_data = {}
        self.current_angles = {
            "pan": 90.0,
            "tilt": 90.0,
            "height": 90.0,
        }
        if hasattr(self.motor_rig, "current"):
            self.current_angles.update(self.motor_rig.current)
        self.default_angles = dict(self.current_angles)
        self.previous_angles = dict(self.current_angles)
        self.last_command = build_motor_command(
            self.current_angles["pan"],
            self.current_angles["tilt"],
            self.current_angles["height"],
            "FSM idle",
        )
        self.debug_problems = ["FSM idle"]

    def init(self):
        if self.initialized:
            return

        if hasattr(self.motor_rig, "current"):
            self.current_angles.update(self.motor_rig.current)
            self.default_angles = dict(self.motor_rig.current)

        self.initialized = True
        self.switch_state(STATE_SETTING)

    def deinit(self):
        if not self.initialized:
            return

        self.api.set_light("off")
        self.last_command = build_motor_command(
            self.default_angles["pan"],
            self.default_angles["tilt"],
            self.default_angles["height"],
            "FSM deinit - return to center",
        )
        self.current_angles = {
            "pan": self.last_command["pan_angle"],
            "tilt": self.last_command["tilt_angle"],
            "height": self.last_command["height_angle"],
        }
        self.initialized = False

    def request_capture(self):
        self.api.request_capture()

    def request_manual_mode(self):
        self.api.request_manual_mode()

    def request_auto_mode(self):
        self.api.request_auto_mode()

    def switch_state(self, new_state):
        previous_state = self.state
        previous_state_data = self.state_data
        log_event("state", f"State Transition: {previous_state} -> {new_state}", throttle_seconds=0.0)
        self.state = new_state
        self.state_started_at = time.monotonic()
        self.state_data = build_state_data(previous_state_data, self.current_angles, new_state)
        self.debug_problems = [f"state={new_state}"]
        enter_state(self, new_state)

    def update(self, context):
        if not self.initialized:
            raise RuntimeError("CameraRigFSM.init() must be called before update().")

        if self.api.consume_manual_mode_request(context):
            self.switch_state(STATE_MANUAL_CONTROL)
        elif self.api.consume_auto_mode_request(context) and self.state != STATE_AUTO_CONTROL:
            self.switch_state(STATE_AUTO_CONTROL)

        handler = STATE_HANDLERS.get(self.state)
        if handler is None:
            self.switch_state(STATE_SETTING)
            command = self.last_command
        else:
            command = handler(self, context)

        self.last_command = command
        self.previous_angles = dict(self.current_angles)
        self.current_angles = {
            "pan": command["pan_angle"],
            "tilt": command["tilt_angle"],
            "height": command["height_angle"],
        }
        return command

    def get_debug_view(self, indices):
        people_count = len(indices)
        quality_score = min(100, people_count * 30)
        photo_good = self.state in (STATE_VERTICAL_FIX, STATE_PHOTO_CAPTURE)

        return {
            "photo_good": photo_good,
            "quality_score": quality_score,
            "quality_problems": self.debug_problems,
            "adjustment": build_adjustment_status(self.previous_angles, self.last_command),
        }
