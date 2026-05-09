from config import (
    ENABLE_STEPPER_OUTPUT,
    STEPPER_BOTTOM_SWITCH_ACTIVE_LOW,
    STEPPER_BOTTOM_SWITCH_LINE,
    STEPPER_DIR_LINE,
    STEPPER_GPIOCHIP,
    STEPPER_HOME_DIRECTION,
    STEPPER_MAX_HOME_CM,
    STEPPER_STEP_HIGH_TIME,
    STEPPER_STEP_LINE,
    STEPPER_STEP_LOW_TIME,
    STEPPER_STEPS_PER_CM,
    STEPPER_UP_DIRECTION,
)
from event_logger import log_event
from stepper_a4988_api import A4988Axis


class StepperAxisController:
    def __init__(self):
        self.enabled = False
        self.axis = None
        self.position_cm = 0.0
        self.homed = False

        if not ENABLE_STEPPER_OUTPUT:
            log_event("system", "Stepper output disabled by config.", throttle_seconds=0.0)
            return

        try:
            self.axis = A4988Axis(
                gpiochip=STEPPER_GPIOCHIP,
                step_line=STEPPER_STEP_LINE,
                dir_line=STEPPER_DIR_LINE,
                bottom_switch_line=STEPPER_BOTTOM_SWITCH_LINE,
                bottom_switch_active_low=STEPPER_BOTTOM_SWITCH_ACTIVE_LOW,
                step_high_time=STEPPER_STEP_HIGH_TIME,
                step_low_time=STEPPER_STEP_LOW_TIME,
                steps_per_cm=STEPPER_STEPS_PER_CM,
                up_direction=STEPPER_UP_DIRECTION,
                home_direction=STEPPER_HOME_DIRECTION,
                max_home_cm=STEPPER_MAX_HOME_CM,
            )
            self.enabled = True
            log_event("system", "StepperAxisController initialized.", throttle_seconds=0.0)
        except Exception as exc:
            log_event("error", f"Failed to initialize stepper axis: {exc}", throttle_seconds=0.0)

    def home_bottom(self):
        if not self.enabled:
            self.position_cm = 0.0
            self.homed = True
            log_event("motor", "Simulated stepper home_bottom.", throttle_seconds=0.0)
            return {
                "enabled": False,
                "homed": True,
                "steps": 0,
                "position_cm": self.position_cm,
            }

        result = self.axis.home_bottom()
        self.position_cm = result["position_cm"]
        self.homed = result["homed"]
        result["enabled"] = True
        log_event(
            "motor",
            f"Stepper homed to bottom in {result['steps']} step(s).",
            throttle_seconds=0.0,
        )
        return result

    def adjust_x_cm(self, distance_cm):
        distance_cm = float(distance_cm)

        if not self.enabled:
            self.position_cm += distance_cm
            log_event(
                "motor",
                f"Simulated stepper adjust_x_cm={distance_cm:.2f}; position={self.position_cm:.2f}",
                throttle_seconds=0.0,
            )
            return {
                "enabled": False,
                "requested_cm": distance_cm,
                "moved_cm": distance_cm,
                "steps": 0,
                "position_cm": self.position_cm,
                "homed": self.homed,
            }

        result = self.axis.adjust_x_cm(distance_cm)
        self.position_cm = result["position_cm"]
        self.homed = result["homed"]
        result["enabled"] = True
        log_event(
            "motor",
            (
                f"Stepper adjust_x_cm={distance_cm:.2f}; "
                f"steps={result['steps']}; position={self.position_cm:.2f}"
            ),
            throttle_seconds=0.0,
        )
        return result

    def move_to_x_cm(self, target_cm):
        return self.adjust_x_cm(float(target_cm) - self.position_cm)

    def shutdown(self):
        if self.axis is not None:
            self.axis.close()
            self.axis = None
        self.enabled = False
