1. Return motors and servo to default
2. Hand sign to indicate numbers; if any number is detected, branch into three states:
- 1 -> single-user, auto mode:
    This mode is basically the same as the previous main - the program automatically takes three photos at specific heights. The only change is that since there is only a single person, the panning angle is reduced to += 15 degrees, but the rest are completely the same
- 2 -> mutli-user, auto mode:
    completely the same as the previous main, panning angle remains at += 45 degrees.
- 3 -> maunal-control mode:
    In this mode, we will grant full control of the camera to the user. Center the servos, and move the stepper to middle height (10cm). The user will use body gestures to control the camera. there are 7 options : 
    - left arm horizontally raised to the left : pan the camera to the left by 4 degrees.
    - right arm horizontally raised to the right : pan the camera to the right by 4 degrees.
    - left arm horizontal up to the elbow and forearm perpendicular to the ground, upwards(left up lock) : tilt up by 4 degrees
    - right arm horizontal up to the elbow and forearm perpendicular to the ground, downwards(left down lock) : tilt down by 4 degrees
    - right arm up lock : adjust height upward by 1cm
    - right arm down lock : adjust height downward by 1cm
    - raise (any) hand straight up : finished adjusting, entering photo taking stage.

The lower level control logics are persists from the existing files, so you can refer to them.
