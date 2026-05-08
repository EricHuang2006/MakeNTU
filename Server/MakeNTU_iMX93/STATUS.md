# Camera Servo Rig - State Machine (Action-Oriented)

## Overview
This state machine describes the auto-tracking and capturing process for the camera servo rig. It strictly separates state transitions (triggers) from internal state actions (do-activities).

## State Machine Diagram (Mermaid)

```mermaid
stateDiagram-v2
    %% State Definitions with Internal Actions (UML Notation: do / action)
    state "SETTING" as SETTING
    state "MANUAL_CONTROL" as MANUAL_CONTROL
    state "AUTO_CONTROL" as AUTO_CONTROL
    
    state "HORIZONTAL_SWEEP 
    state "HORIZONTAL_FIX
    state "VERTICAL_SWEEP" 
    state "VERTICAL_FIX" 
    state "FAILURE<br> do / red light 5s" as FAILURE
    state "PHOTO_CAPTURE<br> do / green light<br> do / take photo" as PHOTO_CAPTURE

    %% Initial Transitions
    [*] --> SETTING
    
    %% Height Control Group (Conceptual)
    SETTING --> MANUAL_CONTROL
    SETTING --> AUTO_CONTROL

    %% State Transitions (Triggers and Conditions)
    AUTO_CONTROL --> HORIZONTAL_SWEEP : pose / api call

    HORIZONTAL_SWEEP --> FAILURE : [Target Lost]
    HORIZONTAL_SWEEP --> HORIZONTAL_FIX : [Target Found]
    
    HORIZONTAL_FIX --> VERTICAL_SWEEP : [Angle Fixed]
    
    VERTICAL_SWEEP --> FAILURE : [Target Lost]
    VERTICAL_SWEEP --> VERTICAL_FIX : [Target Found]

    VERTICAL_FIX --> PHOTO_CAPTURE : [Angle Fixed]
    
    PHOTO_CAPTURE --> AUTO_CONTROL : [Process Complete]
    FAILURE --> AUTO_CONTROL : [Timeout Completed]

Descriptions:

SETTING : Default centered at 90 degree.

AUTO_CONTROL : Use full body mode. 

HORIZONTAL_SWEEP :  Upon entering this state, pan horiziontally to 0 degree. Next, increment 1 deg at a time; everytime a new person entered the frame, record the angle that the person would be at the center of the frame(this can be computed directly is it's possible to pan such that the person is centered; if not (for instance, if the person is at the boundaries, use the atan function to compute the angle)). Note that you may not see a face here, as the vertical angle hasn't been adjusted yet, so you can assume a person is there if you see a skeleton. To determine if it is possible to fix everyone in a single frame, record the degree when the bounding box of the leftest person leaves the frame. If after this point, you identified someone new entering the frame, we can transition to FAILURE state.

HORIZONTAL_FIX : From the recorded angles in the previous state, if l = leftest angle, r = rightest angle, pan the camera to angle (l + r) / 2. 

VERTICAL_SWEEP : Do a sweep similarly to HORIZONTAL_SWEEP, but identify people by their faces instead of their skeleton. Record their degrees.

HORIZONTAL_FIX : From the recorded angles in the previous state, if l = lowest angle, h = highest angle, define the vertical center as  (l + h) / 2. Put this point to the top 1/3 of the frame. If the highest person would leave the frame, lower this point. 

FAILURE : blink red light for for 5 seconds, then return to state CONTROLs. (the control logic for lights hasn't been implemented yet, you can put a dummy api for now).

PHOTO_CAPTURE : take a photo, and upload it to discord.