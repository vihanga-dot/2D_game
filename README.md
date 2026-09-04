# Neon Dot-Face Tracker

A Python/OpenCV + MediaPipe webcam application that mirrors the camera feed beside a glowing landmark-only face replica. The right hand selects a neon color by extended-finger count; the left hand controls replica opacity using thumb/index pinch distance.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Press **q** while the OpenCV window is focused to exit. The app requests 640×480 and falls back to the camera's supported resolution. Detection thresholds and pinch calibration fractions are constants near the top of `main.py` and may need adjustment for another webcam or lighting conditions.

## Adding gesture features

Hand interpretation is centralized in `GestureController`. It exposes stable names such as `right_fist`, `right_peace`, `right_open_palm`, `left_pinch`, and `left_open_hand`. Register a transition callback in `main()` without changing the camera loop:

```python
controller.on("right_open_palm", lambda state: start_particle_pulse())
```

The recommended control map is:

| Gesture | Behavior | Event/state |
|---|---|---|
| Right hand, 0–5 fingers | Selects white, cyan, magenta, green, yellow, or purple | `right_fist` through `right_open_palm` |
| Right thumbs-up | Requests a temporary hologram boost | `right_thumbs_up`, `state.boost` |
| Left pinch | Sets continuous opacity | `left_pinch`, `state.opacity` |
| Left open hand | Resets opacity to 100% | `left_open_hand` |
| Both open palms | Requests a scan pulse | `both_open`, `state.scan_pulse` |

Add additional gesture names or classification rules inside `GestureController.update()`. Continuous values remain available through `state.opacity`, `state.right_fingers`, and `state.events`.

## Calibration

For faster response without jitter, tune `FACE_SMOOTHING_ALPHA` in small steps. Start at `0.35`, try `0.45`, then `0.55` if movement still feels delayed; stop increasing when the mouth or jaw begins to shimmer. If the physical hand shown for color selection appears under the wrong label, set `SWAP_HANDEDNESS = True`. Keep `SHOW_HAND_DEBUG = True` while testing: the replica panel will show the raw MediaPipe label, effective label, and counted fingers. Set it to `False` after calibration.
