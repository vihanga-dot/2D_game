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
