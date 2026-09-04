"""Neon Dot-Face Tracker

Run with: python main.py
Install dependencies first: pip install -r requirements.txt
The first run downloads two small MediaPipe Tasks model files into models/.
Press q in the OpenCV window to quit.
"""

from collections import deque
from pathlib import Path
from typing import Iterable, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# These values are intentionally easy to tune for different cameras and lighting.
CAMERA_INDEX = 0
REQUESTED_WIDTH = 640
REQUESTED_HEIGHT = 480
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_FACES = 1
MAX_NUM_HANDS = 2
OPACITY_HISTORY_LENGTH = 5

# Pinch distances are fractions of the actual frame width. Recalibrate these
# fractions if a different camera, distance, or lighting setup needs it.
PINCH_MIN_FRACTION = 0.035
PINCH_MAX_FRACTION = 0.24

MODEL_DIR = Path(__file__).with_name("models")
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# OpenCV uses BGR tuples. The fist selection is a bright, neutral "off" mode.
NEON_PALETTE = {
    0: (255, 255, 255),  # white
    1: (255, 255, 0),    # cyan
    2: (255, 0, 255),    # magenta
    3: (0, 255, 0),      # green
    4: (0, 255, 255),    # yellow
    5: (255, 0, 128),    # purple
}
DEFAULT_COLOR = NEON_PALETTE[1]
DEFAULT_OPACITY = 1.0

# A curated subset keeps the constellation readable while showing key contours.
FACE_CONNECTIONS = {
    (connection.start, connection.end)
    for connection in (
        list(vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS)
        + list(vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS)
        + list(vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS)
    )
}


def ensure_model(path: Path, url: str) -> Path:
    """Download a Tasks model once, keeping runtime setup self-contained."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {path.name}...")
        urlretrieve(url, path)
    return path


def normalized_to_pixel(landmark, width: int, height: int) -> Tuple[int, int]:
    """Convert a MediaPipe normalized landmark to a clipped panel coordinate."""
    x = int(np.clip(landmark.x * width, 0, width - 1))
    y = int(np.clip(landmark.y * height, 0, height - 1))
    return x, y


def count_extended_fingers(hand_landmarks: Iterable, handedness: str) -> int:
    """Count raised fingers using fingertip-to-joint geometry.

    Four fingers use vertical tip/PIP comparisons. The thumb uses a horizontal
    comparison whose direction follows the handedness label on mirrored input.
    """
    points = list(hand_landmarks)
    count = 0

    # Index, middle, ring, and pinky: a raised fingertip is above its PIP joint.
    for tip_index, pip_index in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if points[tip_index].y < points[pip_index].y:
            count += 1

    if handedness.lower() == "right":
        if points[4].x < points[3].x:
            count += 1
    elif points[4].x > points[3].x:
        count += 1
    return count


def pinch_opacity(hand_landmarks: Iterable, frame_width: int, frame_height: int) -> float:
    """Map thumb/index separation in actual pixels to a clamped opacity."""
    points = list(hand_landmarks)
    thumb, index = points[4], points[8]
    distance_pixels = float(
        np.hypot((thumb.x - index.x) * frame_width, (thumb.y - index.y) * frame_height)
    )
    min_pixels = PINCH_MIN_FRACTION * frame_width
    max_pixels = PINCH_MAX_FRACTION * frame_width
    normalized = (distance_pixels - min_pixels) / (max_pixels - min_pixels)
    return float(np.clip(0.1 + normalized * 0.9, 0.1, 1.0))


def draw_neon_face(
    landmarks: Iterable,
    panel_width: int,
    panel_height: int,
    color: Tuple[int, int, int],
    opacity: float,
) -> np.ndarray:
    """Render the face as one glow layer plus a crisp landmark layer."""
    glow = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    crisp = np.zeros_like(glow)
    landmarks = list(landmarks)
    points = [normalized_to_pixel(point, panel_width, panel_height) for point in landmarks]

    # Soft, blurred geometry underneath bright geometry creates the neon effect.
    for start, end in FACE_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(glow, points[start], points[end], color, 2, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(glow, (x, y), 5, color, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=3.0)

    for start, end in FACE_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(crisp, points[start], points[end], color, 1, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(crisp, (x, y), 1, color, -1, cv2.LINE_AA)

    rendered = cv2.addWeighted(glow, 0.85, crisp, 1.0, 0)
    return cv2.addWeighted(np.zeros_like(rendered), 1.0 - opacity, rendered, opacity, 0)


def put_status(canvas: np.ndarray, color: Tuple[int, int, int], opacity: float) -> None:
    """Add small, unobtrusive control feedback to the replica panel."""
    text = f"opacity {opacity:.0%}  |  color BGR {color}"
    cv2.putText(canvas, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)


def main() -> None:
    face_model = ensure_model(MODEL_DIR / "face_landmarker.task", FACE_MODEL_URL)
    hand_model = ensure_model(MODEL_DIR / "hand_landmarker.task", HAND_MODEL_URL)

    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(face_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=MAX_NUM_FACES,
        min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAMERA_INDEX and camera permissions.")

    selected_color = DEFAULT_COLOR
    opacity = DEFAULT_OPACITY
    opacity_history: deque[float] = deque(maxlen=OPACITY_HISTORY_LENGTH)
    timestamp_ms = 0

    with vision.FaceLandmarker.create_from_options(face_options) as face_landmarker, vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                mirrored = cv2.flip(frame, 1)
                frame_height, frame_width = mirrored.shape[:2]
                rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33
                face_result = face_landmarker.detect_for_video(image, timestamp_ms)
                hand_result = hand_landmarker.detect_for_video(image, timestamp_ms)

                # No hands means controls retain their last known values.
                for hand_landmarks, handedness_list in zip(hand_result.hand_landmarks, hand_result.handedness):
                    label = handedness_list[0].category_name
                    if label.lower() == "right":
                        fingers = count_extended_fingers(hand_landmarks, label)
                        selected_color = NEON_PALETTE[fingers]
                    elif label.lower() == "left":
                        opacity_history.append(pinch_opacity(hand_landmarks, frame_width, frame_height))
                        opacity = float(np.mean(opacity_history))

                replica = np.zeros_like(mirrored)
                if face_result.face_landmarks:
                    replica = draw_neon_face(
                        face_result.face_landmarks[0], frame_width, frame_height, selected_color, opacity
                    )
                put_status(replica, selected_color, opacity)
                combined = np.hstack((mirrored, replica))
                cv2.imshow("Neon Dot-Face Tracker", combined)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
