"""Neon Dot-Face Tracker

Run with: python main.py
Install dependencies first: pip install -r requirements.txt
The first run downloads two small MediaPipe Tasks model files into models/.
Press q in the OpenCV window to quit.
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple
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
FACE_SMOOTHING_ALPHA = 0.35  # Lower = smoother; higher = more responsive.
SWAP_HANDEDNESS = False  # Set True if the camera labels your physical hands backward.
MAX_NUM_FACES = 1
MAX_NUM_HANDS = 2
OPACITY_HISTORY_LENGTH = 5

# Pinch distances are fractions of the actual frame width. Recalibrate these
# fractions if a different camera, distance, or lighting setup needs it.
PINCH_MIN_FRACTION = 0.035
PINCH_MAX_FRACTION = 0.24
FINGER_Y_MARGIN = 0.025  # Ignore tiny vertical landmark jitter when counting.
THUMB_X_MARGIN = 0.025
THUMB_REACH_RATIO = 1.08

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

RIGHT_GESTURE_NAMES = {
    0: "fist",
    1: "point",
    2: "peace",
    3: "three",
    4: "four",
    5: "open_palm",
}

# The full tessellation adds structure; the contour subset stays brighter so the
# mesh remains readable instead of becoming a wall of equally bright lines.
FACE_TESSELLATION = {
    (connection.start, connection.end)
    for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
}
FACE_CONNECTIONS = {
    (connection.start, connection.end)
    for connection in (
        list(vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS)
        + list(vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS)
        + list(vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS)
    )
}
FEATURE_INDICES = {0, 13, 14, 17, 61, 291, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477}


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
    wrist = points[0]

    # A finger must clear its PIP joint by a margin and extend farther from the
    # wrist than that joint. Using both tests avoids counting curled fingers when
    # the hand is tilted or the webcam landmarks jitter by a few pixels.
    for tip_index, pip_index in ((8, 6), (12, 10), (16, 14), (20, 18)):
        tip = points[tip_index]
        pip = points[pip_index]
        tip_reaches_up = tip.y < pip.y - FINGER_Y_MARGIN
        tip_reaches_out = np.hypot(tip.x - wrist.x, tip.y - wrist.y) > (
            np.hypot(pip.x - wrist.x, pip.y - wrist.y) * 1.05
        )
        if tip_reaches_up and tip_reaches_out:
            count += 1

    # Compare the thumb tip with both its IP joint and wrist. The direction
    # check handles left/right hands; the reach check rejects a curled thumb
    # that happens to sit slightly outside the hand silhouette.
    thumb_tip, thumb_ip = points[4], points[3]
    thumb_reaches_out = np.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y) > (
        np.hypot(thumb_ip.x - wrist.x, thumb_ip.y - wrist.y) * THUMB_REACH_RATIO
    )
    if handedness.lower() == "right":
        thumb_points_out = thumb_tip.x < thumb_ip.x - THUMB_X_MARGIN
    else:
        thumb_points_out = thumb_tip.x > thumb_ip.x + THUMB_X_MARGIN
    if thumb_points_out and thumb_reaches_out:
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


@dataclass
class GestureState:
    """Current controls plus named gestures that other features can consume."""

    color: Tuple[int, int, int] = DEFAULT_COLOR
    opacity: float = DEFAULT_OPACITY
    right_fingers: int = 1
    right_gesture: str = "point"
    left_gesture: str = "unknown"
    boost: bool = False
    scan_pulse: float = 0.0
    hand_debug: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)


@dataclass
class SmoothPoint:
    """Small landmark-compatible point used by the temporal face filter."""

    x: float
    y: float
    z: float = 0.0


class LandmarkSmoother:
    """Apply an exponential moving average to all face landmarks."""

    def __init__(self, alpha: float = FACE_SMOOTHING_ALPHA) -> None:
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        """Discard stale coordinates after a face leaves the camera view."""
        self._previous = None

    def update(self, landmarks: Iterable) -> List[SmoothPoint]:
        """Return smoothed x/y/z landmarks while preserving the face shape."""
        current = np.array([(point.x, point.y, getattr(point, "z", 0.0)) for point in landmarks], dtype=np.float32)
        if self._previous is None or self._previous.shape != current.shape:
            self._previous = current
        else:
            self._previous += self.alpha * (current - self._previous)
        return [SmoothPoint(float(x), float(y), float(z)) for x, y, z in self._previous]


class GestureController:
    """Translate raw Tasks landmarks into stable controls and extensible events.

    Register a callback for any event name, for example:
        controller.on("right_open_palm", lambda state: trigger_particles())
    Add new behavior in this class without changing the webcam/render loop.
    """

    def __init__(self) -> None:
        self.state = GestureState()
        self._opacity_history: deque[float] = deque(maxlen=OPACITY_HISTORY_LENGTH)
        self._last_labels: Dict[str, str] = {}
        self._handlers: Dict[str, List[Callable[[GestureState], None]]] = {}

    def on(self, event_name: str, handler: Callable[[GestureState], None]) -> None:
        """Register a function called once when a named gesture changes into view."""
        self._handlers.setdefault(event_name, []).append(handler)

    def _emit_transition(self, hand: str, label: str) -> None:
        event_name = f"{hand}_{label}"
        if self._last_labels.get(hand) == label:
            return
        self._last_labels[hand] = label
        self.state.events.append(event_name)
        for handler in self._handlers.get(event_name, []):
            handler(self.state)

    def update(self, hand_landmarks_list: Iterable, handedness_list: Iterable, width: int, height: int) -> GestureState:
        """Update controls from one frame of Tasks hand results.

        Missing hands intentionally do not reset state, so controls remain stable.
        """
        self.state.events = []
        self.state.hand_debug = []
        self.state.scan_pulse *= 0.90
        visible_gestures = {}
        for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
            raw_label = handedness[0].category_name.lower()
            label = ("left" if raw_label == "right" else "right") if SWAP_HANDEDNESS else raw_label
            detected_fingers = count_extended_fingers(hand_landmarks, label)
            self.state.hand_debug.append(f"raw:{raw_label} -> {label}  fingers:{detected_fingers}")
            if label == "right":
                fingers = detected_fingers
                gesture = RIGHT_GESTURE_NAMES[fingers]
                points = list(hand_landmarks)
                # A thumb pointing upward with all other fingers folded is a
                # semantic thumbs-up, while still retaining the cyan color slot.
                other_fingers_folded = all(points[tip].y > points[pip].y for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))
                if fingers == 1 and points[4].y < points[0].y and other_fingers_folded:
                    gesture = "thumbs_up"
                self.state.right_fingers = fingers
                self.state.right_gesture = gesture
                self.state.color = NEON_PALETTE[fingers]
                self.state.boost = gesture == "thumbs_up"
                visible_gestures[label] = gesture
                self._emit_transition("right", gesture)
            elif label == "left":
                current_opacity = pinch_opacity(hand_landmarks, width, height)
                self._opacity_history.append(current_opacity)
                self.state.opacity = float(np.mean(self._opacity_history))
                gesture = "pinch" if self.state.opacity < 0.28 else "open_hand"
                if gesture == "open_hand" and self._last_labels.get("left") == "pinch":
                    self._opacity_history.clear()
                    self.state.opacity = 1.0
                self.state.left_gesture = gesture
                visible_gestures[label] = gesture
                self._emit_transition("left", gesture)
        if visible_gestures.get("right") == "open_palm" and visible_gestures.get("left") == "open_hand":
            self.state.scan_pulse = 1.0
            self._emit_transition("both", "open")
        else:
            self._last_labels.pop("both", None)
        return self.state


def draw_neon_face(
    landmarks: Iterable,
    panel_width: int,
    panel_height: int,
    color: Tuple[int, int, int],
    opacity: float,
) -> np.ndarray:
    """Render the face as one glow layer plus a crisp landmark layer."""
    glow = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    mesh = np.zeros_like(glow)
    crisp = np.zeros_like(glow)
    landmarks = list(landmarks)
    points = [normalized_to_pixel(point, panel_width, panel_height) for point in landmarks]
    # Keep structure independent from hue: a restrained blue-gray wireframe
    # remains readable even when the selected neon color is very saturated.
    mesh_color = (42, 68, 74)
    # Avoid mixing every color with white; that was making some hues bloom and
    # obscure the face. The selected color now controls a crisp, modest accent.
    accent_color = tuple(min(230, max(28, int(channel * 0.82))) for channel in color)

    # A dim full wireframe provides a futuristic 3D feel without overpowering
    # the brighter face outline and expression landmarks.
    for start, end in FACE_TESSELLATION:
        if start < len(points) and end < len(points):
            cv2.line(mesh, points[start], points[end], mesh_color, 1, cv2.LINE_AA)

    # Soft, blurred geometry underneath bright geometry creates the neon effect.
    for start, end in FACE_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(glow, points[start], points[end], accent_color, 2, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(glow, (x, y), 5, accent_color, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=3.0)

    for start, end in FACE_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(crisp, points[start], points[end], accent_color, 1, cv2.LINE_AA)
    for index, (x, y) in enumerate(points):
        cv2.circle(crisp, (x, y), 2 if index in FEATURE_INDICES else 1, accent_color, -1, cv2.LINE_AA)

    # Layering is deliberately light: mesh first, then glow, then crisp details.
    rendered = cv2.addWeighted(mesh, 1.15, glow, 0.55, 0)
    rendered = cv2.addWeighted(rendered, 1.0, crisp, 1.0, 0)

    return cv2.addWeighted(np.zeros_like(rendered), 1.0 - opacity, rendered, opacity, 0)


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

    controller = GestureController()
    face_smoother = LandmarkSmoother()
    # Future features can subscribe without modifying the tracking loop:
    # controller.on("right_open_palm", lambda state: start_particle_pulse())
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
                state = controller.update(
                    hand_result.hand_landmarks,
                    hand_result.handedness,
                    frame_width,
                    frame_height,
                )

                replica = np.zeros_like(mirrored)
                if face_result.face_landmarks:
                    smooth_face = face_smoother.update(face_result.face_landmarks[0])
                    replica = draw_neon_face(
                        smooth_face, frame_width, frame_height, state.color, state.opacity
                    )
                else:
                    face_smoother.reset()
                combined = np.hstack((mirrored, replica))
                cv2.imshow("Neon Dot-Face Tracker", combined)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
