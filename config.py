APP_NAME = "HandGesture Vision"

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 30

MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

GESTURE_HISTORY_SIZE = 20

SUPPORTED_GESTURES = [
    "Open Palm",
    "Fist",
    "Thumbs Up",
    "Thumbs Down",
    "Victory",
    "Pointing",
    "I Love You",
    "Three",
    "Four",
    "Unknown",
]