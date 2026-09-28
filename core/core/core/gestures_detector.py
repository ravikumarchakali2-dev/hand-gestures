from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from config import (
    MAX_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)


@dataclass
class GestureResult:
    gesture: str
    confidence: float
    handedness: str


class GestureDetector:
    """
    Real-time hand landmark and gesture detector.

    Uses MediaPipe Hands for landmark detection and
    geometric rules for gesture classification.
    """

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            model_complexity=1,
        )

    @staticmethod
    def _distance(a, b):
        return np.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2 +
            (a.z - b.z) ** 2
        )

    @staticmethod
    def _finger_extended(landmarks, tip, pip, mcp):
        """
        Determines whether a finger is extended.

        For index/middle/ring/pinky, the fingertip should be
        farther from the wrist than the PIP/MCP region.
        """

        wrist = landmarks[0]

        tip_point = landmarks[tip]
        pip_point = landmarks[pip]
        mcp_point = landmarks[mcp]

        tip_distance = GestureDetector._distance(
            wrist,
            tip_point
        )

        pip_distance = GestureDetector._distance(
            wrist,
            pip_point
        )

        mcp_distance = GestureDetector._distance(
            wrist,
            mcp_point
        )

        return (
            tip_distance > pip_distance * 1.08
            and tip_distance > mcp_distance * 1.15
        )

    @staticmethod
    def _thumb_extended(landmarks):
        """
        Thumb detection based on distance from wrist and MCP.
        """

        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]

        tip_distance = GestureDetector._distance(
            wrist,
            thumb_tip
        )

        mcp_distance = GestureDetector._distance(
            wrist,
            thumb_mcp
        )

        ip_distance = GestureDetector._distance(
            wrist,
            thumb_ip
        )

        return (
            tip_distance > mcp_distance * 1.25
            and tip_distance > ip_distance * 1.08
        )

    def _get_finger_states(self, landmarks):
        thumb = self._thumb_extended(landmarks)

        index = self._finger_extended(
            landmarks,
            8,
            6,
            5,
        )

        middle = self._finger_extended(
            landmarks,
            12,
            10,
            9,
        )

        ring = self._finger_extended(
            landmarks,
            16,
            14,
            13,
        )

        pinky = self._finger_extended(
            landmarks,
            20,
            18,
            17,
        )

        return {
            "thumb": thumb,
            "index": index,
            "middle": middle,
            "ring": ring,
            "pinky": pinky,
        }

    def _classify_gesture(self, states, landmarks):
        thumb = states["thumb"]
        index = states["index"]
        middle = states["middle"]
        ring = states["ring"]
        pinky = states["pinky"]

        fingers = [
            thumb,
            index,
            middle,
            ring,
            pinky,
        ]

        count = sum(fingers)

        # Five fingers
        if count == 5:
            return "Open Palm", 0.97

        # No fingers
        if count == 0:
            return "Fist", 0.96

        # Index + middle
        if (
            not thumb
            and index
            and middle
            and not ring
            and not pinky
        ):
            return "Victory", 0.95

        # Index only
        if (
            not thumb
            and index
            and not middle
            and not ring
            and not pinky
        ):
            return "Pointing", 0.93

        # Thumb only
        if (
            thumb
            and not index
            and not middle
            and not ring
            and not pinky
        ):
            wrist = landmarks[0]
            thumb_tip = landmarks[4]

            if thumb_tip.y < wrist.y:
                return "Thumbs Up", 0.94

            return "Thumbs Down", 0.94

        # Thumb + index + pinky
        if (
            thumb
            and index
            and not middle
            and not ring
            and pinky
        ):
            return "I Love You", 0.91

        # Three fingers
        if (
            not thumb
            and index
            and middle
            and ring
            and not pinky
        ):
            return "Three", 0.90

        # Four fingers
        if (
            not thumb
            and index
            and middle
            and ring
            and pinky
        ):
            return "Four", 0.90

        return "Unknown", 0.50

    def process(self, frame):
        """
        Process one BGR OpenCV frame.

        Returns:
            annotated_frame,
            list[GestureResult]
        """

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        results = self.hands.process(rgb)

        rgb.flags.writeable = True

        annotated = frame.copy()

        gesture_results: List[GestureResult] = []

        if not results.multi_hand_landmarks:
            return annotated, gesture_results

        for index, hand_landmarks in enumerate(
            results.multi_hand_landmarks
        ):

            self.mp_draw.draw_landmarks(
                annotated,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )

            handedness = "Unknown"

            if results.multi_handedness:
                handedness = (
                    results
                    .multi_handedness[index]
                    .classification[0]
                    .label
                )

            landmarks = hand_landmarks.landmark

            states = self._get_finger_states(
                landmarks
            )

            gesture, confidence = self._classify_gesture(
                states,
                landmarks,
            )

            gesture_results.append(
                GestureResult(
                    gesture=gesture,
                    confidence=confidence,
                    handedness=handedness,
                )
            )

            self._draw_gesture_label(
                annotated,
                hand_landmarks,
                gesture,
                confidence,
                handedness,
            )

        return annotated, gesture_results

    def _draw_gesture_label(
        self,
        frame,
        hand_landmarks,
        gesture,
        confidence,
        handedness,
    ):
        h, w, _ = frame.shape

        x_values = [
            int(point.x * w)
            for point in hand_landmarks.landmark
        ]

        y_values = [
            int(point.y * h)
            for point in hand_landmarks.landmark
        ]

        x = max(min(x_values), 10)
        y = max(min(y_values) - 20, 30)

        text = (
            f"{gesture}  "
            f"{confidence * 100:.0f}%  "
            f"{handedness}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        text_size = cv2.getTextSize(
            text,
            font,
            0.7,
            2,
        )[0]

        padding = 10

        cv2.rectangle(
            frame,
            (
                x - padding,
                y - text_size[1] - padding,
            ),
            (
                x + text_size[0] + padding,
                y + padding,
            ),
            (25, 25, 25),
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def close(self):
        self.hands.close()