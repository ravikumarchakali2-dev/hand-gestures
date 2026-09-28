import os
import time
from collections import Counter, deque
from datetime import datetime

import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from config import (
    APP_NAME,
    GESTURE_HISTORY_SIZE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)

from core.camera import Camera
from core.gesture_detector import GestureDetector


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_NAME)
        self.resize(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        )

        self.camera = Camera()
        self.detector = GestureDetector()

        self.timer = QTimer(self)
        self.timer.timeout.connect(
            self.update_frame
        )

        self.running = False

        self.last_time = time.perf_counter()
        self.fps = 0.0

        self.history = deque(
            maxlen=GESTURE_HISTORY_SIZE
        )

        self.gesture_counter = Counter()

        self._build_ui()
        self._apply_styles()

    def _build_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(
            20,
            20,
            20,
            20,
        )
        root.setSpacing(15)

        # Header
        header = QHBoxLayout()

        title = QLabel(
            "✋ HandGesture Vision"
        )
        title.setObjectName("title")

        subtitle = QLabel(
            "Real-time Computer Vision Gesture Recognition"
        )
        subtitle.setObjectName("subtitle")

        header.addWidget(title)
        header.addStretch()
        header.addWidget(subtitle)

        root.addLayout(header)

        # Main area
        main_layout = QHBoxLayout()
        main_layout.setSpacing(15)

        # Camera section
        camera_card = QFrame()
        camera_card.setObjectName("card")

        camera_layout = QVBoxLayout(
            camera_card
        )

        camera_header = QHBoxLayout()

        camera_title = QLabel(
            "LIVE CAMERA"
        )
        camera_title.setObjectName(
            "sectionTitle"
        )

        self.camera_status = QLabel(
            "● CAMERA OFF"
        )
        self.camera_status.setObjectName(
            "statusOff"
        )

        camera_header.addWidget(
            camera_title
        )
        camera_header.addStretch()
        camera_header.addWidget(
            self.camera_status
        )

        camera_layout.addLayout(
            camera_header
        )

        self.video_label = QLabel()
        self.video_label.setAlignment(
            Qt.AlignCenter
        )
        self.video_label.setMinimumSize(
            750,
            500,
        )
        self.video_label.setObjectName(
            "video"
        )

        camera_layout.addWidget(
            self.video_label
        )

        # Buttons
        controls = QHBoxLayout()

        self.start_button = QPushButton(
            "▶  Start Camera"
        )
        self.start_button.clicked.connect(
            self.start_camera
        )

        self.stop_button = QPushButton(
            "■  Stop Camera"
        )
        self.stop_button.clicked.connect(
            self.stop_camera
        )
        self.stop_button.setEnabled(False)

        self.capture_button = QPushButton(
            "📷  Screenshot"
        )
        self.capture_button.clicked.connect(
            self.capture_frame
        )
        self.capture_button.setEnabled(False)

        controls.addWidget(
            self.start_button
        )
        controls.addWidget(
            self.stop_button
        )
        controls.addWidget(
            self.capture_button
        )

        camera_layout.addLayout(
            controls
        )

        main_layout.addWidget(
            camera_card,
            3,
        )

        # Right panel
        side_card = QFrame()
        side_card.setObjectName("card")

        side_layout = QVBoxLayout(
            side_card
        )

        panel_title = QLabel(
            "DETECTION"
        )
        panel_title.setObjectName(
            "sectionTitle"
        )

        side_layout.addWidget(
            panel_title
        )

        # Main gesture
        self.gesture_label = QLabel(
            "No Gesture"
        )
        self.gesture_label.setObjectName(
            "gesture"
        )
        self.gesture_label.setAlignment(
            Qt.AlignCenter
        )

        side_layout.addWidget(
            self.gesture_label
        )

        self.confidence_label = QLabel(
            "Confidence: --"
        )
        self.confidence_label.setAlignment(
            Qt.AlignCenter
        )

        side_layout.addWidget(
            self.confidence_label
        )

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(
            0,
            100,
        )
        self.confidence_bar.setValue(0)

        side_layout.addWidget(
            self.confidence_bar
        )

        # Info grid
        grid = QGridLayout()

        self.fps_value = self._create_metric(
            "FPS"
        )
        self.hands_value = self._create_metric(
            "HANDS"
        )
        self.hand_value = self._create_metric(
            "SIDE"
        )

        grid.addWidget(
            QLabel("FPS"),
            0,
            0,
        )
        grid.addWidget(
            self.fps_value,
            0,
            1,
        )

        grid.addWidget(
            QLabel("Hands"),
            1,
            0,
        )
        grid.addWidget(
            self.hands_value,
            1,
            1,
        )

        grid.addWidget(
            QLabel("Hand"),
            2,
            0,
        )
        grid.addWidget(
            self.hand_value,
            2,
            1,
        )

        side_layout.addLayout(
            grid
        )

        # History
        history_title = QLabel(
            "RECENT GESTURES"
        )
        history_title.setObjectName(
            "sectionTitle"
        )

        side_layout.addWidget(
            history_title
        )

        self.history_label = QLabel(
            "No detections yet"
        )
        self.history_label.setWordWrap(True)

        side_layout.addWidget(
            self.history_label
        )

        side_layout.addStretch()

        # Footer
        self.engine_label = QLabel(
            "Engine: OpenCV + MediaPipe"
        )
        self.engine_label.setObjectName(
            "footer"
        )

        side_layout.addWidget(
            self.engine_label
        )

        main_layout.addWidget(
            side_card,
            1,
        )

        root.addLayout(
            main_layout
        )

    def _create_metric(self, text):
        label = QLabel("--")
        label.setObjectName("metric")
        label.setAlignment(
            Qt.AlignRight
        )
        return label

    def _apply_styles(self):

        self.setStyleSheet(
            """
            QMainWindow {
                background: #0b1120;
            }

            QWidget {
                color: #e5e7eb;
                font-family: "Segoe UI";
                font-size: 14px;
            }

            #title {
                font-size: 28px;
                font-weight: 700;
                color: #ffffff;
            }

            #subtitle {
                color: #94a3b8;
                font-size: 14px;
            }

            #card {
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 16px;
            }

            #sectionTitle {
                font-size: 13px;
                font-weight: 700;
                color: #94a3b8;
                letter-spacing: 1px;
            }

            #video {
                background: #020617;
                border-radius: 12px;
                border: 1px solid #334155;
            }

            QPushButton {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 9px;
                padding: 10px 16px;
                color: #f8fafc;
                font-weight: 600;
            }

            QPushButton:hover {
                background: #334155;
            }

            QPushButton:disabled {
                color: #64748b;
                background: #0f172a;
            }

            #gesture {
                font-size: 30px;
                font-weight: 700;
                color: #60a5fa;
                padding: 20px;
            }

            #metric {
                font-size: 18px;
                font-weight: 700;
                color: #f8fafc;
            }

            QProgressBar {
                background: #0f172a;
                border: none;
                border-radius: 5px;
                height: 10px;
            }

            QProgressBar::chunk {
                background: #3b82f6;
                border-radius: 5px;
            }

            #statusOff {
                color: #f87171;
                font-weight: 700;
            }

            #footer {
                color: #64748b;
                font-size: 12px;
            }
            """
        )

    def start_camera(self):

        if not self.camera.start():
            self.camera_status.setText(
                "● CAMERA ERROR"
            )
            return

        self.running = True

        self.camera_status.setText(
            "● CAMERA LIVE"
        )

        self.camera_status.setObjectName(
            "statusOn"
        )

        self.camera_status.setStyleSheet(
            "color: #34d399; font-weight: 700;"
        )

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.capture_button.setEnabled(True)

        self.timer.start(30)

    def stop_camera(self):

        self.timer.stop()
        self.camera.stop()

        self.running = False

        self.camera_status.setText(
            "● CAMERA OFF"
        )

        self.camera_status.setStyleSheet(
            "color: #f87171; font-weight: 700;"
        )

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.capture_button.setEnabled(False)

        self.video_label.clear()

        self.gesture_label.setText(
            "No Gesture"
        )

        self.confidence_label.setText(
            "Confidence: --"
        )

        self.confidence_bar.setValue(0)

        self.hands_value.setText("--")
        self.hand_value.setText("--")

    def update_frame(self):

        frame = self.camera.read()

        if frame is None:
            self.stop_camera()
            return

        # Mirror camera for natural interaction.
        frame = cv2.flip(
            frame,
            1,
        )

        processed_frame, detections = (
            self.detector.process(frame)
        )

        self._update_fps()

        self.fps_value.setText(
            f"{self.fps:.1f}"
        )

        self.hands_value.setText(
            str(len(detections))
        )

        if detections:

            primary = detections[0]

            self.gesture_label.setText(
                primary.gesture
            )

            confidence = int(
                primary.confidence * 100
            )

            self.confidence_label.setText(
                f"Confidence: {confidence}%"
            )

            self.confidence_bar.setValue(
                confidence
            )

            self.hand_value.setText(
                primary.handedness
            )

            if primary.gesture != "Unknown":

                self.history.appendleft(
                    primary.gesture
                )

                self.gesture_counter[
                    primary.gesture
                ] += 1

                self._update_history()

        else:

            self.gesture_label.setText(
                "No Hand Detected"
            )

            self.confidence_label.setText(
                "Confidence: --"
            )

            self.confidence_bar.setValue(0)

            self.hands_value.setText("0")
            self.hand_value.setText("--")

        self._display_frame(
            processed_frame
        )

    def _update_fps(self):

        current = time.perf_counter()

        elapsed = current - self.last_time

        if elapsed > 0:
            instant_fps = 1.0 / elapsed

            # Smooth FPS
            self.fps = (
                self.fps * 0.9
                + instant_fps * 0.1
            )

        self.last_time = current

    def _update_history(self):

        counts = Counter(
            self.history
        )

        recent = list(
            self.history
        )[:8]

        lines = []

        for gesture in recent:

            lines.append(
                f"• {gesture} "
                f"({counts[gesture]})"
            )

        self.history_label.setText(
            "\n".join(lines)
        )

    def _display_frame(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        h, w, channels = rgb.shape

        bytes_per_line = (
            channels * w
        )

        image = QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(
            image
        )

        pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.video_label.setPixmap(
            pixmap
        )

        self.last_frame = frame.copy()

    def capture_frame(self):

        if not hasattr(
            self,
            "last_frame"
        ):
            return

        directory = os.path.join(
            "assets",
            "screenshots",
        )

        os.makedirs(
            directory,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        path = os.path.join(
            directory,
            f"gesture_{timestamp}.jpg",
        )

        cv2.imwrite(
            path,
            self.last_frame,
        )

        self.capture_button.setText(
            "✓ Saved"
        )

        QTimer.singleShot(
            1200,
            lambda: self.capture_button.setText(
                "📷  Screenshot"
            ),
        )

    def closeEvent(self, event):

        self.timer.stop()
        self.camera.stop()
        self.detector.close()

        event.accept()