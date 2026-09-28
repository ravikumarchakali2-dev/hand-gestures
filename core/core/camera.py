import cv2

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)


class Camera:
    def __init__(self, camera_index=CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None
        self.running = False

    def start(self):
        if self.running:
            return True

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.cap = None
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # Try to reduce camera buffering.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        return True

    def read(self):
        if not self.running or self.cap is None:
            return None

        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def stop(self):
        self.running = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_running(self):
        return self.running and self.cap is not None

    def __del__(self):
        self.stop()