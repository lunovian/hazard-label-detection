from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np


class CameraView(QWidget):
    """Widget for displaying camera feed with detection overlays"""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Image display label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #1e1e1e;")

        # Set an initial "No camera" message
        no_camera_pixmap = QPixmap(self.image_label.size())
        no_camera_pixmap.fill(Qt.GlobalColor.transparent)
        self.image_label.setPixmap(no_camera_pixmap)
        self.image_label.setText("No Camera Feed")
        self.image_label.setStyleSheet(
            "QLabel { color: white; background-color: #1e1e1e; font-size: 18px; }"
        )

        self.layout.addWidget(self.image_label)

        # Last frame cache
        self.last_frame = None

    def update_frame(self, frame: np.ndarray):
        """Update displayed frame with new camera frame"""
        if frame is None:
            return

        self.last_frame = frame

        # Convert numpy ndarray to QImage
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        q_image = QImage(
            frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )

        # Convert QImage to QPixmap for display
        pixmap = QPixmap.fromImage(q_image)

        # Scale pixmap to fit label while maintaining aspect ratio
        pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Update the label with the new pixmap
        self.image_label.setPixmap(pixmap)
        self.image_label.setText("")  # Clear any text

    def resizeEvent(self, event):
        """Handle widget resize event"""
        super().resizeEvent(event)
        # If we have a cached frame, update it to fit the new size
        if self.last_frame is not None:
            self.update_frame(self.last_frame)

    def clear(self):
        """Clear the camera view"""
        self.last_frame = None
        self.image_label.clear()
        self.image_label.setText("No Camera Feed")
