from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QSizePolicy,
    QSplitter,
    QHBoxLayout,
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np


class CameraView(QWidget):
    """Widget for displaying camera feed with detection overlays"""

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Create main layout
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Create splitter for side-by-side view
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Original image label (left side)
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(320, 240)
        self.original_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.original_label.setObjectName("originalView")

        # Predicted image label (right side)
        self.predicted_label = QLabel()
        self.predicted_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.predicted_label.setMinimumSize(320, 240)
        self.predicted_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.predicted_label.setObjectName("predictedView")

        # Add labels to splitter
        self.splitter.addWidget(self.original_label)
        self.splitter.addWidget(self.predicted_label)

        # Set equal initial sizes
        self.splitter.setSizes([1, 1])

        # Add splitter to layout
        self.layout.addWidget(self.splitter)

        # Create single view label for camera mode
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.camera_label.setObjectName("cameraView")

        # Stack both views
        self.camera_label.hide()  # Hide initially
        self.layout.addWidget(self.camera_label)

        # Track current mode
        self.current_mode = "split"  # or "camera"

        # Initialize state
        self.clear()
        self.last_frame = None  # Add this attribute
        self.last_original = None
        self.last_predicted = None

    def set_mode(self, mode: str):
        """Switch between camera and split view modes"""
        self.current_mode = mode
        if mode == "camera":
            self.splitter.hide()
            self.camera_label.show()
        else:  # split mode
            self.camera_label.hide()
            self.splitter.show()

    def update_frame(self, frame: np.ndarray):
        """Update frame based on current mode"""
        if frame is None:
            return

        self.last_frame = frame.copy()  # Store the frame

        if self.current_mode == "camera":
            self._update_label(self.camera_label, frame)
        else:
            self.last_predicted = frame
            self._update_label(self.predicted_label, frame)

    def show_original_image(self, frame: np.ndarray):
        """Display the original image in split mode"""
        if self.current_mode != "camera" and frame is not None:
            self.last_original = frame.copy()
            self._update_label(self.original_label, frame)
            self.original_label.setText("")
            self.predicted_label.setText("Detection results will appear here")

    def _update_label(self, label: QLabel, frame: np.ndarray):
        """Update a label with a frame"""
        if frame is None:
            return

        height, width, channels = frame.shape
        bytes_per_line = channels * width
        q_image = QImage(
            frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)

        # Scale pixmap to fit label while maintaining aspect ratio
        pixmap = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        label.setPixmap(pixmap)

    def resizeEvent(self, event):
        """Handle resize for both modes"""
        super().resizeEvent(event)
        if self.current_mode == "camera":
            if hasattr(self, "last_frame"):
                self._update_label(self.camera_label, self.last_frame)
        else:
            if self.last_original is not None:
                self._update_label(self.original_label, self.last_original)
            if self.last_predicted is not None:
                self._update_label(self.predicted_label, self.last_predicted)

    def clear(self):
        """Clear view based on current mode"""
        self.last_frame = None  # Clear the frame
        self.last_original = None
        self.last_predicted = None

        empty_pixmap = QPixmap(self.camera_label.size())
        empty_pixmap.fill(Qt.GlobalColor.transparent)

        if self.current_mode == "camera":
            self.camera_label.setPixmap(empty_pixmap)
            self.camera_label.setText("No Camera Feed")
        else:
            self.original_label.setPixmap(empty_pixmap)
            self.predicted_label.setPixmap(empty_pixmap)
            self.original_label.setText("Original Image")
            self.predicted_label.setText("Detection Results")
