import logging
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QSplitter,
    QSizePolicy,
    QComboBox,
    QFrame,
)
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtCore import Qt, pyqtSignal, QSize
import numpy as np
import cv2
import os


class UnifiedDisplayView(QWidget):
    """Unified display area for camera feed, images, and video"""

    view_toggled = pyqtSignal(bool)  # True for processed view, False for original
    capture_requested = pyqtSignal()  # Request to capture current frame
    save_requested = pyqtSignal()  # Request to save current view

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(5)

        # Create display toolbar
        self._create_toolbar()

        # Create main display container with flexible size
        self.display_container = QFrame()
        self.display_container.setFrameShape(QFrame.Shape.StyledPanel)
        self.display_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.container_layout = QVBoxLayout(self.display_container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)

        # Create camera display
        self.camera_display = QLabel()
        self.camera_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.camera_display.setStyleSheet("background-color: #1a1a1a; color: #cccccc;")
        self.camera_display.setText("No camera feed")

        # Create split view
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Create original display (left side)
        self.original_display = QLabel()
        self.original_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.original_display.setStyleSheet(
            "background-color: #1a1a1a; color: #cccccc;"
        )
        self.original_display.setText("Original Preview")

        # Create prediction display (right side)
        self.prediction_display = QLabel()
        self.prediction_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.prediction_display.setStyleSheet(
            "background-color: #1a1a1a; color: #cccccc;"
        )
        self.prediction_display.setText("Detection Preview")

        # Add displays to splitter
        self.splitter.addWidget(self.original_display)
        self.splitter.addWidget(self.prediction_display)

        # Add widgets to container
        self.container_layout.addWidget(self.camera_display)
        self.container_layout.addWidget(self.splitter)

        # Add container to main layout
        self.layout.addWidget(self.display_container)

        # Create status bar
        self.status_layout = QHBoxLayout()
        self.resolution_label = QLabel("Resolution: N/A")
        self.detection_count_label = QLabel("Detections: 0")
        self.status_layout.addWidget(self.resolution_label)
        self.status_layout.addStretch()
        self.status_layout.addWidget(self.detection_count_label)
        self.layout.addLayout(self.status_layout)

        # Initial setup
        self.current_mode = "camera"
        self.camera_display.show()
        self.splitter.hide()

        # Store frames
        self.original_frame = None
        self.processed_frame = None
        self.last_frame = None

        # Flag for whether we're showing original or processed in camera mode
        self.showing_processed = False

        # Create detached window
        self.detached_window = None

    def _create_toolbar(self):
        """Create toolbar with display controls"""
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(5, 0, 5, 0)

        # Display mode selector
        self.mode_selector = QComboBox()
        self.mode_selector.addItem("Single View", "camera")
        self.mode_selector.addItem("Split View", "split")
        self.mode_selector.currentIndexChanged.connect(self._on_mode_changed)
        toolbar.addWidget(QLabel("Display Mode:"))
        toolbar.addWidget(self.mode_selector)

        toolbar.addStretch(1)

        # Toggle button for original/processed view
        self.toggle_btn = QPushButton("Toggle View")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setToolTip("Switch between original and processed views")
        self.toggle_btn.clicked.connect(self._on_toggle)
        toolbar.addWidget(self.toggle_btn)

        # Capture button
        self.capture_btn = QPushButton("Capture Frame")
        self.capture_btn.setToolTip("Capture current frame")
        self.capture_btn.clicked.connect(lambda: self.capture_requested.emit())
        toolbar.addWidget(self.capture_btn)

        # Save button
        self.save_btn = QPushButton("Save")
        self.save_btn.setToolTip("Save current view")
        self.save_btn.clicked.connect(lambda: self.save_requested.emit())
        toolbar.addWidget(self.save_btn)

        # Detach button
        self.detach_btn = QPushButton("Detach")
        self.detach_btn.setCheckable(True)
        self.detach_btn.setToolTip("Open in separate window")
        self.detach_btn.clicked.connect(self._on_detach)
        toolbar.addWidget(self.detach_btn)

        self.layout.addLayout(toolbar)

    def set_mode(self, mode: str):
        """Set display mode (camera, split)"""
        self.current_mode = mode
        self.mode_selector.setCurrentIndex(self.mode_selector.findData(mode))
        self._update_display_mode()

    def _update_display_mode(self):
        """Update display based on current mode"""
        if self.current_mode == "camera":
            self.splitter.hide()
            self.camera_display.show()

            # Update with appropriate frame
            if self.showing_processed and self.processed_frame is not None:
                self._show_frame(self.camera_display, self.processed_frame)
            elif self.original_frame is not None:
                self._show_frame(self.camera_display, self.original_frame)
        else:
            self.camera_display.hide()
            self.splitter.show()

            # Update split view
            if self.original_frame is not None:
                self._show_frame(self.original_display, self.original_frame)
            if self.processed_frame is not None:
                self._show_frame(self.prediction_display, self.processed_frame)

    def update_frame(self, frame: np.ndarray, is_processed=False):
        """Update display with new frame"""
        if frame is None:
            return

        # Store frame copy
        frame_copy = frame.copy()

        # Update resolution status
        height, width = frame_copy.shape[:2]
        self.resolution_label.setText(f"Resolution: {width}x{height}")

        if is_processed:
            self.processed_frame = frame_copy
            # In split mode, always update prediction display
            if self.current_mode == "split":
                self._show_frame(self.prediction_display, frame_copy)
            # In camera mode, only update if showing processed
            elif self.showing_processed:
                self._show_frame(self.camera_display, frame_copy)
        else:
            self.original_frame = frame_copy
            # In split mode, always update original display
            if self.current_mode == "split":
                self._show_frame(self.original_display, frame_copy)
            # In camera mode, only update if showing original
            elif not self.showing_processed:
                self._show_frame(self.camera_display, frame_copy)

        # Store last frame for resize events
        self.last_frame = frame_copy

    def update_detection_count(self, count):
        """Update detection count in status bar"""
        self.detection_count_label.setText(f"Detections: {count}")

    def _show_frame(self, display: QLabel, frame: np.ndarray):
        """Display a frame with proper scaling and aspect ratio"""
        if frame is None:
            return

        # Ensure correct color format
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        # Get dimensions
        display_width = display.width()
        display_height = display.height()
        frame_height, frame_width = frame.shape[:2]

        # Skip if display has no size yet
        if display_width <= 1 or display_height <= 1:
            return

        # Calculate aspect ratios
        display_ratio = display_width / display_height
        frame_ratio = frame_width / frame_height

        # Calculate new dimensions to fit display while preserving aspect ratio
        if frame_ratio > display_ratio:
            # Frame is wider than display
            new_width = display_width
            new_height = int(new_width / frame_ratio)
        else:
            # Frame is taller than display
            new_height = display_height
            new_width = int(new_height * frame_ratio)

        # Resize frame
        resized_frame = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        # Convert to QImage
        height, width, channel = resized_frame.shape
        bytes_per_line = channel * width
        qimage = QImage(
            resized_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        # Create pixmap and set to display
        pixmap = QPixmap.fromImage(qimage)
        display.setPixmap(pixmap)

    def clear(self):
        """Clear all displays"""
        self.camera_display.clear()
        self.original_display.clear()
        self.prediction_display.clear()

        self.camera_display.setText("No camera feed")
        self.original_display.setText("Original Preview")
        self.prediction_display.setText("Detection Preview")

        self.original_frame = None
        self.processed_frame = None
        self.last_frame = None

        self.resolution_label.setText("Resolution: N/A")
        self.detection_count_label.setText("Detections: 0")

    def _on_toggle(self, checked):
        """Handle view toggle button click"""
        self.showing_processed = checked
        self.view_toggled.emit(checked)

        # Update display based on mode
        if self.current_mode == "camera":
            if checked and self.processed_frame is not None:
                self._show_frame(self.camera_display, self.processed_frame)
            elif not checked and self.original_frame is not None:
                self._show_frame(self.camera_display, self.original_frame)

    def _on_mode_changed(self, index):
        """Handle display mode change"""
        mode = self.mode_selector.currentData()
        if mode != self.current_mode:
            self.current_mode = mode
            self._update_display_mode()

    def _on_detach(self, checked):
        """Handle detaching/reattaching view"""
        if checked:
            from PyQt6.QtWidgets import QDialog

            # Create detached window
            self.detached_window = QDialog(None)  # No parent for independent window
            self.detached_window.setWindowTitle(
                "Hazard Label Detection - Detached View"
            )

            # Create layout for detached window
            detached_layout = QVBoxLayout(self.detached_window)
            detached_layout.setContentsMargins(0, 0, 0, 0)

            # Move appropriate widget to detached window
            if self.current_mode == "camera":
                detached_layout.addWidget(self.camera_display)
            else:
                detached_layout.addWidget(self.splitter)

            # Set window size
            self.detached_window.resize(1280, 720)

            # Handle window close
            self.detached_window.closeEvent = self._handle_detached_close
            self.detached_window.show()
        else:
            self._reattach_display()

    def _reattach_display(self):
        """Reattach display to main window"""
        if self.detached_window:
            # Move widget back to container
            if self.current_mode == "camera":
                self.container_layout.addWidget(self.camera_display)
            else:
                self.container_layout.addWidget(self.splitter)

            # Close detached window
            self.detached_window.close()
            self.detached_window = None

    def _handle_detached_close(self, event):
        """Handle closing of detached window"""
        self.detach_btn.setChecked(False)
        self._reattach_display()
        event.accept()

    def resizeEvent(self, event):
        """Handle widget resize"""
        super().resizeEvent(event)

        # Update frames to fit new size
        if self.current_mode == "camera":
            if self.showing_processed and self.processed_frame is not None:
                self._show_frame(self.camera_display, self.processed_frame)
            elif self.original_frame is not None:
                self._show_frame(self.camera_display, self.original_frame)
        else:
            if self.original_frame is not None:
                self._show_frame(self.original_display, self.original_frame)
            if self.processed_frame is not None:
                self._show_frame(self.prediction_display, self.processed_frame)

        # Set splitter sizes
        if self.current_mode == "split":
            self.splitter.setSizes([self.width() // 2, self.width() // 2])
