import time
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QComboBox,
    QLabel,
    QSpinBox,
    QFileDialog,
    QFormLayout,
    QProgressBar,
    QSplitter,
    QApplication,
)
from PyQt6.QtGui import QImage, QPixmap  # Add QImage here
from PyQt6.QtCore import pyqtSignal, Qt
import platform
import cv2
import numpy as np
import glob
import os
from ..models.camera_model import CameraBackend


class LiveCameraPanel(QWidget):
    """Panel for live camera controls"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main vertical splitter
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Settings widget containing all controls
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(5, 5, 5, 5)

        # Horizontal splitter for device and stream settings
        self.settings_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Device Settings
        device_group = QGroupBox("Device Settings")
        device_layout = QFormLayout()

        # Camera selector with refresh button
        camera_select_widget = QWidget()
        camera_select_layout = QHBoxLayout(camera_select_widget)

        self.camera_selector = QComboBox()
        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setProperty("iconOnly", True)
        self.refresh_btn.setProperty("secondary", True)

        camera_select_layout.addWidget(self.camera_selector)
        camera_select_layout.addWidget(self.refresh_btn)

        device_layout.addRow("Camera:", camera_select_widget)

        # Add backend selector
        self.backend_selector = QComboBox()
        system = platform.system().lower()

        if system == "windows":
            self.backend_selector.addItem("DirectShow", CameraBackend.DSHOW)
            self.backend_selector.addItem("Media Foundation", CameraBackend.MSMF)
        elif system == "linux":
            self.backend_selector.addItem("V4L2", CameraBackend.V4L)
            self.backend_selector.addItem("GStreamer", CameraBackend.GSTREAMER)
        elif system == "darwin":  # macOS
            self.backend_selector.addItem("AVFoundation", CameraBackend.AVFOUNDATION)

        device_layout.addRow("Backend:", self.backend_selector)

        # Add status display to device settings
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel("Not connected")
        self.status_progress = QProgressBar()
        self.status_progress.setVisible(False)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.status_progress)
        device_layout.addRow("Status:", status_widget)

        # Add control buttons to device settings
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)

        self.start_btn = QPushButton("Start Camera")
        self.stop_btn = QPushButton("Stop Camera")
        self.start_btn.setProperty("primary", True)
        self.stop_btn.setProperty("primary", True)
        self.stop_btn.setEnabled(False)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        device_layout.addRow("Controls:", button_widget)

        device_group.setLayout(device_layout)
        self.settings_splitter.addWidget(device_group)

        # Stream Settings
        stream_group = QGroupBox("Stream Settings")
        stream_layout = QFormLayout()

        # Resolution selector
        self.resolution_selector = QComboBox()
        self.resolution_selector.addItem("640x480", (640, 480))
        self.resolution_selector.addItem("800x600", (800, 600))
        self.resolution_selector.addItem("1280x720", (1280, 720))
        self.resolution_selector.addItem("1920x1080", (1920, 1080))
        stream_layout.addRow("Resolution:", self.resolution_selector)

        # FPS control
        self.fps_spinner = QSpinBox()
        self.fps_spinner.setRange(1, 60)
        self.fps_spinner.setValue(30)
        stream_layout.addRow("FPS:", self.fps_spinner)

        stream_group.setLayout(stream_layout)
        self.settings_splitter.addWidget(stream_group)

        settings_layout.addWidget(self.settings_splitter)
        self.main_splitter.addWidget(settings_widget)

        # Add splitter to main layout
        layout.addWidget(self.main_splitter)

        # Set initial splitter sizes
        self.settings_splitter.setSizes([300, 300])  # Equal width for settings panels

    def get_camera_id(self):
        return self.camera_selector.currentData()

    def get_resolution(self):
        return self.resolution_selector.currentData()

    def get_fps(self):
        return self.fps_spinner.value()

    def get_backend(self):
        return self.backend_selector.currentData()


class ImageInputPanel(QWidget):
    """Panel for image input controls"""

    image_selected = pyqtSignal(str)  # Emit selected file path
    processing_requested = pyqtSignal(np.ndarray)  # Emit image for processing

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Initialize instance variables first
        self.current_image = None
        self.file_list = []
        self.current_file_index = 0
        self.current_file_path = None

        # Create labels first before using them
        self.file_label = QLabel("No image selected")
        self.file_label.setWordWrap(True)

        # Input controls group
        input_group = QGroupBox("Image Input")
        input_layout = QVBoxLayout()

        # File controls
        select_layout = QHBoxLayout()
        self.file_btn = QPushButton("Select Image")
        self.folder_btn = QPushButton("Select Folder")
        self.process_btn = QPushButton("Process and Save")

        select_layout.addWidget(self.file_btn)
        select_layout.addWidget(self.folder_btn)
        select_layout.addWidget(self.process_btn)

        # Add all to main layout
        input_layout.addLayout(select_layout)
        input_layout.addWidget(self.file_label)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        layout.addStretch()

        # Connect signals
        self.file_btn.clicked.connect(self._select_image)
        self.folder_btn.clicked.connect(self._select_folder)
        self.process_btn.clicked.connect(self.process_current_image)

        # Store current image
        self.current_image = None

        # Store file list
        self.file_list = []
        self.current_file_index = 0

    def _select_image(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )
        if file_paths:
            self.file_list = file_paths
            self.current_file_index = 0
            self._load_current_image()

    def _select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            # Get all image files in folder
            self.file_list = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                self.file_list.extend(glob.glob(os.path.join(folder_path, ext)))

            if self.file_list:
                self.current_file_index = 0
                self._load_current_image()
            else:
                self.file_label.setText("No compatible images found in folder")

    def _load_current_image(self):
        """Load the current image from file list"""
        if 0 <= self.current_file_index < len(self.file_list):
            file_path = self.file_list[self.current_file_index]
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.current_image = image
                    self.current_file_path = file_path
                    # Only emit image_selected to show original
                    self.image_selected.emit(file_path)
                    self.file_label.setText(
                        f"Image {self.current_file_index + 1}/{len(self.file_list)}: {os.path.basename(file_path)}"
                    )
                    self.process_btn.setEnabled(True)
            except Exception as e:
                self.file_label.setText(f"Error loading image: {str(e)}")

    def process_current_image(self):
        """Process all images in the list when button clicked"""
        if not self.file_list:
            return

        processed_count = 0
        error_count = 0

        # Process all images in the list
        for i, file_path in enumerate(self.file_list):
            try:
                # Update UI to show current progress
                self.file_label.setText(
                    f"Processing image {i + 1}/{len(self.file_list)}: {os.path.basename(file_path)}"
                )
                QApplication.processEvents()

                # Load and emit image for processing
                image = cv2.imread(file_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.current_image = image
                    self.current_file_path = file_path
                    # Emit for processing
                    self.processing_requested.emit(image)
                    processed_count += 1

                    # Brief pause to allow processing to complete
                    QApplication.processEvents()
                    time.sleep(0.1)  # Short sleep

            except Exception as e:
                error_count += 1
                continue

        # Show final summary message
        summary = []
        if processed_count > 0:
            summary.append(f"Successfully processed {processed_count} images")
        if error_count > 0:
            summary.append(f"Failed to process {error_count} images")

        self.file_label.setText(" | ".join(summary))

    def update_detection_result(self, result_image: np.ndarray):
        """Store detection results"""
        if result_image is not None:
            self.current_prediction = result_image


class VideoInputPanel(QWidget):
    """Panel for video input controls"""

    video_selected = pyqtSignal(str)
    playback_started = pyqtSignal(str)  # Emit video path when starting
    playback_stopped = pyqtSignal()
    frame_processed = pyqtSignal(np.ndarray)  # Emit processed frame

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video controls group
        controls_group = QGroupBox("Video Controls")
        controls_layout = QVBoxLayout()

        # File controls
        file_layout = QHBoxLayout()
        self.file_btn = QPushButton("Select Video")
        self.folder_btn = QPushButton("Select Folder")
        file_layout.addWidget(self.file_btn)
        file_layout.addWidget(self.folder_btn)

        # Playback controls
        control_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.stop_btn = QPushButton("Stop")
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.stop_btn)

        # Progress indicator
        self.progress_widget = QWidget()
        progress_layout = QHBoxLayout(self.progress_widget)
        self.progress_bar = QProgressBar()
        self.time_label = QLabel("00:00 / 00:00")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.time_label)
        self.progress_widget.setVisible(False)

        # File info
        self.file_label = QLabel("No video selected")
        self.file_label.setWordWrap(True)

        # Add playlist controls
        playlist_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        playlist_layout.addWidget(self.prev_btn)
        playlist_layout.addWidget(self.next_btn)

        # Add everything to layouts
        controls_layout.addLayout(file_layout)
        controls_layout.addWidget(self.file_label)
        controls_layout.addLayout(control_layout)
        controls_layout.addWidget(self.progress_widget)
        controls_layout.addLayout(playlist_layout)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        layout.addStretch()

        # Connect signals
        self.file_btn.clicked.connect(self._select_video)
        self.folder_btn.clicked.connect(self._select_folder)
        self.play_btn.clicked.connect(self._on_play)
        self.stop_btn.clicked.connect(self._on_stop)
        self.prev_btn.clicked.connect(self._play_previous)
        self.next_btn.clicked.connect(self._play_next)

        # Store current video path
        self.current_video_path = None

        # Store video list
        self.video_list = []
        self.current_video_index = 0

    def _select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            # Get all video files in folder
            self.video_list = []
            for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
                self.video_list.extend(glob.glob(os.path.join(folder_path, ext)))

            if self.video_list:
                self.current_video_index = 0
                self._load_current_video()
                self.prev_btn.setEnabled(False)
                self.next_btn.setEnabled(len(self.video_list) > 1)
            else:
                self.file_label.setText("No compatible videos found in folder")

    def _select_video(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Videos",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if file_paths:
            self.video_list = file_paths
            self.current_video_index = 0
            self._load_current_video()
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(len(self.video_list) > 1)

    def _load_current_video(self):
        """Load the current video from playlist"""
        if 0 <= self.current_video_index < len(self.video_list):
            file_path = self.video_list[self.current_video_index]
            self.current_video_path = file_path
            self.file_label.setText(
                f"Video {self.current_video_index + 1}/{len(self.video_list)}: {os.path.basename(file_path)}"
            )
            self.play_btn.setEnabled(True)

            # Show first frame preview without processing
            try:
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cap.release()
            except Exception as e:
                self.file_label.setText(f"Error loading video: {str(e)}")

            # Emit video selected signal but don't start processing
            self.video_selected.emit(file_path)

    def update_detection_result(self, result_frame: np.ndarray):
        """Update preview with detection results"""
        if result_frame is not None:
            self.current_prediction = result_frame

    def _on_play(self):
        """Start video processing when play button clicked"""
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.file_btn.setEnabled(False)
        self.progress_widget.setVisible(True)
        # Now start processing and playback
        self.playback_started.emit(self.current_video_path)
        self._process_and_save_video()

    def _on_stop(self):
        """Handle stop button click"""
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.file_btn.setEnabled(True)
        self.folder_btn.setEnabled(True)  # Also enable folder button
        self.progress_widget.setVisible(False)
        self.time_label.setText("00:00 / 00:00")
        self.progress_bar.setValue(0)
        self.playback_stopped.emit()

    def _play_previous(self):
        """Play previous video in playlist"""
        if self.current_video_index > 0:
            self._on_stop()  # Stop current video
            self.current_video_index -= 1
            self._load_current_video()
            self.next_btn.setEnabled(True)
            self.prev_btn.setEnabled(self.current_video_index > 0)

    def _play_next(self):
        """Play next video in playlist"""
        if self.current_video_index < len(self.video_list) - 1:
            self._on_stop()  # Stop current video
            self.current_video_index += 1
            self._load_current_video()
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(
                self.current_video_index < len(self.video_list) - 1
            )

    def _process_and_save_video(self):
        output_dir = os.path.join("output", "predictions", "videos")
        os.makedirs(output_dir, exist_ok=True)

        # Create output video path
        base_name = os.path.basename(self.current_video_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_prediction{ext}")

        # Set up video writer
        cap = cv2.VideoCapture(self.current_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            total_frames_processed = 0
            error_frames = 0
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress_bar.setRange(0, total_frames)

            # Add a flag to check if video should continue processing
            self.is_processing = True

            while cap.isOpened() and self.is_processing:  # Check flag in loop
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to RGB and show original
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_processed.emit(frame_rgb)

                # Wait briefly for processing
                QApplication.processEvents()
                time.sleep(0.01)  # Small delay to allow processing

                # Update progress
                frame_count += 1
                self.progress_bar.setValue(frame_count)

                # Update time display
                current_time = frame_count / cap.get(cv2.CAP_PROP_FPS)
                total_time = total_frames / cap.get(cv2.CAP_PROP_FPS)
                time_str = f"{int(current_time // 60):02d}:{int(current_time % 60):02d} / {int(total_time // 60):02d}:{int(total_time % 60):02d}"
                self.time_label.setText(time_str)

                if self.play_btn.isEnabled():  # If stop was clicked
                    self.is_processing = False
                    break

                QApplication.processEvents()

            # Clean up
            cap.release()
            out.release()

            if not self.is_processing:
                self.file_label.setText("Video processing stopped")
            else:
                # Show final summary message only at the end
                if total_frames_processed > 0:
                    self.file_label.setText(
                        f"Processed {total_frames_processed} frames | Saved to {output_path}"
                    )
                if error_frames > 0:
                    self.file_label.setText(
                        f"Processed {total_frames_processed} frames ({error_frames} errors) | Saved to {output_path}"
                    )

        except Exception as e:
            self.file_label.setText(f"Error processing video: {str(e)}")
        finally:
            self.is_processing = False
