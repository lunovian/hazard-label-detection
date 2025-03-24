import cv2
import time
import os
import numpy as np
import pandas as pd
import logging
from PyQt6.QtCore import QObject, QTimer
from datetime import datetime
from ..models.camera_model import CameraModel, CameraBackend
from ..models.detection_model import DetectionModel, DetectionResult
from ..views.main_window import MainWindow
from ..utils.model_utils import ModelManager
from ..utils.compatibility import log_dependency_versions

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AppController")


class AppController(QObject):
    """Main application controller that connects models and views"""

    def __init__(self, model=None, view=None):
        super().__init__()

        # Log dependency versions to help with troubleshooting
        log_dependency_versions()

        # Initialize models
        self.detection_model = model if model else DetectionModel()
        self.camera_model = CameraModel()

        # Initialize view
        self.view = view if view else MainWindow()

        # State management
        self.detection_enabled = True
        self.processing_frame = False
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        self.output_dir = "output"

        # Connect signals from view
        self._connect_view_signals()

        # Connect signals from model
        self._connect_model_signals()

        # Set up camera list in view
        self.view.controls_panel.set_camera_list(
            self.camera_model.get_available_cameras()
        )

        # Set up model list in view but don't load any model
        self.refresh_models()

        # Update status bar to indicate no model loaded
        self.view.status_model.setText("Model: Not loaded")

    def _connect_view_signals(self):
        """Connect signals from the view to controller methods"""
        # Camera controls
        self.view.camera_start_requested.connect(self.start_camera)
        self.view.camera_stop_requested.connect(self.stop_camera)

        # Connect the refresh cameras signal
        self.view.controls_panel.camera_refresh_clicked.connect(self.refresh_cameras)

        # Connect backend change signal
        self.view.controls_panel.backend_changed.connect(self.set_camera_backend)

        # Detection controls
        self.view.detection_toggle_requested.connect(self.toggle_detection)
        self.view.tracking_toggle_requested.connect(self.toggle_tracking)
        self.view.confidence_changed.connect(self.set_confidence)
        self.view.iou_changed.connect(self.set_iou)

        # Model controls
        self.view.model_load_requested.connect(self.load_model)

        # Export controls
        self.view.screenshot_requested.connect(self.take_screenshot)
        self.view.export_results_requested.connect(self.export_results)

        # Connect model selection signals
        self.view.controls_panel.model_selected.connect(self.load_model)
        self.view.controls_panel.refresh_models_clicked.connect(self.refresh_models)

    def _connect_model_signals(self):
        """Connect signals from models to controller methods"""
        # Camera model signals
        self.camera_model.frame_captured.connect(self.process_frame)
        self.camera_model.camera_error.connect(self.handle_camera_error)
        self.camera_model.camera_status.connect(self.handle_camera_status)
        self.camera_model.camera_connected.connect(self.handle_camera_connection)
        self.camera_model.camera_list_updated.connect(self.handle_camera_list_updated)
        self.camera_model.camera_progress.connect(
            self.handle_camera_progress
        )  # Add this line

    def handle_camera_error(self, error_message):
        """Handle camera error messages"""
        logger.error(f"Camera error: {error_message}")
        self.view.show_error(f"Camera Error: {error_message}")

    def handle_camera_status(self, status_message):
        """Handle camera status updates"""
        logger.info(f"Camera status: {status_message}")
        self.view.controls_panel.set_status_message(status_message)

    def handle_camera_connection(self, connected):
        """Handle camera connection state changes"""
        self.view.controls_panel.set_connection_status(connected)

    def handle_camera_list_updated(self, camera_list):
        """Handle updated camera list"""
        self.view.controls_panel.set_camera_list(camera_list)

    def handle_camera_progress(self, progress_value):
        """Handle camera initialization progress updates"""
        self.view.controls_panel.set_progress(progress_value)

    def _load_default_model(self):
        """
        Don't load a default model automatically, just update the UI
        to reflect that no model is loaded
        """
        self.view.status_model.setText("Model: Not loaded")
        self.view.show_info(
            "Please select a model from the dropdown and click 'Load Selected Model'"
        )

    def refresh_models(self):
        """Refresh the list of available models"""
        models = self.detection_model.refresh_available_models()
        self.view.controls_panel.set_model_list(models)

        # Log the found models
        logger.info(f"Found {len(models)} model(s).")
        for path, name in models:
            logger.info(f"Model: {name} at {path}")

    def start_camera(self, camera_id=None, resolution=None, fps=None):
        """Start camera capture"""
        # Use values from view if not provided
        if camera_id is None:
            camera_id = self.view.controls_panel.get_camera_id()
        if resolution is None:
            resolution = self.view.controls_panel.get_resolution()
        if fps is None:
            fps = self.view.controls_panel.get_fps()
        backend = self.view.controls_panel.get_backend()

        # Handle special case for IP camera (-1)
        if camera_id == -1:
            from PyQt6.QtWidgets import QInputDialog, QLineEdit

            camera_url, ok = QInputDialog.getText(
                self.view,
                "IP Camera URL",
                "Enter RTSP, HTTP or local video file URL:",
                QLineEdit.EchoMode.Normal,
                "rtsp://user:pass@192.168.1.100:554/stream",
            )
            if ok and camera_url:
                camera_id = camera_url
            else:
                # User canceled
                self.view.controls_panel.set_connection_status(False)
                return

        # Start camera
        success = self.camera_model.start_camera(camera_id, resolution, fps, backend)
        if not success:
            logger.error(f"Failed to start camera {camera_id}")
            self.view.controls_panel.set_connection_status(False)

    def stop_camera(self):
        """Stop camera capture"""
        self.camera_model.stop_camera()
        self.view.camera_view.clear()
        self.view.results_table.clear()
        self.view.update_status(0, 0)

    def refresh_cameras(self):
        """Refresh the list of available cameras"""
        self.camera_model.refresh_cameras()

    def set_camera_backend(self, backend):
        """Set the camera backend"""
        self.camera_model.set_backend(backend)

    def process_frame(self, frame):
        """Process a frame from the camera"""
        if self.processing_frame:
            return

        self.processing_frame = True

        # Update frame count and calculate FPS
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_update

        if elapsed >= 1.0:  # Update FPS once per second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time

        if self.detection_enabled and self.detection_model.model:
            try:
                # Run detection on frame
                result = self.detection_model.detect(frame)

                # Update view with detection results
                self.view.update_camera_frame(result.annotated_frame)

                # Update results table
                if result.detections is not None:
                    self.view.update_results_table(
                        result.detections, self.detection_model.class_names
                    )

                    # Update status bar
                    num_objects = len(result.detections) if result.detections else 0
                    self.view.update_status(self.fps, num_objects)
                else:
                    self.view.update_status(self.fps, 0)
            except Exception as e:
                logger.error(f"Error during frame processing: {str(e)}")
                # Just display the frame without detection on error
                self.view.update_camera_frame(frame)
                self.view.update_status(self.fps, 0)
        else:
            # Just display the frame without detection
            self.view.update_camera_frame(frame)
            self.view.update_status(self.fps, 0)

        self.processing_frame = False

    def toggle_detection(self, enabled):
        """Toggle detection processing"""
        self.detection_enabled = enabled

    def toggle_tracking(self, enabled):
        """Toggle object tracking"""
        self.detection_model.toggle_tracking(enabled)

    def set_confidence(self, value):
        """Set confidence threshold"""
        self.detection_model.set_conf_threshold(value)

    def set_iou(self, value):
        """Set IoU threshold"""
        self.detection_model.set_iou_threshold(value)

    def load_model(self, model_path):
        """Load a YOLO model from file"""
        try:
            logger.info(f"Loading model: {model_path}")
            model_loaded = self.detection_model.load_model(model_path)
            if model_loaded:
                model_name = os.path.basename(model_path)
                self.view.update_status(self.fps, 0, model_name)
                self.view.show_info(f"Model '{model_name}' loaded successfully")

                # Update selected model in view
                self.view.controls_panel.set_current_model(model_path)
            else:
                self.view.show_error(f"Failed to load model from {model_path}")
        except Exception as e:
            self.view.show_error(f"Error loading model: {str(e)}")

    def take_screenshot(self):
        """Take a screenshot of the current frame"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"screenshot_{timestamp}.jpg")

        # Get the current frame from the camera view
        if self.view.camera_view.last_frame is not None:
            # Convert from RGB to BGR for OpenCV
            frame = cv2.cvtColor(self.view.camera_view.last_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, frame)
            self.view.show_info(f"Screenshot saved as {filename}")
        else:
            self.view.show_error("No camera frame available")

    def export_results(self):
        """Export detection results to CSV"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"detections_{timestamp}.csv")

        # Get current detections
        if (
            self.detection_model.model
            and hasattr(self.view.results_table, "table")
            and self.view.results_table.table.rowCount() > 0
        ):  # Fixed: Changed curly brace to parenthesis
            rows = []
            for row in range(self.view.results_table.table.rowCount()):
                track_id = self.view.results_table.table.item(row, 0).text()
                class_name = self.view.results_table.table.item(row, 1).text()
                confidence = self.view.results_table.table.item(row, 2).text()
                position = self.view.results_table.table.item(row, 3).text()
                size = self.view.results_table.table.item(row, 4).text()

                rows.append(
                    {
                        "ID": track_id,
                        "Class": class_name,
                        "Confidence": confidence,
                        "Position": position,
                        "Size": size,
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(filename, index=False)
                self.view.show_info(f"Detection results exported to {filename}")
                return

        self.view.show_error("No detection results to export")
