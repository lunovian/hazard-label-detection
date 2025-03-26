from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QGroupBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QMessageBox,
    QCheckBox,
    QSpinBox,
    QStatusBar,
    QSizePolicy,
)
from PyQt6.QtGui import QImage, QPixmap, QIcon, QAction
from PyQt6.QtCore import Qt, pyqtSignal, QSize
import cv2
import numpy as np
import os
from .camera_view import CameraView
from .controls_panel import ControlsPanel
from .results_table import ResultsTable
from .input_panels import LiveCameraPanel, ImageInputPanel, VideoInputPanel
from ..utils.ui_helpers import show_styled_help, show_styled_confirmation
from .unified_display import UnifiedDisplayView


class MainWindow(QMainWindow):
    """Main application window"""

    # Define signals
    camera_start_requested = pyqtSignal(int, tuple, int)  # camera_id, resolution, fps
    camera_stop_requested = pyqtSignal()
    detection_toggle_requested = pyqtSignal(bool)
    tracking_toggle_requested = pyqtSignal(bool)
    model_load_requested = pyqtSignal(str)
    confidence_changed = pyqtSignal(float)
    iou_changed = pyqtSignal(float)
    screenshot_requested = pyqtSignal()
    export_results_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GHS Hazard Label Detector")

        # Set application icon
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "resources",
            "icons",
            "app_icon.png",
        )
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.resize(1200, 800)
        self.setMinimumSize(800, 600)  # Set a reasonable minimum size

        # Create the main layout with proper spacing
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Create a splitter to allow resizing panels with better grip
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(4)  # Slightly wider handle for easier grabbing
        self.main_layout.addWidget(self.splitter)

        # Create left panel (camera view and controls) with improved layout
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)

        # Replace camera_view with unified_display
        self.unified_display = UnifiedDisplayView()
        self.unified_display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.left_layout.addWidget(self.unified_display, 1)

        # Add controls panel with fixed height
        self.controls_panel = ControlsPanel()
        self.controls_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.left_layout.addWidget(self.controls_panel)

        # Create right panel (results and settings) with proper layout
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)

        # Add results table with proper stretch factor
        self.results_table = ResultsTable()
        self.results_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.right_layout.addWidget(self.results_table, 1)  # Add stretch factor

        # Add panels to splitter with better proportions
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.right_panel)

        # Set initial split ratio to 70:30 (percentage based for flexibility)
        total_width = self.width()
        left_width = int(total_width * 0.7)
        right_width = int(total_width * 0.3)
        self.splitter.setSizes([left_width, right_width])

        # Set up status bar with fixed height and consistent padding
        self.statusBar = QStatusBar()
        self.statusBar.setFixedHeight(28)
        self.setStatusBar(self.statusBar)

        # Add status elements with consistent spacing and proper IDs for styling
        self.status_fps = QLabel("FPS: --")
        self.status_fps.setObjectName("status_fps")
        self.status_fps.setMinimumWidth(85)

        self.status_objects = QLabel("Objects: --")
        self.status_objects.setObjectName("status_objects")
        self.status_objects.setMinimumWidth(110)

        self.status_model = QLabel("Model: Not loaded")
        self.status_model.setObjectName("status_model")
        self.status_model.setMinimumWidth(160)

        self.statusBar.addPermanentWidget(self.status_model)
        self.statusBar.addPermanentWidget(self.status_objects)
        self.statusBar.addPermanentWidget(self.status_fps)

        # Set up menu bar
        self.setup_menu()

        # Connect internal signals
        self._connect_signals()

        # Connect tab changes
        self.controls_panel.tab_widget.currentChanged.connect(self._handle_tab_change)
        self.unified_display.view_toggled.connect(self._handle_view_toggle)

        # Set the camera tab as default
        self.controls_panel.tab_widget.setCurrentIndex(0)
        self.unified_display.set_mode("camera")

        # Set more reasonable default size for camera view
        self.resize(1280, 800)

    def setup_menu(self):
        """Set up application menu bar"""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        load_model_action = QAction("Load Model From File...", self)
        load_model_action.triggered.connect(self._on_load_model)
        file_menu.addAction(load_model_action)

        refresh_models_action = QAction("Refresh Model List", self)
        refresh_models_action.triggered.connect(
            self.controls_panel.refresh_models_clicked
        )
        file_menu.addAction(refresh_models_action)

        file_menu.addSeparator()

        screenshot_action = QAction("Take Screenshot", self)
        screenshot_action.triggered.connect(self.screenshot_requested)
        file_menu.addAction(screenshot_action)

        export_action = QAction("Export Results", self)
        export_action.triggered.connect(self.export_results_requested)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Camera menu
        camera_menu = menu_bar.addMenu("&Camera")

        start_camera_action = QAction("Start Camera", self)
        start_camera_action.triggered.connect(self._on_start_camera)
        camera_menu.addAction(start_camera_action)

        stop_camera_action = QAction("Stop Camera", self)
        stop_camera_action.triggered.connect(self.camera_stop_requested)
        camera_menu.addAction(stop_camera_action)

        # Models menu
        models_menu = menu_bar.addMenu("&Models")

        download_model_action = QAction("Download Sample Model...", self)
        download_model_action.triggered.connect(self._on_download_model)
        models_menu.addAction(download_model_action)

        open_models_folder_action = QAction("Open Models Folder", self)
        open_models_folder_action.triggered.connect(self._on_open_models_folder)
        models_menu.addAction(open_models_folder_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _connect_signals(self):
        """Connect internal signals between UI components"""
        # Connect controls panel signals
        self.controls_panel.camera_start_clicked.connect(self._on_start_camera)
        self.controls_panel.camera_stop_clicked.connect(self.camera_stop_requested)
        self.controls_panel.detection_toggled.connect(self.detection_toggle_requested)
        self.controls_panel.tracking_toggled.connect(self.tracking_toggle_requested)
        self.controls_panel.confidence_changed.connect(self.confidence_changed)
        self.controls_panel.iou_changed.connect(self.iou_changed)

        # Add tab change handling with state tracking
        self.is_camera_active = False
        self.controls_panel.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index):
        """Handle tab changes and update unified display accordingly"""
        if index == 0:  # Camera tab
            self.unified_display.set_mode("camera")
            if self.is_camera_active:
                camera_id = self.controls_panel.get_camera_id()
                resolution = self.controls_panel.get_resolution()
                fps = self.controls_panel.get_fps()
                self.camera_start_requested.emit(camera_id, resolution, fps)
        else:
            self.unified_display.set_mode("split")
            # Store camera state before stopping
            self.is_camera_active = (
                self.controls_panel.camera_panel.stop_btn.isEnabled()
            )
            # Stop camera when switching to other tabs
            self.camera_stop_requested.emit()
            # Clear the display to prepare for other content
            self.unified_display.clear()

    def _handle_tab_change(self, index):
        """Handle tab changes"""
        modes = ["camera", "image", "video"]
        self.unified_display.set_mode(modes[index])

        # Stop camera if switching away from camera tab
        if index != 0 and self.is_camera_active:
            self.is_camera_active = False
            self.camera_stop_requested.emit()

    def _handle_view_toggle(self, show_processed):
        """Handle view toggle between original and processed"""
        # Update display based on current content
        if show_processed and hasattr(self, "last_processed_frame"):
            self.unified_display.update_frame(self.last_processed_frame, True)
        elif hasattr(self, "last_original_frame"):
            self.unified_display.update_frame(self.last_original_frame, False)

    def start_camera(self):
        """Start camera and update state"""
        self.is_camera_active = True
        self._on_start_camera()

    def stop_camera(self):
        """Stop camera and update state"""
        self.is_camera_active = False
        self.camera_stop_requested.emit()
        self.unified_display.clear()  # Use unified_display instead of camera_view

    def _on_load_model(self):
        """Handle load model action"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load YOLO Model",
            "",
            "Model Files (*.pt *.pth *.weights);;All Files (*)",
        )
        if file_path:
            self.model_load_requested.emit(file_path)

    def _on_start_camera(self):
        """Handle start camera action"""
        camera_id = self.controls_panel.get_camera_id()
        resolution = self.controls_panel.get_resolution()
        fps = self.controls_panel.get_fps()
        self.camera_start_requested.emit(camera_id, resolution, fps)

    def _on_download_model(self):
        """Handle download sample model action"""
        try:
            import subprocess
            import sys
            import os

            # Get the path to download_sample_model.py
            script_path = os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                ),
                "download_sample_model.py",
            )

            if sys.platform.startswith("win"):
                # On Windows, use a new console window
                subprocess.Popen(
                    ["python", script_path], creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # On other platforms, use terminal
                subprocess.Popen(["python", script_path])

            self.show_info(
                "Download script started. Please follow the instructions in the new window."
            )
        except Exception as e:
            self.show_error(f"Failed to start download script: {str(e)}")

    def _on_open_models_folder(self):
        """Open the models folder in file explorer"""
        import os
        import subprocess
        import sys

        models_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            ),
            "models",
        )

        # Create the directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Open the directory
        if sys.platform.startswith("win"):
            os.startfile(models_dir)
        elif sys.platform.startswith("darwin"):  # macOS
            subprocess.Popen(["open", models_dir])
        else:  # Linux
            subprocess.Popen(["xdg-open", models_dir])

    def _show_about(self):
        """Show about dialog with consistent styling"""
        about_box = QMessageBox(self)
        about_box.setWindowTitle("About GHS Hazard Label Detector")
        about_box.setText("GHS Hazard Label Detector v0.1")
        about_box.setInformativeText(
            "A real-time detection and tracking application for GHS hazard labels.\n\n"
            "Developed using PyQt6 and YOLO."
        )
        about_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        # Ensure styling is applied
        about_box.setObjectName("aboutDialog")
        about_box.exec()

    def _show_help(self):
        """Show styled help dialog"""
        help_content = """
        <h2>GHS Hazard Label Detector Help</h2>
        <p>This application helps you detect and track GHS hazard labels in real-time using computer vision.</p>
        
        <h3>Key Features:</h3>
        <ul>
            <li><b>Real-time Detection:</b> View hazard labels as they're detected</li>
            <li><b>Object Tracking:</b> Follow hazard labels across video frames</li>
            <li><b>Result Export:</b> Save detection results to CSV format</li>
        </ul>
        
        <h3>Getting Started:</h3>
        <ol>
            <li>Select a camera from the dropdown</li>
            <li>Load a detection model</li>
            <li>Click "Start Camera" to begin</li>
        </ol>
        
        <h3>Keyboard Shortcuts:</h3>
        <table>
            <tr><td><b>Ctrl+O</b></td><td>Load Model</td></tr>
            <tr><td><b>Ctrl+S</b></td><td>Take Screenshot</td></tr>
            <tr><td><b>Ctrl+E</b></td><td>Export Results</td></tr>
            <tr><td><b>F1</b></td><td>Show Help</td></tr>
        </table>
        
        <p>For more information, visit our <a href="https://github.com/yourusername/hazard-label-detection">GitHub repository</a>.</p>
        """

        show_styled_help(self, "Application Help", help_content)

    def _on_close_request(self):
        """Show confirmation before closing"""
        confirmed = show_styled_confirmation(
            self,
            "Confirm Exit",
            "Are you sure you want to exit the application? Any unsaved detection results will be lost.",
        )
        if confirmed:
            self.camera_stop_requested.emit()
            self.close()

    def closeEvent(self, event):
        """Handle window close event"""
        event.ignore()  # Don't close yet
        self._on_close_request()  # Show confirmation dialog

    def update_camera_frame(self, frame: np.ndarray, detections=None):
        """Update camera view with a new frame"""
        # Only update if we're on the camera tab
        if self.controls_panel.tab_widget.currentIndex() == 0:
            self.unified_display.update_frame(frame, True)  # True for processed frame
            self.last_processed_frame = frame

    def update_results_table(self, detections, class_names):
        """Update results table with new detections"""
        self.results_table.update_detections(detections, class_names)

    def update_status(self, fps, num_objects, model_name=None):
        """Update status bar information"""
        self.status_fps.setText(f"FPS: {fps:.1f}")
        self.status_objects.setText(f"Objects: {num_objects}")
        if model_name:
            self.status_model.setText(f"Model: {model_name}")

    def show_error(self, message):
        """Display styled error message"""
        error_box = QMessageBox(self)
        error_box.setWindowTitle("Error")
        error_box.setText(message)
        error_box.setIcon(QMessageBox.Icon.Critical)
        error_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        # Ensure styling is applied
        error_box.setObjectName("errorDialog")
        error_box.exec()

    def show_info(self, message):
        """Display styled information message"""
        info_box = QMessageBox(self)
        info_box.setWindowTitle("Information")
        info_box.setText(message)
        info_box.setIcon(QMessageBox.Icon.Information)
        info_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        # Ensure styling is applied
        info_box.setObjectName("infoDialog")
        info_box.exec()

    def resizeEvent(self, event):
        """Handle window resize event more effectively"""
        super().resizeEvent(event)

        # Maintain a 70:30 split ratio when resizing
        total_width = self.splitter.width()
        left_width = int(total_width * 0.7)
        right_width = total_width - left_width
        self.splitter.setSizes([left_width, right_width])

        # Update display if we have content
        if hasattr(self, "last_processed_frame"):
            self.unified_display.update_frame(self.last_processed_frame, True)
        elif hasattr(self, "last_original_frame"):
            self.unified_display.update_frame(self.last_original_frame, False)

    def update_camera_progress(self, value):
        """Update camera initialization progress"""
        self.controls_panel.set_progress(value)

    def show_original_image(self, frame: np.ndarray):
        """Show original image in unified display"""
        self.unified_display.update_frame(frame, False)  # False for original frame
        self.last_original_frame = frame
        # Reset detection count when showing original image
        self.unified_display.update_detection_count(0)

    def update_preview(self, frame: np.ndarray, detection_count=0):
        """Update preview with processed frame and detection count"""
        if self.is_camera_active:
            self.stop_camera()
        self.unified_display.update_frame(frame, True)  # True for processed frame
        self.last_processed_frame = frame
        # Update detection count
        self.unified_display.update_detection_count(detection_count)

    def clear_preview(self):
        """Clear unified display"""
        self.unified_display.clear()
