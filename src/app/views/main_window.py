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
)
from PyQt6.QtGui import QImage, QPixmap, QIcon, QAction
from PyQt6.QtCore import Qt, pyqtSignal, QSize
import cv2
import numpy as np
from .camera_view import CameraView
from .controls_panel import ControlsPanel
from .results_table import ResultsTable


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
        self.resize(1200, 800)

        # Create the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Create a splitter to allow resizing panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Create left panel (camera view and controls)
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)

        # Add camera view
        self.camera_view = CameraView()
        self.left_layout.addWidget(self.camera_view, 3)

        # Add controls panel
        self.controls_panel = ControlsPanel()
        self.left_layout.addWidget(self.controls_panel, 1)

        # Create right panel (results and settings)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)

        # Add results table
        self.results_table = ResultsTable()
        self.right_layout.addWidget(self.results_table)

        # Add panels to splitter
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setSizes([700, 500])  # Initial split sizes

        # Set up status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.status_fps = QLabel("FPS: --")
        self.status_objects = QLabel("Objects: --")
        self.status_model = QLabel("Model: Not loaded")
        self.statusBar.addPermanentWidget(self.status_model)
        self.statusBar.addPermanentWidget(self.status_objects)
        self.statusBar.addPermanentWidget(self.status_fps)

        # Set up menu bar
        self.setup_menu()

        # Connect internal signals
        self._connect_signals()

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
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About GHS Hazard Label Detector",
            "GHS Hazard Label Detector v0.1\n\n"
            "A real-time detection and tracking application for GHS hazard labels.\n\n"
            "Developed using PyQt6 and YOLO.",
        )

    def update_camera_frame(self, frame: np.ndarray, detections=None):
        """Update camera view with a new frame"""
        self.camera_view.update_frame(frame)

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
        """Display error message"""
        QMessageBox.critical(self, "Error", message)

    def show_info(self, message):
        """Display information message"""
        QMessageBox.information(self, "Information", message)

    def closeEvent(self, event):
        """Handle window close event"""
        self.camera_stop_requested.emit()
        event.accept()

    def update_camera_progress(self, value):
        """Update camera initialization progress"""
        self.controls_panel.set_progress(value)
