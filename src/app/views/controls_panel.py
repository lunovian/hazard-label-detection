from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QCheckBox,
    QSpinBox,
    QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from ..models.camera_model import CameraInfo, CameraBackend


class ControlsPanel(QWidget):
    """Panel for camera and detection controls"""

    # Define signals
    camera_start_clicked = pyqtSignal()
    camera_stop_clicked = pyqtSignal()
    camera_refresh_clicked = pyqtSignal()
    detection_toggled = pyqtSignal(bool)
    tracking_toggled = pyqtSignal(bool)
    confidence_changed = pyqtSignal(float)
    iou_changed = pyqtSignal(float)
    backend_changed = pyqtSignal(CameraBackend)
    model_selected = pyqtSignal(str)
    refresh_models_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)

        # Create camera controls group
        self.camera_group = QGroupBox("Camera Controls")
        camera_layout = QVBoxLayout()

        # Camera selection
        camera_select_layout = QHBoxLayout()
        camera_select_layout.addWidget(QLabel("Camera:"))
        self.camera_selector = QComboBox()
        self.camera_selector.addItem("Default Camera", 0)
        camera_select_layout.addWidget(self.camera_selector)

        # Add refresh button
        self.refresh_camera_button = QPushButton("🔄")
        self.refresh_camera_button.setToolTip("Refresh camera list")
        self.refresh_camera_button.setMaximumWidth(30)
        camera_select_layout.addWidget(self.refresh_camera_button)

        camera_layout.addLayout(camera_select_layout)

        # Camera backend selection
        backend_layout = QHBoxLayout()
        backend_layout.addWidget(QLabel("Backend:"))
        self.backend_selector = QComboBox()
        self.backend_selector.addItem("Auto/Default", CameraBackend.ANY)
        self.backend_selector.addItem("DirectShow (Windows)", CameraBackend.DSHOW)
        self.backend_selector.addItem("Media Foundation (Windows)", CameraBackend.MSMF)
        # Fix: Changed V4 to V4L to match the correct enum value
        self.backend_selector.addItem("V4L2 (Linux)", CameraBackend.V4L)
        self.backend_selector.addItem("GStreamer", CameraBackend.GSTREAMER)
        backend_layout.addWidget(self.backend_selector)
        camera_layout.addLayout(backend_layout)

        # Resolution selection
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Resolution:"))
        self.resolution_selector = QComboBox()
        self.resolution_selector.addItem("640x480", (640, 480))
        self.resolution_selector.addItem("800x600", (800, 600))
        self.resolution_selector.addItem("1280x720", (1280, 720))
        self.resolution_selector.addItem("1920x1080", (1920, 1080))
        resolution_layout.addWidget(self.resolution_selector)
        camera_layout.addLayout(resolution_layout)

        # FPS control
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_spinner = QSpinBox()
        self.fps_spinner.setMinimum(1)
        self.fps_spinner.setMaximum(60)
        self.fps_spinner.setValue(30)
        fps_layout.addWidget(self.fps_spinner)
        camera_layout.addLayout(fps_layout)

        # Status indicator with meaningful progress
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        status_layout.addWidget(self.status_label)

        # Progress bar that actually shows the initialization progress
        self.status_progress = QProgressBar()
        self.status_progress.setMaximumHeight(15)
        self.status_progress.setTextVisible(True)  # Changed to show percentage
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_progress.setVisible(False)
        status_layout.addWidget(self.status_progress)

        camera_layout.addLayout(status_layout)

        # Camera buttons
        button_layout = QHBoxLayout()
        self.start_camera_button = QPushButton("Start Camera")
        self.stop_camera_button = QPushButton("Stop Camera")
        self.stop_camera_button.setEnabled(False)
        button_layout.addWidget(self.start_camera_button)
        button_layout.addWidget(self.stop_camera_button)
        camera_layout.addLayout(button_layout)

        self.camera_group.setLayout(camera_layout)
        self.layout.addWidget(self.camera_group)

        # Create model selection group
        self.model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()

        # Model selector
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Model:"))
        self.model_selector = QComboBox()
        self.model_selector.setToolTip("Select a YOLO model for detection")
        model_select_layout.addWidget(self.model_selector)

        # Add refresh button for models
        self.refresh_models_button = QPushButton("🔄")
        self.refresh_models_button.setToolTip("Refresh model list")
        self.refresh_models_button.setMaximumWidth(30)
        model_select_layout.addWidget(self.refresh_models_button)

        model_layout.addLayout(model_select_layout)

        # Load model button
        self.load_model_button = QPushButton("Load Selected Model")
        model_layout.addWidget(self.load_model_button)

        self.model_group.setLayout(model_layout)
        self.layout.addWidget(self.model_group)

        # Create detection controls group
        self.detection_group = QGroupBox("Detection Controls")
        detection_layout = QVBoxLayout()

        # Enable detection checkbox
        self.detection_checkbox = QCheckBox("Enable Detection")
        self.detection_checkbox.setChecked(True)
        detection_layout.addWidget(self.detection_checkbox)

        # Enable tracking checkbox
        self.tracking_checkbox = QCheckBox("Enable Tracking")
        self.tracking_checkbox.setChecked(True)
        detection_layout.addWidget(self.tracking_checkbox)

        # Confidence threshold slider
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.conf_label = QLabel("0.25")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(99)
        self.conf_slider.setValue(25)
        self.conf_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.conf_slider.setTickInterval(10)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        detection_layout.addLayout(conf_layout)

        # IoU threshold slider
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU:"))
        self.iou_label = QLabel("0.45")
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setMinimum(1)
        self.iou_slider.setMaximum(99)
        self.iou_slider.setValue(45)
        self.iou_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.iou_slider.setTickInterval(10)
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_label)
        detection_layout.addLayout(iou_layout)

        self.detection_group.setLayout(detection_layout)
        self.layout.addWidget(self.detection_group)

        # Set up the status animation timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status_animation)
        self.status_animation_value = 0

        # Connect signals
        self._connect_signals()

    def _connect_signals(self):
        """Connect internal signals"""
        self.start_camera_button.clicked.connect(self._on_start_camera)
        self.stop_camera_button.clicked.connect(self._on_stop_camera)
        self.refresh_camera_button.clicked.connect(self.camera_refresh_clicked)
        self.detection_checkbox.toggled.connect(self.detection_toggled)
        self.tracking_checkbox.toggled.connect(self.tracking_toggled)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        self.iou_slider.valueChanged.connect(self._on_iou_changed)
        self.backend_selector.currentIndexChanged.connect(self._on_backend_changed)

        # Connect model selection signals
        self.model_selector.currentIndexChanged.connect(self._on_model_changed)
        self.load_model_button.clicked.connect(self._on_load_model_clicked)
        self.refresh_models_button.clicked.connect(self.refresh_models_clicked)

    def _on_start_camera(self):
        """Handle start camera button click"""
        self.start_camera_button.setEnabled(False)
        self.stop_camera_button.setEnabled(True)

        # Show the progress bar and start the animation
        self.status_progress.setVisible(True)
        self.status_timer.start(100)

        self.camera_start_clicked.emit()

    def _on_stop_camera(self):
        """Handle stop camera button click"""
        self.stop_camera_button.setEnabled(False)
        self.start_camera_button.setEnabled(True)

        # Hide the progress bar and stop the animation
        self.status_progress.setVisible(False)
        self.status_timer.stop()

        self.camera_stop_clicked.emit()

    def _on_conf_changed(self, value):
        """Handle confidence slider change"""
        conf_value = value / 100.0
        self.conf_label.setText(f"{conf_value:.2f}")
        self.confidence_changed.emit(conf_value)

    def _on_iou_changed(self, value):
        """Handle IoU slider change"""
        iou_value = value / 100.0
        self.iou_label.setText(f"{iou_value:.2f}")
        self.iou_changed.emit(iou_value)

    def _on_backend_changed(self, index):
        """Handle backend selection change"""
        backend = self.backend_selector.currentData()
        self.backend_changed.emit(backend)

    def _update_status_animation(self):
        """Update the status animation during camera connection"""
        self.status_animation_value = (self.status_animation_value + 10) % 100
        self.status_progress.setValue(self.status_animation_value)

    def set_camera_list(self, camera_infos):
        """Update camera selection combobox with available cameras"""
        self.camera_selector.clear()
        if not camera_infos:
            # Add fallback option if no cameras detected
            self.camera_selector.addItem("Default Camera (0)", 0)
        else:
            for camera_info in camera_infos:
                self.camera_selector.addItem(camera_info.name, camera_info.id)

        # Add IP/URL camera option
        self.camera_selector.addItem("IP/URL Camera", -1)  # Special ID for URL camera

    def set_status_message(self, message):
        """Update status message"""
        self.status_label.setText(message)

    def set_connection_status(self, connected):
        """Update connection status UI"""
        if connected:
            # Camera is connected, show ready state
            self.status_progress.setVisible(False)
            self.status_timer.stop()
            self.status_label.setText("Connected")
            self.stop_camera_button.setEnabled(True)
            self.start_camera_button.setEnabled(False)
        else:
            # Camera is disconnected
            self.status_progress.setVisible(False)
            self.status_timer.stop()
            self.status_label.setText("Disconnected")
            self.stop_camera_button.setEnabled(False)
            self.start_camera_button.setEnabled(True)

    def get_camera_id(self):
        """Get selected camera ID"""
        return self.camera_selector.currentData()

    def get_resolution(self):
        """Get selected resolution"""
        return self.resolution_selector.currentData()

    def get_fps(self):
        """Get selected FPS"""
        return self.fps_spinner.value()

    def get_backend(self):
        """Get selected camera backend"""
        return self.backend_selector.currentData()

    def _on_model_changed(self, index):
        """Handle model selection change"""
        if index >= 0 and self.model_selector.itemData(index):
            model_path = self.model_selector.itemData(index)
            # Signal is emitted when Load button is clicked

    def _on_load_model_clicked(self):
        """Handle load model button click"""
        index = self.model_selector.currentIndex()
        if index >= 0 and self.model_selector.itemData(index):
            model_path = self.model_selector.itemData(index)
            self.model_selected.emit(model_path)

    def set_model_list(self, models):
        """Update model selection combobox with available models"""
        current_text = self.model_selector.currentText()
        self.model_selector.clear()

        if not models:
            self.model_selector.addItem("No models found", None)
            self.load_model_button.setEnabled(False)
        else:
            for file_path, display_name in models:
                self.model_selector.addItem(display_name, file_path)
            self.load_model_button.setEnabled(True)

            # Try to restore previous selection
            if current_text:
                index = self.model_selector.findText(current_text)
                if index >= 0:
                    self.model_selector.setCurrentIndex(index)

    def set_current_model(self, model_path):
        """Set the current model in the dropdown"""
        for i in range(self.model_selector.count()):
            if self.model_selector.itemData(i) == model_path:
                self.model_selector.setCurrentIndex(i)
                break

    def set_progress(self, value):
        """Update connection progress"""
        if value == 0:
            # Hide if reset to zero
            if not self.status_progress.isVisible():
                return
        elif not self.status_progress.isVisible():
            # Show the progress bar when value > 0
            self.status_progress.setVisible(True)

        # Set the progress value
        self.status_progress.setValue(value)
