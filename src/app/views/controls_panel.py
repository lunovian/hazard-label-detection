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
    QSizePolicy,
    QFormLayout,
    QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from ..models.camera_model import CameraInfo, CameraBackend
from .input_panels import LiveCameraPanel, ImageInputPanel, VideoInputPanel


class ControlsPanel(QWidget):
    """Panel for camera and detection controls with tabbed interface"""

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
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Create tab widget first
        self.tab_widget = QTabWidget()
        self.tab_widget.setProperty("mainTabs", True)  # For styling

        # Create input panels
        self.camera_panel = LiveCameraPanel()
        self.image_panel = ImageInputPanel()
        self.video_panel = VideoInputPanel()

        # Add tabs
        self.tab_widget.addTab(self.camera_panel, "Live Camera")
        self.tab_widget.addTab(self.image_panel, "Image Input")
        self.tab_widget.addTab(self.video_panel, "Video Input")

        # Add tab widget to layout
        self.layout.addWidget(self.tab_widget)

        # Create shared control groups
        self._create_shared_controls()

        # Set up timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status_animation)
        self.status_animation_value = 0

        # Connect signals
        self._connect_signals()

    def _create_shared_controls(self):
        """Create controls shared across all input types"""
        # MODEL SELECTION
        self.model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout()
        model_layout.setVerticalSpacing(8)
        model_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Model selector with refresh button
        model_select_widget = QWidget()
        model_select_layout = QHBoxLayout(model_select_widget)
        model_select_layout.setContentsMargins(0, 0, 0, 0)

        self.model_selector = QComboBox()
        self.model_selector.setToolTip("Select a YOLO model for detection")

        self.refresh_models_button = QPushButton("↻")
        self.refresh_models_button.setToolTip("Refresh model list")
        self.refresh_models_button.setProperty("iconOnly", True)
        self.refresh_models_button.setProperty("secondary", True)

        model_select_layout.addWidget(
            self.model_selector, 1
        )  # Give the combo box stretch priority
        model_select_layout.addWidget(
            self.refresh_models_button, 0
        )  # No stretch for button

        model_layout.addRow("Model:", model_select_widget)

        # Load model button
        button_container = QWidget()
        button_container_layout = QHBoxLayout(button_container)
        button_container_layout.setContentsMargins(0, 0, 0, 0)

        self.load_model_button = QPushButton("Load Selected Model")
        self.load_model_button.setProperty("primary", True)  # Add this property
        button_container_layout.addWidget(self.load_model_button)

        # Center the button by spanning both columns
        model_layout.addRow(button_container)

        self.model_group.setLayout(model_layout)
        self.layout.addWidget(self.model_group)

        # DETECTION PARAMETERS
        self.detection_group = QGroupBox("Detection Parameters")
        detection_layout = QVBoxLayout()
        detection_layout.setSpacing(8)

        # Remove the checkboxes and directly use sliders in a form layout
        slider_form = QFormLayout()
        slider_form.setVerticalSpacing(5)
        slider_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Confidence slider
        conf_widget = QWidget()
        conf_layout = QVBoxLayout(conf_widget)
        conf_layout.setContentsMargins(0, 0, 0, 0)
        conf_layout.setSpacing(2)

        conf_display = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(99)
        self.conf_slider.setValue(25)
        self.conf_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.conf_slider.setTickInterval(10)

        self.conf_label = QLabel("0.25")
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.conf_label.setMinimumWidth(40)

        conf_display.addWidget(self.conf_slider)
        conf_display.addWidget(self.conf_label)
        conf_layout.addLayout(conf_display)

        slider_form.addRow("Confidence:", conf_widget)

        # IoU slider
        iou_widget = QWidget()
        iou_layout = QVBoxLayout(iou_widget)
        iou_layout.setContentsMargins(0, 0, 0, 0)
        iou_layout.setSpacing(2)

        iou_display = QHBoxLayout()
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setMinimum(1)
        self.iou_slider.setMaximum(99)
        self.iou_slider.setValue(45)
        self.iou_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.iou_slider.setTickInterval(10)

        self.iou_label = QLabel("0.45")
        self.iou_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.iou_label.setMinimumWidth(40)

        iou_display.addWidget(self.iou_slider)
        iou_display.addWidget(self.iou_label)
        iou_layout.addLayout(iou_display)

        slider_form.addRow("IoU:", iou_widget)

        detection_layout.addLayout(slider_form)

        # Add a note about automatic detection and tracking
        auto_note = QLabel("Detection and tracking are automatically enabled")
        auto_note.setProperty("note", True)  # Use global note styling
        auto_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detection_layout.addWidget(auto_note)

        self.detection_group.setLayout(detection_layout)
        self.layout.addWidget(self.detection_group)

    def _connect_signals(self):
        """Connect all signals including tab changes"""
        # Remove camera control signals since they're now in LiveCameraPanel
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        self.iou_slider.valueChanged.connect(self._on_iou_changed)

        # Connect model selection signals
        self.model_selector.currentIndexChanged.connect(self._on_model_changed)
        self.load_model_button.clicked.connect(self._on_load_model_clicked)
        self.refresh_models_button.clicked.connect(self.refresh_models_clicked)

        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Connect panel signals
        self.camera_panel.start_btn.clicked.connect(self.camera_start_clicked)
        self.camera_panel.stop_btn.clicked.connect(self.camera_stop_clicked)
        self.camera_panel.refresh_btn.clicked.connect(self.camera_refresh_clicked)

    def _on_start_camera(self):
        """Handle start camera button click"""
        self.start_camera_button.setEnabled(False)
        self.stop_camera_button.setEnabled(True)

        # Show the progress bar and start the animation
        self.status_progress.setVisible(True)
        self.status_timer.start(100)

        # Automatically enable detection and tracking
        self.detection_toggled.emit(True)
        self.tracking_toggled.emit(True)

        self.camera_start_clicked.emit()

    def _on_stop_camera(self):
        """Handle stop camera button click"""
        self.stop_camera_button.setEnabled(False)
        self.start_camera_button.setEnabled(True)

        # Hide the progress bar and stop the animation
        self.status_progress.setVisible(False)
        self.status_timer.stop()

        # Emit the signal to stop the camera
        self.camera_stop_clicked.emit()

        # Reset status
        self.status_label.setText("Disconnected")

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
        self.camera_panel.camera_selector.clear()
        if not camera_infos:
            self.camera_panel.camera_selector.addItem("Default Camera (0)", 0)
        else:
            for camera_info in camera_infos:
                self.camera_panel.camera_selector.addItem(
                    camera_info.name, camera_info.id
                )
        self.camera_panel.camera_selector.addItem("IP/URL Camera", -1)

    def set_status_message(self, message):
        """Update status message"""
        self.camera_panel.status_label.setText(message)

    def set_connection_status(self, connected):
        """Update connection status UI"""
        self.camera_panel.status_progress.setVisible(False)
        if connected:
            self.camera_panel.status_label.setText("Connected")
            self.camera_panel.stop_btn.setEnabled(True)
            self.camera_panel.start_btn.setEnabled(False)
        else:
            self.camera_panel.status_label.setText("Disconnected")
            self.camera_panel.stop_btn.setEnabled(False)
            self.camera_panel.start_btn.setEnabled(True)

    def get_camera_id(self):
        return self.camera_panel.camera_selector.currentData()

    def get_resolution(self):
        return self.camera_panel.resolution_selector.currentData()

    def get_fps(self):
        return self.camera_panel.fps_spinner.value()

    def get_backend(self):
        return self.camera_panel.backend_selector.currentData()

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
            if not self.camera_panel.status_progress.isVisible():
                return
        elif not self.camera_panel.status_progress.isVisible():
            # Show the progress bar when value > 0
            self.camera_panel.status_progress.setVisible(True)

        # Set the progress value
        self.camera_panel.status_progress.setValue(value)

    def _on_tab_changed(self, index):
        """Handle tab changes"""
        # Stop any running capture when switching tabs
        if index != 0:  # If not on camera tab
            self.camera_stop_clicked.emit()
