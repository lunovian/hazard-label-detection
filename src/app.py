import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from app.views.main_window import MainWindow
from app.controllers.app_controller import AppController
from app.models.detection_model import DetectionModel


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("GHS Hazard Label Detector")

    # Set application icon
    icon_path = os.path.join(
        os.path.dirname(__file__), "app", "resources", "icons", "app_icon.png"
    )
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    app.setStyle("windows11")

    # Initialize model, view, and controller
    detection_model = DetectionModel()
    main_window = MainWindow()
    controller = AppController(model=detection_model, view=main_window)

    # Show the main window
    main_window.show()

    # Start the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
