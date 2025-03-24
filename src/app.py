import sys
from PyQt6.QtWidgets import QApplication
from app.views.main_window import MainWindow
from app.controllers.app_controller import AppController
from app.models.detection_model import DetectionModel


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("GHS Hazard Label Detector")
    app.setStyle("Fusion")  # Use Fusion style for consistent cross-platform look

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
