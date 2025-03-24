import cv2
import supervision as sv
from ultralytics import YOLO
import time
import numpy as np

# Configuration
CONFIDENCE_THRESHOLD = 0.5
RESOLUTION_PRESETS = {"HD": (1280, 720), "Full HD": (1920, 1080), "SD": (640, 480)}


class CameraManager:
    def __init__(self):
        self.available_cameras = self.get_available_cameras()
        self.current_camera = 0
        self.resolution = RESOLUTION_PRESETS["HD"]
        self.brightness = 100
        self.contrast = 100

    def get_available_cameras(self):
        cameras = []
        for i in range(5):  # Check first 5 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(i)
                cap.release()
        return cameras

    def switch_camera(self):
        self.current_camera = (self.current_camera + 1) % len(self.available_cameras)
        return self.available_cameras[self.current_camera]


class UI:
    def __init__(self):
        self.show_fps = True
        self.show_controls = True
        self.filter_mode = 0  # 0: None, 1: Enhance, 2: Grayscale

    def draw_status_bar(self, frame, fps, camera_id, detection_count):
        # Draw semi-transparent status bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Add status information
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Camera: {camera_id}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Detections: {detection_count}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Draw controls help
        if self.show_controls:
            controls = [
                "Q/ESC: Quit",
                "C: Switch Camera",
                "F: Toggle FPS",
                "H: Hide Controls",
                "E: Enhanced Mode",
                "G: Grayscale Mode",
            ]
            for i, control in enumerate(controls):
                cv2.putText(
                    frame,
                    control,
                    (frame.shape[1] - 200, 30 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

    def process_frame(self, frame):
        if self.filter_mode == 1:  # Enhanced mode
            frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
        elif self.filter_mode == 2:  # Grayscale
            frame = cv2.cvtColor(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
            )
        return frame


# FPS counter class
class FPS:
    def __init__(self):
        self.ptime = 0

    def update(self, img):
        ctime = time.time()
        fps = 1 / (ctime - self.ptime) if self.ptime != 0 else 0
        self.ptime = ctime
        cv2.putText(
            img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2
        )
        return img


# Load YOLO model
model = YOLO(r"D:\Github\hazard-label-dataset\yolo12.pt")

# Initialize annotators with correct settings
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator()

# Initialize custom FPS counter
fps_counter = FPS()

# Initialize camera manager and UI
camera_manager = CameraManager()
ui = UI()

try:
    cap = cv2.VideoCapture(camera_manager.current_camera)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Switching cameras...")
            new_camera = camera_manager.switch_camera()
            cap = cv2.VideoCapture(new_camera)
            continue

        # Process frame with selected filter
        frame = ui.process_frame(frame)

        # Run YOLOv8 inference with confidence threshold
        result = model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Draw bounding boxes and labels with confidence scores
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in detections
        ]

        annotated_image = box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

        # Update FPS counter
        annotated_image = fps_counter.update(annotated_image)

        # Update UI
        ctime = time.time()
        fps = 1 / (ctime - fps_counter.ptime) if fps_counter.ptime != 0 else 0
        fps_counter.ptime = ctime

        ui.draw_status_bar(
            annotated_image, fps, camera_manager.current_camera, len(detections)
        )

        # Show frame
        cv2.imshow("YOLOv8 Detection", annotated_image)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), 27]:  # q or ESC
            break
        elif key == ord("c"):  # Switch camera
            new_camera = camera_manager.switch_camera()
            cap = cv2.VideoCapture(new_camera)
        elif key == ord("f"):  # Toggle FPS
            ui.show_fps = not ui.show_fps
        elif key == ord("h"):  # Toggle controls
            ui.show_controls = not ui.show_controls
        elif key == ord("e"):  # Toggle enhanced mode
            ui.filter_mode = 1 if ui.filter_mode != 1 else 0
        elif key == ord("g"):  # Toggle grayscale
            ui.filter_mode = 2 if ui.filter_mode != 2 else 0

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    if "cap" in locals():
        cap.release()
    cv2.destroyAllWindows()
