import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import supervision as sv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from ..utils.model_utils import ModelManager
import inspect


@dataclass
class DetectionResult:
    """Container for detection results"""

    frame: np.ndarray
    detections: Optional[sv.Detections] = None
    annotated_frame: Optional[np.ndarray] = None
    processing_time: float = 0.0


class DetectionModel:
    """Handles YOLO model loading and inference for hazard label detection"""

    DEFAULT_CONF_THRESHOLD = 0.25
    DEFAULT_IOU_THRESHOLD = 0.45

    def __init__(self):
        self.model = None
        self.tracker = None
        self.model_path = None  # Changed: don't set a default model path
        self.conf_threshold = self.DEFAULT_CONF_THRESHOLD
        self.iou_threshold = self.DEFAULT_IOU_THRESHOLD
        self.class_names = []
        self.tracking_enabled = True
        self.annotator = None
        self.available_models = []
        self.refresh_available_models()

        # Initialize supervision version compatibility helpers
        self._init_supervision_compatibility()

    def _init_supervision_compatibility(self):
        """Initialize compatibility helpers for supervision library API changes"""
        # Check if ColorPalette has a default() method
        self.has_color_palette_default = hasattr(sv.ColorPalette, "default")

        # Check BoxAnnotator API version - if it accepts labels keyword
        try:
            box_annotator = sv.BoxAnnotator(thickness=2)
            sig = inspect.signature(box_annotator.annotate)
            self.box_annotator_supports_labels = "labels" in sig.parameters
        except:
            self.box_annotator_supports_labels = True  # Default to older API

        # Check ByteTrack API version
        if hasattr(sv, "ByteTrack"):
            try:
                tracker = sv.ByteTrack()
                if hasattr(tracker, "update_with_detections"):
                    self.byte_track_method = "update_with_detections"
                elif hasattr(tracker, "update"):
                    self.byte_track_method = "update"
                elif hasattr(tracker, "track_objects"):
                    self.byte_track_method = "track_objects"
                else:
                    self.byte_track_method = (
                        "update_with_detections"  # Default to newer API
                    )
            except:
                # Fallback if we couldn't instantiate
                self.byte_track_method = (
                    "update_with_detections"  # Default to newer API
                )
        else:
            self.byte_track_method = "update"  # Default to older API

    def refresh_available_models(self):
        """Refresh the list of available models"""
        self.available_models = ModelManager.get_available_models()
        return self.available_models

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load YOLO model from specified path or default path"""
        try:
            path = model_path if model_path else self.model_path
            if not os.path.exists(path):
                print(f"Model not found at {path}")
                return False

            # Load the model
            self.model = YOLO(path)
            self.model_path = path
            self.class_names = self.model.names

            # Initialize tracker
            self.tracker = sv.ByteTrack()

            # Initialize annotator with compatible color palette
            # Handle annotation API differences
            try:
                # Create a simple color list instead of trying to use ColorPalette
                colors = self._create_default_color_palette()

                # Simplified annotator creation - use only parameters known to work
                self.annotator = sv.BoxAnnotator(thickness=2)

                # Store colors for manual labeling
                self.label_colors = colors

            except Exception as e:
                print(f"Error setting up annotator: {e}")
                # Ultimate fallback
                self.annotator = sv.BoxAnnotator(thickness=2)
                self.label_colors = [(0, 0, 255)]  # Default to red

            # Save this as the last used model
            ModelManager.save_last_model(path)

            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def _create_default_color_palette(self):
        """Create a default color palette for compatibility with older supervision versions"""
        # Define default colors similar to supervision's default palette
        # Using BGR format for OpenCV compatibility
        colors = [
            (0, 0, 255),  # Red (BGR)
            (0, 255, 0),  # Green (BGR)
            (255, 0, 0),  # Blue (BGR)
            (0, 255, 255),  # Yellow (BGR)
            (255, 0, 255),  # Magenta (BGR)
            (255, 255, 0),  # Cyan (BGR)
            (0, 128, 255),  # Orange (BGR)
            (255, 0, 128),  # Purple (BGR)
            (128, 255, 0),  # Mint (BGR)
            (128, 0, 255),  # Pink (BGR)
        ]

        # For supervision version compatibility, just return the color list
        # Don't try to create ColorPalette object which might fail
        return colors

    def _format_label(self, detection_index: int, detections: sv.Detections) -> str:
        """Format detection label with class name and confidence"""
        class_id = int(detections.class_id[detection_index])
        confidence = detections.confidence[detection_index]
        class_name = self.class_names[class_id]
        return f"{class_name} {confidence:.2f}"

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a single frame"""
        if self.model is None:
            return DetectionResult(frame=frame)

        # Create a copy of the frame for processing
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)[0]

        # Convert YOLO results to supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # Apply tracking if enabled
        if self.tracking_enabled and detections and len(detections) > 0:
            # Use the appropriate ByteTrack method based on the version
            if self.byte_track_method == "update":
                # Older API
                detections = self.tracker.update(detections=detections, frame=frame)
            elif self.byte_track_method == "update_with_detections":
                # Current API
                detections = self.tracker.update_with_detections(detections=detections)
            else:
                # Newer API with track_objects
                detections = self.tracker.track_objects(
                    detections=detections, frame=frame
                )

        # Annotate the frame
        if detections and len(detections) > 0:
            labels = [self._format_label(i, detections) for i in range(len(detections))]
            try:
                # Create a copy of the frame for annotation
                annotated_frame = frame.copy()

                # First draw all bounding boxes without labels
                try:
                    annotated_frame = self.annotator.annotate(
                        scene=annotated_frame, detections=detections
                    )
                except Exception as e:
                    print(f"Box annotation error: {e}")
                    # If box annotation fails, just use the original frame

                # Then manually add labels to each box
                for i, label in enumerate(labels):
                    if i < len(detections.xyxy):
                        # Get box coordinates
                        box = detections.xyxy[i].astype(int)
                        x1, y1, x2, y2 = box

                        # Get color for this detection (by class or index)
                        if (
                            hasattr(detections, "class_id")
                            and len(detections.class_id) > i
                        ):
                            color_idx = int(detections.class_id[i]) % len(
                                self.label_colors
                            )
                        else:
                            color_idx = i % len(self.label_colors)

                        color = self.label_colors[color_idx]

                        # Draw text background
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, font, font_scale, thickness
                        )

                        # Draw background rectangle for text
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1 - text_height - 5),
                            (x1 + text_width + 5, y1),
                            color,  # Use same color as box
                            -1,  # Filled rectangle
                        )

                        # Draw text
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 5),
                            font,
                            font_scale,
                            (255, 255, 255),  # White text
                            thickness,
                        )

            except Exception as e:
                print(f"Annotation error: {e}")
                # Fallback to original frame if annotation fails
                annotated_frame = frame.copy()
        else:
            annotated_frame = frame.copy()

        return DetectionResult(
            frame=frame,
            detections=detections,
            annotated_frame=annotated_frame,
            processing_time=results.speed.get("inference", 0),
        )

    def set_conf_threshold(self, value: float):
        """Set confidence threshold"""
        self.conf_threshold = value

    def set_iou_threshold(self, value: float):
        """Set IoU threshold"""
        self.iou_threshold = value

    def toggle_tracking(self, enabled: bool):
        """Enable or disable tracking"""
        self.tracking_enabled = enabled
        if enabled and self.tracker is None:
            self.tracker = sv.ByteTrack()
