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
import logging


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
        self.box_annotator = None
        self.label_annotator = None
        self.refresh_available_models()
        self.initialize_annotators()

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

    def initialize_annotators(self):
        """Initialize annotators with consistent styling"""
        try:
            # Use the correct BoxAnnotator and LabelAnnotator classes
            self.box_annotator = sv.BoxAnnotator(thickness=2)
            self.label_annotator = sv.LabelAnnotator()
        except Exception as e:
            print(f"Error initializing annotators: {e}")
            self.box_annotator = None
            self.label_annotator = None

    def detect(self, frame: np.ndarray, is_video: bool = False) -> DetectionResult:
        """Run detection on a single frame
        Args:
            frame: Input frame
            is_video: Whether this is part of a video/camera feed (for tracking)
        """
        if self.model is None:
            logging.error("No model loaded")
            return DetectionResult(frame=frame)

        try:
            frame_copy = frame.copy()
            results = self.model(
                frame_copy, conf=self.conf_threshold, iou=self.iou_threshold
            )[0]
            logging.info(f"Detection results: {len(results.boxes)} boxes found")

            # Convert YOLO results to supervision Detections
            detections = sv.Detections.from_ultralytics(results)
            logging.info(
                f"Converted to supervision detections: {len(detections) if detections else 0} detections"
            )

            # Only apply tracking if this is a video/camera feed
            if (
                is_video
                and self.tracking_enabled
                and detections
                and len(detections) > 0
            ):
                if self.byte_track_method == "update_with_detections":
                    detections = self.tracker.update_with_detections(
                        detections=detections
                    )
                    logging.info("Applied tracking with update_with_detections")
                else:
                    detections = self.tracker.update(
                        detections=detections, frame=frame_copy
                    )
                    logging.info("Applied tracking with update")

            # Create annotated frame regardless of detections
            annotated_frame = frame_copy.copy()

            # Always attempt to draw boxes and labels if we have detections
            if detections is not None and len(detections) > 0:
                try:
                    logging.info(f"Preparing to annotate {len(detections)} detections")

                    # Prepare labels
                    labels = [
                        f"{self.class_names[class_id]} {confidence:0.2f}"
                        for class_id, confidence in zip(
                            detections.class_id, detections.confidence
                        )
                    ]
                    logging.info(f"Created labels: {labels}")

                    # Draw boxes and labels
                    if self.box_annotator:
                        annotated_frame = self.box_annotator.annotate(
                            scene=annotated_frame, detections=detections
                        )
                        logging.info("Applied box annotations")

                    if self.label_annotator:
                        annotated_frame = self.label_annotator.annotate(
                            scene=annotated_frame, detections=detections, labels=labels
                        )
                        logging.info("Applied label annotations")

                except Exception as annotation_error:
                    logging.error(f"Annotation error: {str(annotation_error)}")
                    logging.error(f"Detections data: {detections}")
                    return DetectionResult(
                        frame=frame, detections=detections, annotated_frame=frame_copy
                    )

            # Always return a result with the annotated frame
            return DetectionResult(
                frame=frame,
                detections=detections,
                annotated_frame=annotated_frame,
                processing_time=results.speed.get("inference", 0),
            )

        except Exception as e:
            logging.error(f"Detection error: {str(e)}", exc_info=True)
            return DetectionResult(frame=frame)

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
