from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLabel,
    QGroupBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
import supervision as sv
from typing import Dict, List, Optional


class ResultsTable(QWidget):
    """Widget for displaying detection results in a table"""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)

        # Create group box
        self.group_box = QGroupBox("Detection Results")
        group_layout = QVBoxLayout()

        # Create table widget
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Class", "Confidence", "Position", "Size"]
        )

        # Set table properties
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        # Add table to layout
        group_layout.addWidget(self.table)

        # Add summary label
        self.summary_label = QLabel("No detections")
        group_layout.addWidget(self.summary_label)

        self.group_box.setLayout(group_layout)
        self.layout.addWidget(self.group_box)

    def update_detections(
        self, detections: Optional[sv.Detections], class_names: Dict[int, str]
    ):
        """Update table with new detection results"""
        self.table.setRowCount(0)  # Clear existing rows

        if detections is None or len(detections) == 0:
            self.summary_label.setText("No detections")
            return

        # Add a row for each detection
        self.table.setRowCount(len(detections))

        class_counts = {}

        for i in range(len(detections)):
            # Get detection data
            class_id = (
                int(detections.class_id[i]) if detections.class_id is not None else -1
            )
            class_name = class_names.get(class_id, "Unknown")
            confidence = (
                detections.confidence[i] if detections.confidence is not None else 0.0
            )

            # Get bounding box
            box = detections.xyxy[i]
            x1, y1, x2, y2 = box
            position = f"({int(x1)}, {int(y1)})"
            size = f"{int(x2 - x1)}×{int(y2 - y1)}"

            # Get tracking ID if available
            track_id = "N/A"
            if hasattr(detections, "tracker_id") and detections.tracker_id is not None:
                track_id = str(detections.tracker_id[i])

            # Keep track of class counts
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

            # Create table items
            id_item = QTableWidgetItem(track_id)
            class_item = QTableWidgetItem(class_name)
            conf_item = QTableWidgetItem(f"{confidence:.2f}")
            pos_item = QTableWidgetItem(position)
            size_item = QTableWidgetItem(size)

            # Center align items
            for item in [id_item, class_item, conf_item, pos_item, size_item]:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Set row items
            self.table.setItem(i, 0, id_item)
            self.table.setItem(i, 1, class_item)
            self.table.setItem(i, 2, conf_item)
            self.table.setItem(i, 3, pos_item)
            self.table.setItem(i, 4, size_item)

        # Update summary
        summary_text = f"Total: {len(detections)} detection"
        if len(detections) != 1:
            summary_text += "s"

        # Add class breakdown
        if class_counts:
            summary_text += " ("
            class_parts = []
            for class_name, count in class_counts.items():
                class_parts.append(f"{class_name}: {count}")
            summary_text += ", ".join(class_parts)
            summary_text += ")"

        self.summary_label.setText(summary_text)

    def clear(self):
        """Clear the results table"""
        self.table.setRowCount(0)
        self.summary_label.setText("No detections")
