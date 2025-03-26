from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLabel,
    QGroupBox,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QFontMetrics
import supervision as sv
from typing import Dict, List, Optional


class ResultsTable(QWidget):
    """Widget for displaying detection results in a table"""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(8)

        # Use a more flexible size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Create group box with better spacing
        self.group_box = QGroupBox("Detection Results")
        self.group_box.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        group_layout = QVBoxLayout()
        group_layout.setContentsMargins(8, 12, 8, 8)
        group_layout.setSpacing(8)

        # Create table widget with better styling
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Class", "Confidence", "Position", "Size"]
        )

        # Disable alternating row colors
        self.table.setAlternatingRowColors(False)

        # Set a reasonable row count
        self.fixed_rows = 8
        self.table.setRowCount(self.fixed_rows)

        # Set better row heights
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table.verticalHeader().setVisible(
            False
        )  # Hide row numbers for cleaner look

        # Make the table expand with the window
        self.table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Set column properties - make Class column wider
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )  # ID
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )  # Class
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )  # Confidence
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )  # Position
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )  # Size

        # Enable alternating row colors and row selection
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        # Add table to layout
        group_layout.addWidget(self.table)

        # Add summary label with improved styling
        self.summary_label = QLabel("No detections")
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.summary_label.setMinimumHeight(36)
        self.summary_label.setMaximumHeight(48)
        self.summary_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        group_layout.addWidget(self.summary_label)
        self.group_box.setLayout(group_layout)
        self.layout.addWidget(self.group_box)

        # Initialize with empty rows
        self.clear()

    def update_detections(
        self, detections: Optional[sv.Detections], class_names: Dict[int, str]
    ):
        """Update table with new detection results"""
        # Clear all cells but keep rows
        for row in range(self.fixed_rows):
            for col in range(5):
                self.table.setItem(row, col, QTableWidgetItem(""))

        if detections is None or len(detections) == 0:
            self.summary_label.setText("No detections")
            return

        # Number of detections to display (up to fixed_rows)
        display_count = min(len(detections), self.fixed_rows)

        # Track class counts for summary
        class_counts = {}

        for i in range(display_count):
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
                if (
                    i < len(detections.tracker_id)
                    and detections.tracker_id[i] is not None
                ):
                    track_id = str(detections.tracker_id[i])

            # Keep track of class counts
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

            # Create table items with improved styling
            id_item = QTableWidgetItem(track_id)
            class_item = QTableWidgetItem(class_name)
            conf_item = QTableWidgetItem(f"{confidence:.2f}")
            pos_item = QTableWidgetItem(position)
            size_item = QTableWidgetItem(size)

            # Center align all items
            for item in [id_item, class_item, conf_item, pos_item, size_item]:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Add a special style for high confidence items (> 0.7)
            if confidence > 0.7:
                conf_item.setForeground(QColor("#27ae60"))  # Success green
                # Fix the font issue - don't try to get font from QFontMetrics
                font = conf_item.font()
                font.setBold(True)  # Make high confidence values bold
                conf_item.setFont(font)
            elif confidence < 0.4:
                conf_item.setForeground(QColor("#e74c3c"))  # Warning red

            # Set row items
            self.table.setItem(i, 0, id_item)
            self.table.setItem(i, 1, class_item)
            self.table.setItem(i, 2, conf_item)
            self.table.setItem(i, 3, pos_item)
            self.table.setItem(i, 4, size_item)

        # Create a simple text summary with very short class names to prevent layout issues
        summary_text = f"Total: {len(detections)} detection"
        if len(detections) != 1:
            summary_text += "s"

        # Add indicator if some detections aren't shown
        if len(detections) > self.fixed_rows:
            summary_text += f" (showing {self.fixed_rows} of {len(detections)})"

        # Add class breakdown with compact format and severe truncation if needed
        if class_counts:
            summary_text += (
                " | "  # Use vertical bar instead of parentheses for compactness
            )
            class_parts = []

            # Calculate maximum safe length for the class names portion
            font_metrics = QFontMetrics(self.summary_label.font())
            max_text_width = self.width() - 40  # Allow some margin

            # Add classes one by one, checking total width
            for class_name, count in class_counts.items():
                # Aggressively truncate long class names
                display_name = class_name
                if len(class_name) > 12:  # Even more aggressive truncation
                    display_name = class_name[:10] + ".."

                current_part = f"{display_name}: {count}"
                class_parts.append(current_part)

                # Check if adding all parts would exceed width
                test_text = summary_text + ", ".join(class_parts)
                if font_metrics.horizontalAdvance(test_text) > max_text_width:
                    # If too long, replace the last part with "..."
                    if len(class_parts) > 1:
                        class_parts.pop()  # Remove last item
                        class_parts.append("...")
                    break

            summary_text += ", ".join(class_parts)

        self.summary_label.setText(summary_text)

    def clear(self):
        """Clear the results table"""
        for row in range(self.fixed_rows):
            for col in range(5):
                self.table.setItem(row, col, QTableWidgetItem(""))
        self.summary_label.setText("No detections")

    # Modify size methods to allow vertical expansion
    def minimumSizeHint(self) -> QSize:
        """Return minimum size hint"""
        return QSize(350, 200)  # Smaller minimum to allow more flexibility

    def sizeHint(self) -> QSize:
        """Return size hint"""
        return QSize(450, 300)  # Preferred size, not fixed

    def maximumSize(self) -> QSize:
        """Return maximum size"""
        return QSize(16777215, 16777215)  # Max width, max height to allow expansion
