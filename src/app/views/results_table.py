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

        # Set to fixed size policy
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(350)  # Fixed height to prevent resizing

        # Create group box
        self.group_box = QGroupBox("Detection Results")
        group_layout = QVBoxLayout()

        # Create table widget
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Class", "Confidence", "Position", "Size"]
        )

        # Set a fixed row count with a reasonable number of rows
        self.fixed_rows = 8
        self.table.setRowCount(self.fixed_rows)

        # Fix row heights
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)

        # Make the table a fixed height
        tableHeight = (
            (self.fixed_rows * 30) + self.table.horizontalHeader().height() + 2
        )
        self.table.setFixedHeight(tableHeight)

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
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        # Add table to layout
        group_layout.addWidget(self.table)

        # Add summary label with improved text wrapping
        self.summary_label = QLabel("No detections")
        self.summary_label.setFixedHeight(40)  # Increased height for wrapping
        self.summary_label.setWordWrap(True)  # Enable word wrapping
        self.summary_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        # Add specific style to control text behavior
        self.summary_label.setStyleSheet("""
            QLabel {
                padding: 2px;
                background-color: transparent;
                text-overflow: ellipsis;
            }
        """)

        # Create a container for the label with fixed width
        summary_container = QWidget()
        summary_container.setFixedHeight(44)  # Slightly larger than the label
        summary_layout = QVBoxLayout(summary_container)
        summary_layout.setContentsMargins(2, 2, 2, 2)
        summary_layout.addWidget(self.summary_label)

        group_layout.addWidget(summary_container)

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

            # Create table items
            id_item = QTableWidgetItem(track_id)
            class_item = QTableWidgetItem(class_name)
            conf_item = QTableWidgetItem(f"{confidence:.2f}")
            pos_item = QTableWidgetItem(position)
            size_item = QTableWidgetItem(size)

            # Center align all items
            for item in [id_item, class_item, conf_item, pos_item, size_item]:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

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

    # Override size methods to ensure fixed height
    def minimumSizeHint(self) -> QSize:
        """Return minimum size hint"""
        return QSize(400, 350)

    def sizeHint(self) -> QSize:
        """Return size hint"""
        return QSize(400, 350)

    def maximumSize(self) -> QSize:
        """Return maximum size"""
        return QSize(16777215, 350)  # Max width, fixed height
