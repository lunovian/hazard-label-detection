from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QSizePolicy,
)


class StyledHelpDialog(QDialog):
    """Styled help dialog that matches the application theme"""

    def __init__(self, parent=None, title="Help", content="", width=600, height=400):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(width, height)
        self.setObjectName("helpDialog")

        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Add content browser
        self.text_browser = QTextBrowser()
        self.text_browser.setHtml(content)
        self.text_browser.setOpenExternalLinks(True)
        layout.addWidget(self.text_browser)

        # Add button row
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)

        self.close_button = QPushButton("Close")
        self.close_button.setObjectName("closeButton")
        self.close_button.setFixedWidth(100)
        self.close_button.clicked.connect(self.accept)

        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)


class StyledConfirmDialog(QDialog):
    """Styled confirmation dialog that matches the application theme"""

    def __init__(self, parent=None, title="Confirm", message="", width=400, height=150):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(width, height)
        self.setObjectName("confirmDialog")

        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # Add message
        self.message_label = QLabel(message)
        self.message_label.setWordWrap(True)
        self.message_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.message_label)

        # Add button row
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setFixedWidth(100)
        self.cancel_button.clicked.connect(self.reject)

        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.setObjectName("confirmButton")
        self.confirm_button.setFixedWidth(100)
        self.confirm_button.clicked.connect(self.accept)

        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.confirm_button)
        layout.addLayout(button_layout)


def show_styled_help(parent, title, content, width=600, height=400):
    """Show a styled help dialog"""
    dialog = StyledHelpDialog(parent, title, content, width, height)
    dialog.exec()


def show_styled_confirmation(parent, title, message, width=400, height=150):
    """Show a styled confirmation dialog and return True if confirmed"""
    dialog = StyledConfirmDialog(parent, title, message, width, height)
    result = dialog.exec()
    return result == QDialog.DialogCode.Accepted
