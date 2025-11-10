"""
SEG-Y Header Viewer Widget
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QTextEdit,
                              QTableWidget, QTableWidgetItem, QLabel)
from PyQt5.QtCore import Qt


class HeaderViewer(QWidget):
    """Header information viewer"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.segy_loader = None
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Header Information")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)

        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Text Header tab
        self.text_header_view = QTextEdit()
        self.text_header_view.setReadOnly(True)
        self.text_header_view.setFontFamily("Courier New")
        self.text_header_view.setStyleSheet("background-color: #f5f5f5;")
        self.tabs.addTab(self.text_header_view, "Text Header")

        # Binary Header tab
        self.binary_header_table = QTableWidget()
        self.binary_header_table.setColumnCount(2)
        self.binary_header_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.binary_header_table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.binary_header_table, "Binary Header")

        # Trace Header tab
        self.trace_header_table = QTableWidget()
        self.trace_header_table.setColumnCount(2)
        self.trace_header_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.trace_header_table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.trace_header_table, "Trace Header")

        self.current_trace_label = QLabel("No trace selected")
        layout.addWidget(self.current_trace_label)

    def set_loader(self, loader):
        """Set SEG-Y loader"""
        self.segy_loader = loader
        self.update_headers()

    def update_headers(self):
        """Update header displays"""
        if self.segy_loader is None:
            return

        # Update text header
        text_header = self.segy_loader.get_text_header()
        self.text_header_view.setPlainText(text_header)

        # Update binary header
        binary_header = self.segy_loader.get_binary_header()
        self.binary_header_table.setRowCount(len(binary_header))

        for i, (key, value) in enumerate(binary_header.items()):
            self.binary_header_table.setItem(i, 0, QTableWidgetItem(str(key)))
            self.binary_header_table.setItem(i, 1, QTableWidgetItem(str(value)))

    def show_trace_header(self, trace_idx: int):
        """Show header for specific trace"""
        if self.segy_loader is None:
            return

        trace_header = self.segy_loader.get_trace_header(trace_idx)

        self.current_trace_label.setText(f"Trace {trace_idx}")

        self.trace_header_table.setRowCount(len(trace_header))

        for i, (key, value) in enumerate(trace_header.items()):
            self.trace_header_table.setItem(i, 0, QTableWidgetItem(str(key)))
            self.trace_header_table.setItem(i, 1, QTableWidgetItem(str(value)))
