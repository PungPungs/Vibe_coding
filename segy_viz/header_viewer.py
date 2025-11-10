"""Simple header viewer dock widget."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import QTextEdit, QWidget, QVBoxLayout

from .segy_loader import SegyLoader


class HeaderViewer(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout()
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)
        self.setLayout(layout)

    def set_loader(self, loader: SegyLoader) -> None:
        info = loader.get_header_summary()
        self.text.setPlainText(info)
