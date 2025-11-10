"""RadExPro-style SEG-Y viewer main window."""

from __future__ import annotations

import pathlib
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QAction,
    QWidget,
    QHBoxLayout,
    QDockWidget,
)

from .control_panel import ControlPanel
from .display_modes import DisplayMode
from .colormap import ColormapType
from .gl_display import SeismicDisplay
from .header_viewer import HeaderViewer
from .segy_loader import SegyLoader


class MainWindow(QMainWindow):
    """Professional SEG-Y Viewer Main Window."""

    def __init__(self) -> None:
        super().__init__()
        self.loader = SegyLoader()
        self.max_traces: Optional[int] = None
        self.initial_colormap: Optional[ColormapType] = None
        self.initial_mode: Optional[DisplayMode] = None
        self.control_panel: Optional[ControlPanel] = None
        self.display_widget: Optional[SeismicDisplay] = None
        self.header_viewer: Optional[HeaderViewer] = None
        self.status_label: Optional[QLabel] = None
        self.info_label: Optional[QLabel] = None
        self.mouse_pos_label: Optional[QLabel] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle("RadEx Style SEG-Y Viewer")
        self.setGeometry(50, 50, 1600, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        self.control_panel = ControlPanel()
        self.control_panel.setMaximumWidth(360)
        splitter.addWidget(self.control_panel)

        self.display_widget = SeismicDisplay()
        self.display_widget.setMinimumWidth(800)
        splitter.addWidget(self.display_widget)
        splitter.setSizes([320, 1100])

        self.header_viewer = HeaderViewer()
        dock = QDockWidget("Header Information", self)
        dock.setWidget(self.header_viewer)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self._build_toolbar()
        self._build_statusbar()
        self._connect_signals()
        self._apply_stylesheet()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("Open SEG-Y", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)

        export_action = QAction("Export Image", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_image)
        toolbar.addAction(export_action)

        toolbar.addSeparator()

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        toolbar.addAction(about_action)

    def _build_statusbar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        self.status_label = QLabel("Ready")
        status.addWidget(self.status_label)
        self.info_label = QLabel("")
        status.addPermanentWidget(self.info_label)
        self.mouse_pos_label = QLabel("Trace: --, Sample: --")
        status.addPermanentWidget(self.mouse_pos_label)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        assert self.control_panel is not None
        assert self.display_widget is not None

        self.control_panel.display_mode_changed.connect(self.display_widget.set_display_mode)
        self.control_panel.colormap_changed.connect(self.display_widget.set_colormap)
        self.control_panel.agc_changed.connect(self.display_widget.set_agc)
        self.control_panel.clip_changed.connect(self.display_widget.set_clip_percentile)
        self.control_panel.wiggle_amplitude_changed.connect(self.display_widget.set_wiggle_amplitude)
        self.control_panel.reset_view_requested.connect(self.display_widget.reset_view)

        self.display_widget.mouse_position_changed.connect(self._on_mouse_position)
        self.display_widget.trace_selected.connect(self._on_trace_selected)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def open_file(self, path: Optional[pathlib.Path] = None) -> None:
        if path is None:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Open SEG-Y File",
                "",
                "SEG-Y Files (*.sgy *.segy);;All Files (*)",
            )
            if not filename:
                return
            path = pathlib.Path(filename)

        path = path.resolve()
        if not path.exists():
            QMessageBox.critical(self, "Error", f"SEG-Y file not found:\n{path}")
            return

        assert self.display_widget is not None
        assert self.header_viewer is not None
        assert self.status_label is not None
        assert self.info_label is not None

        self.status_label.setText("Loading SEG-Y file...")
        QApplication.processEvents()

        if self.max_traces is not None:
            self.loader.max_traces = self.max_traces

        if not self.loader.load(str(path)):
            QMessageBox.critical(self, "Error", "Failed to load SEG-Y file")
            self.status_label.setText("Failed to load file")
            return

        data = self.loader.get_data()
        num_samples, num_traces = self.loader.get_dimensions()
        sample_rate = self.loader.get_sample_rate()

        self.display_widget.set_data(data)
        if self.initial_colormap:
            self.display_widget.set_colormap(self.initial_colormap)
        if self.initial_mode:
            self.display_widget.set_display_mode(self.initial_mode)
        self.header_viewer.set_loader(self.loader)
        self.info_label.setText(
            f"Traces: {num_traces} | Samples: {num_samples} | Rate: {sample_rate * 1000:.2f} ms"
        )
        self.status_label.setText(f"Loaded: {path}")

    def set_max_traces(self, value: Optional[int]) -> None:
        self.max_traces = value

    def set_default_colormap(self, cmap: str) -> None:
        if not self.control_panel or not self.display_widget:
            return
        key = cmap.replace('-', '_').upper()
        try:
            enum = ColormapType[key]
        except KeyError:
            enum = ColormapType.SEISMIC
        self.initial_colormap = enum
        self.control_panel.set_colormap(enum)
        if self.display_widget.has_data():
            self.display_widget.set_colormap(enum)

    def set_default_mode(self, mode: DisplayMode) -> None:
        if not self.control_panel or not self.display_widget:
            return
        self.initial_mode = mode
        self.control_panel.set_display_mode(mode)
        if self.display_widget.has_data():
            self.display_widget.set_display_mode(mode)

    def export_image(self) -> None:
        if not self.display_widget:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
        )
        if not filename:
            return
        try:
            self.display_widget.export_image(pathlib.Path(filename))
            QMessageBox.information(self, "Export", "Image exported successfully")
        except Exception as exc:  # pragma: no cover - user interaction
            QMessageBox.critical(self, "Error", f"Failed to export image:\n{exc}")

    def show_about(self) -> None:
        QMessageBox.information(
            self,
            "About",
            "RadEx-style SEG-Y Viewer\n\n"
            "Interactive seismic viewer with multiple display modes, AGC,\n"
            "colormap controls, and header inspection.",
        )

    # ------------------------------------------------------------------
    # Display callbacks
    # ------------------------------------------------------------------
    def _on_mouse_position(self, trace_idx: int, sample_idx: float) -> None:
        if not self.mouse_pos_label:
            return
        self.mouse_pos_label.setText(f"Trace: {trace_idx}, Sample: {sample_idx:.1f}")

    def _on_trace_selected(self, trace_idx: int) -> None:
        if not self.status_label:
            return
        self.status_label.setText(f"Selected trace: {trace_idx}")

    # ------------------------------------------------------------------
    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background-color: #1e1f23; }
            QToolBar { background-color: #27282d; spacing: 8px; padding: 6px; }
            QLabel { color: #f0f0f0; }
            QStatusBar { background-color: #27282d; color: #f0f0f0; }
            ControlPanel { background-color: #202227; }
            """
        )
