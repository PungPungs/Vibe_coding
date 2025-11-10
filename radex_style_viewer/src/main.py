"""
RadExPro-style Professional SEG-Y Viewer
Main Application Window
"""
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QFileDialog, QLabel, QStatusBar,
                              QToolBar, QAction, QMessageBox, QSplitter,
                              QDockWidget)
from PyQt5.QtCore import Qt

from segy_loader import SegyLoader
from gl_display import SeismicDisplay
from header_viewer import HeaderViewer
from control_panel import ControlPanel


class MainWindow(QMainWindow):
    """Professional SEG-Y Viewer Main Window"""

    def __init__(self):
        super().__init__()

        # SEG-Y Loader
        self.loader = SegyLoader()

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle('RadExPro-Style SEG-Y Viewer')
        self.setGeometry(50, 50, 1600, 900)

        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Main splitter (horizontal)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Left: Control Panel
        self.control_panel = ControlPanel()
        self.control_panel.setMaximumWidth(350)
        main_splitter.addWidget(self.control_panel)

        # Center: Display Widget
        self.display_widget = SeismicDisplay()
        self.display_widget.setMinimumWidth(800)
        main_splitter.addWidget(self.display_widget)

        # Right: Header Viewer (as dock widget)
        self.header_viewer = HeaderViewer()
        header_dock = QDockWidget("Header Information", self)
        header_dock.setWidget(self.header_viewer)
        header_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, header_dock)

        # Set splitter sizes
        main_splitter.setSizes([300, 1100])

        # Create toolbar
        self.create_toolbar()

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel('Ready')
        self.status_bar.addWidget(self.status_label)

        # File info label
        self.info_label = QLabel('')
        self.status_bar.addPermanentWidget(self.info_label)

        # Mouse position label
        self.mouse_pos_label = QLabel('Trace: --, Sample: --')
        self.status_bar.addPermanentWidget(self.mouse_pos_label)

        # Connect signals
        self.connect_signals()

        # Apply stylesheet
        self.apply_stylesheet()

    def create_toolbar(self):
        """Create toolbar"""
        toolbar = QToolBar('Main Toolbar')
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Open File
        open_action = QAction('Open SEG-Y', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)

        toolbar.addSeparator()

        # Export Image
        export_action = QAction('Export Image', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_image)
        toolbar.addAction(export_action)

        toolbar.addSeparator()

        # About
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        toolbar.addAction(about_action)

    def connect_signals(self):
        """Connect widget signals"""
        # Control panel signals
        self.control_panel.display_mode_changed.connect(
            self.display_widget.set_display_mode
        )
        self.control_panel.colormap_changed.connect(
            self.display_widget.set_colormap
        )
        self.control_panel.agc_changed.connect(
            self.display_widget.set_agc
        )
        self.control_panel.clip_changed.connect(
            self.display_widget.set_clip_percentile
        )
        self.control_panel.wiggle_amplitude_changed.connect(
            self.display_widget.set_wiggle_amplitude
        )
        self.control_panel.reset_view_requested.connect(
            self.display_widget.reset_view
        )

        # Display widget signals
        self.display_widget.mouse_position_changed.connect(
            self.on_mouse_position_changed
        )
        self.display_widget.trace_selected.connect(
            self.on_trace_selected
        )

    def open_file(self):
        """Open SEG-Y file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            'Open SEG-Y File',
            '',
            'SEG-Y Files (*.sgy *.segy);;All Files (*)'
        )

        if filename:
            self.status_label.setText('Loading SEG-Y file...')
            QApplication.processEvents()

            success = self.loader.load(filename)

            if success:
                # Get data
                data = self.loader.get_data()
                num_samples, num_traces = self.loader.get_dimensions()
                sample_rate = self.loader.get_sample_rate()

                # Set data to display
                self.display_widget.set_data(data)

                # Update header viewer
                self.header_viewer.set_loader(self.loader)

                # Update info
                self.info_label.setText(
                    f'Traces: {num_traces} | '
                    f'Samples: {num_samples} | '
                    f'Rate: {sample_rate * 1000:.2f} ms'
                )

                self.status_label.setText(f'Loaded: {filename}')
                QMessageBox.information(
                    self,
                    'Success',
                    f'SEG-Y file loaded successfully\n\n'
                    f'Traces: {num_traces}\n'
                    f'Samples: {num_samples}\n'
                    f'Sample Rate: {sample_rate * 1000:.2f} ms'
                )
            else:
                QMessageBox.critical(self, 'Error', 'Failed to load SEG-Y file')
                self.status_label.setText('Failed to load file')

    def export_image(self):
        """Export display as image"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Export Image',
            '',
            'PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)'
        )

        if filename:
            # Grab OpenGL widget frame
            pixmap = self.display_widget.grab()
            success = pixmap.save(filename)

            if success:
                self.status_label.setText(f'Image exported: {filename}')
                QMessageBox.information(self, 'Success', 'Image exported successfully')
            else:
                QMessageBox.critical(self, 'Error', 'Failed to export image')

    def show_about(self):
        """Show about dialog"""
        about_text = """
<h2>RadExPro-Style SEG-Y Viewer</h2>

<p><b>Version 1.0</b></p>

<p>Professional SEG-Y data visualization tool inspired by RadExPro.</p>

<h3>Features:</h3>
<ul>
<li>Multiple Display Modes (Wiggle, Variable Area, Variable Density)</li>
<li>AGC (Automatic Gain Control)</li>
<li>Multiple Colormaps</li>
<li>Header Information Viewer</li>
<li>Interactive Navigation</li>
<li>Professional UI Layout</li>
</ul>

<h3>Controls:</h3>
<ul>
<li><b>Mouse Wheel:</b> Zoom in/out</li>
<li><b>Right-Click Drag:</b> Pan view</li>
<li><b>Left Click:</b> Select trace</li>
<li><b>Left/Right Arrows:</b> Smooth scroll through traces</li>
<li><b>Up/Down Arrows:</b> Pan vertically</li>
<li><b>+/-:</b> Zoom in/out</li>
<li><b>Home/End:</b> Jump to first/last position</li>
<li><b>R:</b> Reset view completely</li>
</ul>

<p><b>Performance:</b> Dynamically fits traces to screen width for optimal viewing. Use arrow keys to smoothly scroll through data.</p>

<p><b>Built with:</b> Python, PyQt5, OpenGL, NumPy</p>
        """
        QMessageBox.about(self, 'About', about_text)

    def on_mouse_position_changed(self, trace_idx: int, sample_idx: float):
        """Update mouse position display"""
        self.mouse_pos_label.setText(f'Trace: {trace_idx}, Sample: {sample_idx:.1f}')

    def on_trace_selected(self, trace_idx: int):
        """Handle trace selection"""
        self.header_viewer.show_trace_header(trace_idx)
        self.status_label.setText(f'Trace {trace_idx} selected')

    def apply_stylesheet(self):
        """Apply application stylesheet"""
        stylesheet = """
        QMainWindow {
            background-color: #f0f0f0;
        }
        QToolBar {
            background-color: #e0e0e0;
            border: 1px solid #c0c0c0;
            padding: 5px;
            spacing: 5px;
        }
        QToolBar QToolButton {
            background-color: #ffffff;
            border: 1px solid #c0c0c0;
            padding: 5px 10px;
            margin: 2px;
            border-radius: 3px;
        }
        QToolBar QToolButton:hover {
            background-color: #e8f4f8;
            border: 1px solid #0078d7;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #c0c0c0;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            padding: 5px 15px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #e8f4f8;
        }
        QStatusBar {
            background-color: #e0e0e0;
            border-top: 1px solid #c0c0c0;
        }
        """
        self.setStyleSheet(stylesheet)

    def closeEvent(self, event):
        """Handle window close"""
        self.loader.close()
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName('RadExPro-Style SEG-Y Viewer')

    # Set application style
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
