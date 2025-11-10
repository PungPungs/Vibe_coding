"""
Display Control Panel
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QComboBox, QCheckBox, QSlider, QPushButton,
                              QSpinBox, QGroupBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal

from display_modes import DisplayMode
from colormap import ColormapType


class ControlPanel(QWidget):
    """Display control panel"""

    # Signals
    display_mode_changed = pyqtSignal(DisplayMode)
    colormap_changed = pyqtSignal(ColormapType)
    agc_changed = pyqtSignal(bool, int, str)
    clip_changed = pyqtSignal(float)
    wiggle_amplitude_changed = pyqtSignal(float)
    reset_view_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("Display Controls")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)

        # Display Mode Group
        mode_group = self.create_display_mode_group()
        layout.addWidget(mode_group)

        # Colormap Group
        colormap_group = self.create_colormap_group()
        layout.addWidget(colormap_group)

        # Gain Control Group
        gain_group = self.create_gain_control_group()
        layout.addWidget(gain_group)

        # Wiggle Settings Group
        wiggle_group = self.create_wiggle_settings_group()
        layout.addWidget(wiggle_group)

        # View Controls
        view_group = self.create_view_controls_group()
        layout.addWidget(view_group)

        layout.addStretch()

        # Help text
        help_text = QLabel(
            "Keyboard:\n"
            "← → : Smooth scroll\n"
            "↑ ↓ : Pan vertically\n"
            "Home/End : First/Last\n"
            "+/- : Zoom in/out\n"
            "R : Reset view\n\n"
            "Mouse:\n"
            "Wheel : Zoom\n"
            "Right Drag : Pan\n"
            "Left Click : Select trace\n\n"
            "Note: Fits traces to\n"
            "screen automatically"
        )
        help_text.setStyleSheet(
            "padding: 10px; "
            "background-color: #f0f0f0; "
            "border-radius: 5px; "
            "font-size: 10px;"
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

    def create_display_mode_group(self) -> QGroupBox:
        """Create display mode group"""
        group = QGroupBox("Display Mode")
        layout = QVBoxLayout()
        group.setLayout(layout)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Wiggle", DisplayMode.WIGGLE)
        self.mode_combo.addItem("Variable Area", DisplayMode.VARIABLE_AREA)
        self.mode_combo.addItem("Variable Density", DisplayMode.VARIABLE_DENSITY)
        self.mode_combo.addItem("Wiggle + VA", DisplayMode.WIGGLE_VA)
        self.mode_combo.addItem("Wiggle + VD", DisplayMode.WIGGLE_VD)
        self.mode_combo.setCurrentIndex(3)  # Default: Wiggle + VA

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_combo)

        return group

    def create_colormap_group(self) -> QGroupBox:
        """Create colormap group"""
        group = QGroupBox("Colormap (VD mode)")
        layout = QVBoxLayout()
        group.setLayout(layout)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItem("Seismic", ColormapType.SEISMIC)
        self.colormap_combo.addItem("Grayscale", ColormapType.GRAYSCALE)
        self.colormap_combo.addItem("Jet", ColormapType.JET)
        self.colormap_combo.addItem("Viridis", ColormapType.VIRIDIS)
        self.colormap_combo.addItem("Red-White-Blue", ColormapType.RED_WHITE_BLUE)
        self.colormap_combo.addItem("Brown-White-Green", ColormapType.BROWN_WHITE_GREEN)

        self.colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        layout.addWidget(self.colormap_combo)

        return group

    def create_gain_control_group(self) -> QGroupBox:
        """Create gain control group"""
        group = QGroupBox("Gain Control")
        layout = QVBoxLayout()
        group.setLayout(layout)

        # AGC Enable
        self.agc_checkbox = QCheckBox("Enable AGC")
        self.agc_checkbox.stateChanged.connect(self._on_agc_changed)
        layout.addWidget(self.agc_checkbox)

        # Clipping
        clip_layout = QHBoxLayout()
        clip_layout.addWidget(QLabel("Clip %:"))
        self.clip_spin = QDoubleSpinBox()
        self.clip_spin.setRange(90.0, 100.0)
        self.clip_spin.setValue(99.0)
        self.clip_spin.setSingleStep(0.5)
        self.clip_spin.valueChanged.connect(self._on_clip_changed)
        clip_layout.addWidget(self.clip_spin)
        layout.addLayout(clip_layout)

        return group

    def create_wiggle_settings_group(self) -> QGroupBox:
        """Create wiggle settings group"""
        group = QGroupBox("Wiggle Settings")
        layout = QVBoxLayout()
        group.setLayout(layout)

        # Amplitude
        amp_label = QLabel("Amplitude:")
        layout.addWidget(amp_label)

        self.amplitude_slider = QSlider(Qt.Horizontal)
        self.amplitude_slider.setRange(10, 200)
        self.amplitude_slider.setValue(80)
        self.amplitude_slider.valueChanged.connect(self._on_amplitude_changed)
        layout.addWidget(self.amplitude_slider)

        self.amplitude_value_label = QLabel("0.80")
        self.amplitude_value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.amplitude_value_label)

        return group

    def create_view_controls_group(self) -> QGroupBox:
        """Create view controls group"""
        group = QGroupBox("View Controls")
        layout = QVBoxLayout()
        group.setLayout(layout)

        # Reset button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._on_reset_view)
        reset_btn.setStyleSheet(
            "padding: 8px; "
            "background-color: #4CAF50; "
            "color: white; "
            "font-weight: bold;"
        )
        layout.addWidget(reset_btn)

        return group

    def _on_mode_changed(self, index):
        """Handle display mode change"""
        mode = self.mode_combo.itemData(index)
        self.display_mode_changed.emit(mode)

    def _on_colormap_changed(self, index):
        """Handle colormap change"""
        colormap = self.colormap_combo.itemData(index)
        self.colormap_changed.emit(colormap)

    def _on_agc_changed(self):
        """Handle AGC setting change"""
        enabled = self.agc_checkbox.isChecked()
        window = 100  # Fixed window size
        method = 'rms'  # Fixed method
        self.agc_changed.emit(enabled, window, method)

    def _on_clip_changed(self, value):
        """Handle clip percentile change"""
        self.clip_changed.emit(value)

    def _on_amplitude_changed(self, value):
        """Handle wiggle amplitude change"""
        amplitude = value / 100.0
        self.amplitude_value_label.setText(f"{amplitude:.2f}")
        self.wiggle_amplitude_changed.emit(amplitude)

    def _on_reset_view(self):
        """Handle reset view button"""
        self.reset_view_requested.emit()
