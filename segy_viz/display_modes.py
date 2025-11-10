"""
Display mode definitions for seismic data
"""
from enum import Enum


class DisplayMode(Enum):
    """Display modes for seismic data"""
    WIGGLE = "Wiggle"
    VARIABLE_AREA = "Variable Area"
    VARIABLE_DENSITY = "Variable Density"
    WIGGLE_VA = "Wiggle + VA"
    WIGGLE_VD = "Wiggle + VD"


class DisplaySettings:
    """Display settings container"""

    def __init__(self):
        # Display mode
        self.mode = DisplayMode.WIGGLE_VA

        # Wiggle settings
        self.wiggle_amplitude = 1.5  # Amplitude scaling (increased for visibility)
        self.wiggle_color = (0.0, 0.0, 0.0)  # Black
        self.wiggle_line_width = 1.0

        # Variable Area settings
        self.va_color = (0.0, 0.0, 0.0, 0.3)  # Semi-transparent black
        self.va_polarity = 'positive'  # 'positive', 'negative', 'both'

        # Variable Density settings
        self.vd_colormap = 'Seismic'
        self.vd_alpha = 1.0

        # Gain control
        self.use_agc = False
        self.agc_window = 100  # samples
        self.agc_method = 'rms'  # 'rms' or 'mean'

        # Clipping
        self.clip_percentile = 99.0  # Clip at 99th percentile

        # Trace display range
        self.trace_start = 0
        self.trace_window = 100  # Number of traces to display

        # Zoom and pan
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

    def reset(self):
        """Reset to default settings"""
        self.__init__()
