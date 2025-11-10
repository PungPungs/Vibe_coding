"""
Advanced OpenGL Display Widget with multiple display modes
"""
import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from OpenGL.GL import *
from OpenGL.GLU import *

from display_modes import DisplayMode, DisplaySettings
from colormap import Colormap, ColormapType
from agc import AGC, GainControl


class SeismicDisplay(QOpenGLWidget):
    """Professional seismic data display widget"""

    # Signals
    mouse_position_changed = pyqtSignal(int, float)  # trace_idx, sample_idx
    trace_selected = pyqtSignal(int)  # trace_idx

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data
        self.raw_data = None
        self.processed_data = None
        self.num_samples = 0
        self.num_traces = 0

        # Display settings
        self.settings = DisplaySettings()

        # Viewport-based rendering (fit screen)
        self.view_offset = 0.0  # Current horizontal offset (can be fractional)
        self.traces_per_screen = 50  # Will be calculated based on screen size
        self.pixels_per_trace = 20  # Approximate pixels per trace for good visibility

        # Colormap
        self.colormap = Colormap.get_colormap(ColormapType.SEISMIC)
        self.texture_data = None
        self.texture_id = None
        self._texture_dirty = True

        # Scaling
        self.current_scale_factor = 1.0

        # Mouse state
        self.last_mouse_pos = None
        self.is_panning = False

        # Selected trace
        self.selected_trace = None

        # Focus
        self.setFocusPolicy(Qt.StrongFocus)

    def set_data(self, data: np.ndarray):
        """Set seismic data"""
        self.raw_data = data
        self.num_samples, self.num_traces = data.shape

        # Calculate traces per screen based on initial widget size
        self._calculate_traces_per_screen()

        # Reset view
        self.view_offset = 0.0
        self.settings.zoom = 1.0
        self.settings.offset_x = 0.0
        self.settings.offset_y = 0.0

        # Initial processing
        self._process_data()

        print(f"[Display] Data set: {self.num_samples} x {self.num_traces}")
        print(f"[Display] Range: [{np.min(data):.2e}, {np.max(data):.2e}]")
        print(f"[Display] Viewport mode: ~{self.traces_per_screen} traces fit on screen")
        print(f"[Display] Use Left/Right arrows to scroll")

        self._texture_dirty = True
        self.update()

    def _calculate_traces_per_screen(self):
        """Calculate how many traces fit on screen"""
        widget_width = max(self.width(), 800)  # Minimum 800px
        # Calculate based on pixels per trace for good visibility
        self.traces_per_screen = max(10, int(widget_width / self.pixels_per_trace))
        print(f"[Display] Screen width: {widget_width}px, traces per screen: {self.traces_per_screen}")

    def _process_data(self):
        """Process data based on current settings"""
        if self.raw_data is None:
            return

        self.processed_data = self.raw_data.copy()

        print(f"[Process] Raw data range: [{np.min(self.raw_data):.2e}, {np.max(self.raw_data):.2e}]")

        # Apply AGC if enabled
        if self.settings.use_agc:
            self.processed_data = AGC.apply(
                self.processed_data,
                self.settings.agc_window,
                self.settings.agc_method
            )
            print(f"[Process] After AGC: [{np.min(self.processed_data):.2e}, {np.max(self.processed_data):.2e}]")

        # Apply clipping
        if self.settings.clip_percentile < 100:
            percentile = self.settings.clip_percentile
            threshold = np.percentile(np.abs(self.processed_data), percentile)
            self.processed_data = np.clip(self.processed_data, -threshold, threshold)
            print(f"[Process] After clipping ({percentile}%): threshold = {threshold:.2e}")

        print(f"[Process] Final data range: [{np.min(self.processed_data):.2e}, {np.max(self.processed_data):.2e}]")

        # Prepare texture for Variable Density mode
        if self.settings.mode in [DisplayMode.VARIABLE_DENSITY, DisplayMode.WIGGLE_VD]:
            self._prepare_texture()

    def _get_visible_traces(self) -> tuple:
        """Get currently visible trace range"""
        # Start trace (can be fractional for smooth scrolling)
        start_trace = self.view_offset
        # End trace
        end_trace = min(start_trace + self.traces_per_screen, self.num_traces)

        # Get integer indices with some buffer for smooth scrolling
        start_idx = max(0, int(np.floor(start_trace)))
        end_idx = min(self.num_traces, int(np.ceil(end_trace)) + 1)

        return start_idx, end_idx, start_trace

    def _current_slice(self) -> np.ndarray:
        """Get current visible slice of data"""
        start_idx, end_idx, _ = self._get_visible_traces()
        return self.processed_data[:, start_idx:end_idx]

    def _prepare_texture(self):
        """Prepare texture data for Variable Density display"""
        # Get current slice
        visible_data = self._current_slice()

        # Apply colormap
        rgb_image = Colormap.apply_colormap(visible_data, self.colormap)

        # Store texture data
        self.texture_data = rgb_image

    def initializeGL(self):
        """Initialize OpenGL"""
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Create texture
        self.texture_id = glGenTextures(1)

    def resizeGL(self, w, h):
        """Handle resize"""
        glViewport(0, 0, w, h)
        # Recalculate traces per screen when window is resized
        if self.num_traces > 0:
            self._calculate_traces_per_screen()
            self._texture_dirty = True

    def paintGL(self):
        """Render scene"""
        glClear(GL_COLOR_BUFFER_BIT)

        if self.processed_data is None:
            return

        # Setup projection based on viewport
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Get visible trace range
        start_idx, end_idx, start_trace = self._get_visible_traces()
        visible_traces = self.traces_per_screen

        if visible_traces <= 0:
            return

        # Coordinate system: x=[0, traces_per_screen], y=[0, num_samples]
        # This creates a viewport that fits the screen
        gluOrtho2D(0, visible_traces, self.num_samples, 0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Account for fractional offset (smooth scrolling)
        # Shift by the fractional part of the start position
        fractional_offset = start_trace - start_idx
        glTranslatef(-fractional_offset, 0.0, 0.0)

        # Apply view transforms
        glTranslatef(self.settings.offset_x, self.settings.offset_y, 0.0)
        glScalef(self.settings.zoom, self.settings.zoom, 1.0)

        # Render based on display mode
        mode = self.settings.mode

        if mode == DisplayMode.VARIABLE_DENSITY:
            self._render_variable_density()

        elif mode == DisplayMode.WIGGLE:
            self._render_wiggle()

        elif mode == DisplayMode.VARIABLE_AREA:
            self._render_variable_area()

        elif mode == DisplayMode.WIGGLE_VA:
            self._render_variable_area()
            self._render_wiggle()

        elif mode == DisplayMode.WIGGLE_VD:
            self._render_variable_density()
            self._render_wiggle()

        # Render selected trace highlight
        if self.selected_trace is not None:
            self._render_selected_trace()

        glFlush()

    def _render_variable_density(self):
        """Render Variable Density display using texture"""
        if self.texture_data is None:
            self._prepare_texture()

        if self.texture_data is None:
            return

        # Bind texture
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # Upload texture
        h, w, _ = self.texture_data.shape
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                     GL_RGB, GL_FLOAT, self.texture_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Draw textured quad covering the viewport
        glColor4f(1.0, 1.0, 1.0, self.settings.vd_alpha)

        # The texture spans the full number of visible traces in the slice
        start_idx, end_idx, _ = self._get_visible_traces()
        num_texture_traces = end_idx - start_idx

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(num_texture_traces, 0)
        glTexCoord2f(1, 1); glVertex2f(num_texture_traces, self.num_samples)
        glTexCoord2f(0, 1); glVertex2f(0, self.num_samples)
        glEnd()

        glDisable(GL_TEXTURE_2D)

    def _render_wiggle(self):
        """Render wiggle traces (slice-based)"""
        # Get current slice
        visible_data = self._current_slice()
        num_vis = visible_data.shape[1]

        # Calculate scale factor once for all traces
        rms = np.sqrt(np.mean(visible_data ** 2))
        if rms < 1e-10:
            max_val = np.max(np.abs(visible_data))
            if max_val > 1e-10:
                self.current_scale_factor = max_val
            else:
                self.current_scale_factor = 1.0
        else:
            self.current_scale_factor = rms * 3.0  # 3 sigma

        glColor3f(*self.settings.wiggle_color)
        glLineWidth(self.settings.wiggle_line_width)

        # Render each trace in the slice
        for i in range(num_vis):
            trace_data = visible_data[:, i]
            self._draw_wiggle_trace(i, trace_data)

    def _draw_wiggle_trace(self, display_idx: int, trace_data: np.ndarray):
        """Draw single wiggle trace"""
        # Use pre-calculated scale factor
        scale = self.settings.wiggle_amplitude / self.current_scale_factor

        glBegin(GL_LINE_STRIP)
        for sample_idx in range(self.num_samples):
            value = trace_data[sample_idx] * scale
            x = display_idx + value
            y = sample_idx
            glVertex2f(x, y)
        glEnd()

    def _render_variable_area(self):
        """Render Variable Area (filled wiggles) - slice-based"""
        # Get current slice
        visible_data = self._current_slice()
        num_vis = visible_data.shape[1]

        # Calculate scale factor once for all traces (same as wiggle)
        rms = np.sqrt(np.mean(visible_data ** 2))
        if rms < 1e-10:
            max_val = np.max(np.abs(visible_data))
            if max_val > 1e-10:
                self.current_scale_factor = max_val
            else:
                self.current_scale_factor = 1.0
        else:
            self.current_scale_factor = rms * 3.0

        glColor4f(*self.settings.va_color)

        # Render each trace in the slice
        for i in range(num_vis):
            trace_data = visible_data[:, i]
            self._draw_variable_area(i, trace_data)

    def _draw_variable_area(self, display_idx: int, trace_data: np.ndarray):
        """Draw variable area for single trace"""
        # Use pre-calculated scale factor
        scale = self.settings.wiggle_amplitude / self.current_scale_factor

        glBegin(GL_TRIANGLES)
        for sample_idx in range(self.num_samples - 1):
            value1 = trace_data[sample_idx] * scale
            value2 = trace_data[sample_idx + 1] * scale

            # Fill positive area
            if value1 > 0 or value2 > 0:
                x1 = display_idx + max(0, value1)
                x2 = display_idx + max(0, value2)

                glVertex2f(display_idx, sample_idx)
                glVertex2f(x1, sample_idx)
                glVertex2f(display_idx, sample_idx + 1)

                glVertex2f(display_idx, sample_idx + 1)
                glVertex2f(x1, sample_idx)
                glVertex2f(x2, sample_idx + 1)
        glEnd()

    def _render_selected_trace(self):
        """Highlight selected trace"""
        if self.selected_trace is None:
            return

        # Calculate display position relative to viewport
        start_idx, end_idx, start_trace = self._get_visible_traces()

        # Check if selected trace is in the visible range
        if start_idx <= self.selected_trace < end_idx:
            # Display index is relative to start_idx
            display_idx = self.selected_trace - start_idx

            glColor4f(0.0, 0.0, 1.0, 0.3)
            glLineWidth(3.0)

            glBegin(GL_LINES)
            glVertex2f(display_idx + 0.5, 0)
            glVertex2f(display_idx + 0.5, self.num_samples)
            glEnd()

    def wheelEvent(self, event):
        """Mouse wheel - zoom"""
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9

        self.settings.zoom *= zoom_factor
        self.settings.zoom = max(0.1, min(10.0, self.settings.zoom))

        self.update()

    def mousePressEvent(self, event):
        """Mouse press"""
        self.last_mouse_pos = event.pos()

        if event.button() == Qt.LeftButton:
            # Select trace
            trace_idx = self._screen_to_trace(event.x())
            if trace_idx is not None:
                self.selected_trace = trace_idx
                self.trace_selected.emit(trace_idx)
                self.update()

        elif event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
            self.is_panning = True

    def mouseMoveEvent(self, event):
        """Mouse move"""
        # Update position
        trace_idx, sample_idx = self._screen_to_data(event.x(), event.y())
        if trace_idx is not None and sample_idx is not None:
            self.mouse_position_changed.emit(trace_idx, sample_idx)

        # Panning
        if self.is_panning and self.last_mouse_pos is not None:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()

            self.settings.offset_x += dx * 0.1 / self.settings.zoom
            self.settings.offset_y += dy * 0.1 / self.settings.zoom

            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """Mouse release"""
        if event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.last_mouse_pos = None

    def keyPressEvent(self, event):
        """Keyboard navigation - viewport-based smooth scrolling"""

        if event.key() == Qt.Key_Left:
            # Smooth scroll left by a few traces
            scroll_amount = max(1, self.traces_per_screen // 10)  # 10% of screen
            new_offset = max(0, self.view_offset - scroll_amount)
            if new_offset != self.view_offset:
                self.view_offset = new_offset
                self._texture_dirty = True
                self.update()
                print(f"[Nav] Viewing traces {int(self.view_offset) + 1} - {int(min(self.view_offset + self.traces_per_screen, self.num_traces))} / {self.num_traces}")

        elif event.key() == Qt.Key_Right:
            # Smooth scroll right by a few traces
            scroll_amount = max(1, self.traces_per_screen // 10)  # 10% of screen
            max_offset = max(0, self.num_traces - self.traces_per_screen)
            new_offset = min(max_offset, self.view_offset + scroll_amount)
            if new_offset != self.view_offset:
                self.view_offset = new_offset
                self._texture_dirty = True
                self.update()
                print(f"[Nav] Viewing traces {int(self.view_offset) + 1} - {int(min(self.view_offset + self.traces_per_screen, self.num_traces))} / {self.num_traces}")

        elif event.key() == Qt.Key_Up:
            # Pan up
            self.settings.offset_y += 10.0 / self.settings.zoom
            self.update()

        elif event.key() == Qt.Key_Down:
            # Pan down
            self.settings.offset_y -= 10.0 / self.settings.zoom
            self.update()

        elif event.key() == Qt.Key_Home:
            # Jump to first traces
            if self.view_offset != 0:
                self.view_offset = 0
                self._texture_dirty = True
                self.update()
                print(f"[Nav] Jump to start: traces 1 - {min(self.traces_per_screen, self.num_traces)} / {self.num_traces}")

        elif event.key() == Qt.Key_End:
            # Jump to last traces
            max_offset = max(0, self.num_traces - self.traces_per_screen)
            if self.view_offset != max_offset:
                self.view_offset = max_offset
                self._texture_dirty = True
                self.update()
                print(f"[Nav] Jump to end: traces {int(self.view_offset) + 1} - {self.num_traces} / {self.num_traces}")

        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            # Zoom in
            self.settings.zoom *= 1.2
            self.settings.zoom = min(10.0, self.settings.zoom)
            self.update()

        elif event.key() == Qt.Key_Minus:
            # Zoom out
            self.settings.zoom /= 1.2
            self.settings.zoom = max(0.1, self.settings.zoom)
            self.update()

        elif event.key() == Qt.Key_R:
            self.reset_view()


    def _screen_to_data(self, screen_x: int, screen_y: int) -> tuple:
        """Convert screen coords to data coords"""
        if self.num_traces == 0 or self.num_samples == 0:
            return (None, None)

        norm_x = screen_x / self.width()
        norm_y = screen_y / self.height()

        # Account for viewport-based rendering
        visible_traces = self.traces_per_screen

        # Calculate position within viewport
        trace_idx = int((norm_x * visible_traces - self.settings.offset_x) / self.settings.zoom)
        sample_idx = (norm_y * self.num_samples - self.settings.offset_y) / self.settings.zoom

        # Add the viewport offset to get actual trace index
        trace_idx += int(self.view_offset)

        if 0 <= trace_idx < self.num_traces and 0 <= sample_idx < self.num_samples:
            return (trace_idx, sample_idx)

        return (None, None)

    def _screen_to_trace(self, screen_x: int) -> int:
        """Convert screen x to trace index"""
        trace_idx, _ = self._screen_to_data(screen_x, self.height() // 2)
        return trace_idx

    def reset_view(self):
        """Reset view to initial state"""
        self.settings.zoom = 1.0
        self.settings.offset_x = 0.0
        self.settings.offset_y = 0.0
        self.view_offset = 0
        self._texture_dirty = True
        self.update()
        print(f"[Reset] View reset, showing traces 1 - {min(self.traces_per_screen, self.num_traces)}")

    def set_display_mode(self, mode: DisplayMode):
        """Set display mode"""
        self.settings.mode = mode
        self._process_data()
        self.update()

    def set_colormap(self, colormap_type: ColormapType):
        """Set colormap"""
        self.colormap = Colormap.get_colormap(colormap_type)
        self._prepare_texture()
        self.update()

    def set_agc(self, enabled: bool, window: int = 100, method: str = 'rms'):
        """Set AGC parameters"""
        self.settings.use_agc = enabled
        self.settings.agc_window = window
        self.settings.agc_method = method
        self._process_data()
        self.update()

    def set_clip_percentile(self, percentile: float):
        """Set clipping percentile"""
        self.settings.clip_percentile = percentile
        self._process_data()
        self.update()

    def set_wiggle_amplitude(self, amplitude: float):
        """Set wiggle amplitude"""
        self.settings.wiggle_amplitude = amplitude
        self.update()
